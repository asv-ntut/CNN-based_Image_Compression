import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder

# ==========================================================
# 兼容模式：自動判斷是直接執行還是作為套件導入
# ==========================================================
try:
    from .utils import conv, update_registered_buffers
except ImportError:
    from utils import conv, update_registered_buffers

# ==========================================================
# Helper Functions
# ==========================================================
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    """
    Transposed convolution for upsampling.
    Uses Conv2d + PixelShuffle for Vitis AI compatibility.
    """
    internal_channels = out_channels * (stride ** 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            internal_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
        nn.PixelShuffle(upscale_factor=stride)
    )

# ==========================================================
# TIC: 通用平均-尺度超先驗模型 (Universal Mean-Scale Hyperprior)
# ==========================================================
class TIC(nn.Module):
    """
    Mean-Scale Hyperprior Image Compression Model (Unified).
    
    Architecture:
    - g_a: Encoder (Spatial -> Latent y)
    - g_s: Decoder (Latent y_hat -> Reconstruction x_hat)
    - h_a: Hyper Encoder (Latent y -> Hyper Latent z)
    - h_s: Hyper Decoder (Hyper Latent z_hat -> Gaussian Params sigma, mu)

    Arguments:
        N (int): Number of channels in the hidden layers (Transformation features).
                 - Teacher: 128
                 - Student: 64
                 - Tiny: 32
        M (int): Number of channels in the latent space (Compressed representation).
                 - Default: 192 (Shared across distillation to allow feature alignment)
    """

    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = N
        self.M = M

        # ============================================
        # g_a: Analysis Transform (Encoder)
        # ============================================
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),      # conv Mx5x5/2↓
        )

        # ============================================
        # g_s: Synthesis Transform (Decoder)
        # ============================================
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            GDN(N, inverse=True),                      
            deconv(N, N, kernel_size=5, stride=2),    
            GDN(N, inverse=True),                      
            deconv(N, N, kernel_size=5, stride=2),    
            GDN(N, inverse=True),                      
            deconv(N, 3, kernel_size=5, stride=2),    # deconv 3x5x5/2↑
        )

        # ============================================
        # h_a: Hyper Analysis Transform
        # ============================================
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),      # conv Nx3x3/1
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
        )

        # ============================================
        # h_s: Hyper Synthesis Transform
        # ============================================
        self.h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            nn.ReLU(inplace=True),
            conv(N, M * 2, kernel_size=3, stride=1),  # conv 2Mx3x3/1 (scales + means)
        )

        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        # Encoder: x → y
        y = self.g_a(x)
        
        # Hyper encoder: |y| → z
        z = self.h_a(torch.abs(y))
        
        # Entropy bottleneck for z
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # Hyper decoder: ẑ → (scales, means)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        # Quantize y with Mean-Scale Gaussian model
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        
        # Decoder: ŷ → x̂
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_hat": y_hat, # 必要：用於 Feature Distillation (MSE Loss)
            "z_hat": z_hat, # 建議：保留以供 debug 或進階用途
        }

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck module(s)."""
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values."""
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # Infer N and M from state dict automatically
        # g_a.0.weight shape is [N, 3, 5, 5]
        # g_a.6.weight shape is [M, N, 5, 5]
        try:
            N = state_dict["g_a.0.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
        except KeyError:
            # Fallback for some checkpoint variations
            N = state_dict.get("g_a.0.weight").size(0)
            M = 192 # Default fallback
            
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        
        return {"x_hat": x_hat}

# ==========================================================
# Factory Function (工廠模式)
# ==========================================================
def build_model(model_name, **kwargs):
    """
    Factory function to create TIC models with specific configurations.
    
    Presets:
    - 'tic': N=128, M=192 (Teacher Standard)
    - 'tic_student': N=64, M=192 (Student Standard)
    - 'tic_tiny': N=32, M=192 (Edge/Mobile)
    
    You can override N or M by passing them as kwargs.
    Example: build_model("tic_tiny", M=320)
    """
    # 預設配置表
    configs = {
        "tic":         {"N": 128, "M": 192},
        "tic_student": {"N": 64,  "M": 192},
        "tic_tiny":    {"N": 32,  "M": 192},
    }
    
    if model_name not in configs:
        raise ValueError(f"Model {model_name} not found. Available: {list(configs.keys())}")
    
    # 取得預設參數
    params = configs[model_name].copy()
    
    # 使用者傳入的參數 (kwargs) 優先權最高，覆蓋預設值
    params.update(kwargs)
    
    return TIC(N=params["N"], M=params["M"])