"""
ONNX-based Satellite Image Decompression Tool (PC Multiprocessing Optimized)

Optimizations:
1. Multiprocessing: Utilizes all CPU cores on Ubuntu PC.
2. Distributed IO: Workers read .bin files independently.
3. Fast Reassembly: Main process aggregates patches into the full canvas.
4. Correct Metrics: PSNR/SSIM are calculated on raw reconstruction (before brightness adjust).
5. **Dynamic CDFs + 2-Way Interleaved C++ ANS**: 對應極速編碼端的雙路動態解碼引擎。

Usage:
    python decompress.py ./output_onnx/hualien_tile_r0_c0 --dec decoder.onnx --hyper hyper.onnx --workers 16 --brightness 1.5
"""
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
import glob
import time
import struct
import zlib
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageEnhance
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import CompressAI entropy modules
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# 🚀 引入我們編譯好的 C++ 雙路交錯 ANS 引擎
from compressai import ans 

# Try to import MS-SSIM
try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False

# ==============================================================================
# 0. Global Setup for Workers
# ==============================================================================
worker_dec_sess = None
worker_hyper_sess = None
worker_entropy_models = None
worker_fixed_cdfs = None  # 用於儲存全域的 CDF 表格

def init_worker(dec_path, hyper_path, use_cuda, cdf_module_path):
    """Initializer function for each worker process."""
    global worker_dec_sess, worker_hyper_sess, worker_entropy_models, worker_fixed_cdfs
    
    if cdf_module_path not in sys.path:
        sys.path.insert(0, cdf_module_path)
    
    try:
        import fixed_cdfs
        import importlib
        importlib.reload(fixed_cdfs)
        worker_fixed_cdfs = fixed_cdfs
        HAS_FIXED_CDFS = True
    except ImportError:
        HAS_FIXED_CDFS = False
        print(f"[Worker {os.getpid()}] ⚠️ Cannot find fixed_cdfs.py")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1 
    sess_options.inter_op_num_threads = 1
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    
    try:
        worker_dec_sess = ort.InferenceSession(dec_path, sess_options=sess_options, providers=providers)
        worker_hyper_sess = ort.InferenceSession(hyper_path, sess_options=sess_options, providers=providers)
    except Exception as e:
        print(f"[Worker {os.getpid()}] ❌ Failed to load ONNX models: {e}")
        return

    if HAS_FIXED_CDFS:
        N = len(worker_fixed_cdfs.FIXED_EB_MEDIANS)
        entropy_bottleneck = EntropyBottleneck(N)
        gaussian_conditional = GaussianConditional(None)
        
        device = torch.device("cpu")
        # 初始化 EntropyBottleneck (Z) 所需的參數
        entropy_bottleneck._quantized_cdf.resize_(torch.tensor(worker_fixed_cdfs.FIXED_EB_CDF).shape).copy_(
            torch.tensor(worker_fixed_cdfs.FIXED_EB_CDF, device=device, dtype=torch.int32))
        entropy_bottleneck._offset.resize_(torch.tensor(worker_fixed_cdfs.FIXED_EB_OFFSET).shape).copy_(
            torch.tensor(worker_fixed_cdfs.FIXED_EB_OFFSET, device=device, dtype=torch.int32))
        entropy_bottleneck._cdf_length.resize_(torch.tensor(worker_fixed_cdfs.FIXED_EB_LENGTH).shape).copy_(
            torch.tensor(worker_fixed_cdfs.FIXED_EB_LENGTH, device=device, dtype=torch.int32))
        entropy_bottleneck.quantiles.data[:, 0, 1] = torch.tensor(worker_fixed_cdfs.FIXED_EB_MEDIANS, device=device).squeeze()

        # GaussianConditional (Y) 只需要 scale_table 即可 (CDF 由 C++ 接手)
        gaussian_conditional.scale_table = torch.tensor(worker_fixed_cdfs.FIXED_GC_SCALE_TABLE, device=device)
        
        worker_entropy_models = (entropy_bottleneck, gaussian_conditional)
    else:
        worker_entropy_models = None

# ==============================================================================
# 1. Packet Parsing & Worker Logic
# ==============================================================================
def load_satellite_packet(bin_path):
    try:
        with open(bin_path, "rb") as f:
            data = f.read()
        if len(data) < 22: return None
        
        received_crc = struct.unpack('<I', data[-4:])[0]
        content = data[:-4]
        if received_crc != (zlib.crc32(content) & 0xffffffff):
            return None 

        header_size = 18
        magic, img_id, row, col, h, w, len_y, len_z = struct.unpack('<3sBBBHHII', data[:header_size])
        if magic != b'TIC': return None
        
        y_str = data[header_size : header_size + len_y]
        z_str = data[header_size + len_y : header_size + len_y + len_z]
        
        return {"row": row, "col": col, "img_id": img_id, "strings": [[y_str], [z_str]], "shape": (h, w)}
    except:
        return None

def process_file_task(bin_path, target_id=None):
    global worker_dec_sess, worker_hyper_sess, worker_entropy_models, worker_fixed_cdfs
    
    if worker_entropy_models is None: return None 
    packet = load_satellite_packet(bin_path)
    if packet is None: return None
    if target_id is not None and packet['img_id'] != target_id: return None

    eb, gc = worker_entropy_models
    strings, shape = packet["strings"], packet["shape"]
    
    try:
        # 1. 傳統解碼 Z 特徵
        z_hat = eb.decompress(strings[1], shape)
        z_hat_np = z_hat.numpy()
        
        # 2. HyperDecoder 推論出 scales 與 means
        scales_np, means_np = worker_hyper_sess.run(None, {"z_hat": z_hat_np})
        scales_t = torch.from_numpy(scales_np)
        means_t = torch.from_numpy(means_np)
        
        # 3. 根據 scales 建立動態 indexes
        indexes = gc.build_indexes(scales_t).to(torch.int32)
        idx_flat = indexes.flatten().tolist()
        
        # 4. 🚀 使用我們自訂的 C++ 雙路動態引擎解碼 Y 特徵
        decoder = ans.DynamicRansDecoder2Way()
        sym_flat = decoder.decode_with_indexes(
            strings[0][0],   # Y 的位元流 (bytes)
            idx_flat,        # 攤平的動態索引
            worker_fixed_cdfs.FIXED_GC_CDF,
            worker_fixed_cdfs.FIXED_GC_LENGTH,
            worker_fixed_cdfs.FIXED_GC_OFFSET
        )
        
        # 5. 反量化：將 C++ 吐出的 Symbol 加上 means 還原為 y_hat
        symbols_t = torch.tensor(sym_flat, dtype=torch.float32).reshape(means_t.shape)
        y_hat = symbols_t + means_t
        
        # 6. Main Decoder 推論還原影像
        x_hat_np = worker_dec_sess.run(None, {"y_hat": y_hat.numpy()})[0]
        
        x_hat = torch.from_numpy(x_hat_np).clamp(0, 1)
        target_h, target_w = 256, 256
        if x_hat.size(2) != target_h or x_hat.size(3) != target_w:
            x_hat = x_hat[:, :, :target_h, :target_w]
            
        img_np = (x_hat.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return (packet['row'], packet['col'], img_np)
        
    except Exception as e:
        print(f"Error processing {bin_path}: {e}")
        return None

# ==============================================================================
# 2. Metrics (Main Process)
# ==============================================================================
from torchvision import transforms
def calculate_metrics(recon_img, original_path):
    if not os.path.exists(original_path):
        print(f"⚠️ Original image not found: {original_path}")
        return

    print("Calculating PSNR/SSIM on RAW reconstruction...")
    try:
        if original_path.lower().endswith(('.tif', '.tiff')):
            import tifffile
            gt = tifffile.imread(original_path)
            if gt.ndim == 3 and gt.shape[2] <= 4: gt = np.transpose(gt, (2, 0, 1))
        else:
            gt = np.array(Image.open(original_path).convert("RGB")).transpose(2, 0, 1)
            
        gt = gt.astype(np.float32) / (255.0 if gt.dtype == np.uint8 else 1.0)
        gt = torch.from_numpy(np.clip(gt[:3], 0, 1))
        
        rec = transforms.ToTensor()(recon_img)
        
        h, w = min(gt.shape[1], rec.shape[1]), min(gt.shape[2], rec.shape[2])
        gt, rec = gt[:, :h, :w], rec[:, :h, :w]
        
        mse = torch.mean((gt - rec) ** 2)
        psnr = -10 * torch.log10(mse)
        print(f"✅ PSNR:    {psnr:.4f} dB")
        
        if HAS_MSSSIM:
            val_msssim = ms_ssim(gt.unsqueeze(0), rec.unsqueeze(0), data_range=1.0).item()
            print(f"✅ MS-SSIM: {val_msssim:.4f}")
            
    except Exception as e:
        print(f"❌ Metric calculation failed: {e}")

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PC Parallel Satellite Decompression")
    parser.add_argument("bin_dir", type=str, help="Directory containing .bin files")
    parser.add_argument("--dec", type=str, required=True, help="Decoder ONNX file")
    parser.add_argument("--hyper", type=str, required=True, help="HyperDecoder ONNX file")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of CPU workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--original", type=str, default=None, help="Original image for metrics")
    parser.add_argument("--target_id", type=int, default=None, help="Filter Image ID")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output filename")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness enhancement factor (Default: 1.0)")
    
    args = parser.parse_args()

    bin_files = glob.glob(os.path.join(args.bin_dir, "*.bin"))
    if not bin_files:
        print(f"No .bin files found in {args.bin_dir}")
        return

    print(f"🚀 Starting Decompression on {args.workers} workers...")
    
    start_time = time.time()
    patches = {}
    max_row, max_col = 0, 0
    
    with ProcessPoolExecutor(max_workers=args.workers, 
                             initializer=init_worker, 
                             initargs=(args.dec, args.hyper, not args.cpu, os.getcwd())) as executor:
        
        futures = {executor.submit(process_file_task, f, args.target_id): f for f in bin_files}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(f"Progress: {completed}/{len(bin_files)}", end='\r')
            if result:
                r, c, img_np = result
                patches[(r, c)] = Image.fromarray(img_np)
                max_row = max(max_row, r)
                max_col = max(max_col, c)

    print(f"\n✅ All tasks completed. Assembling image...")
    
    if not patches:
        print("❌ No valid patches decoded.")
        return

    PATCH_SIZE = 256
    canvas_w = (max_col + 1) * PATCH_SIZE
    canvas_h = (max_row + 1) * PATCH_SIZE
    
    # 這是原始的還原影像 (用於計算指標)
    raw_img = Image.new('RGB', (canvas_w, canvas_h))
    
    for (r, c), patch in patches.items():
        raw_img.paste(patch, (c * PATCH_SIZE, r * PATCH_SIZE))
    
    total_time = time.time() - start_time
    print(f"⏱️ Total Time: {total_time:.2f}s")
    
    # ==========================================================================
    # 1. 先計算指標 (使用 Raw Image)
    # ==========================================================================
    if args.original:
        calculate_metrics(raw_img, args.original)
        
    # ==========================================================================
    # 2. 再調整亮度並存檔 (只影響輸出圖片)
    # ==========================================================================
    final_output_img = raw_img
    
    if args.brightness != 1.0:
        print(f"🌟 Applying brightness adjustment (Factor: {args.brightness}) for output only...")
        enhancer = ImageEnhance.Brightness(raw_img)
        final_output_img = enhancer.enhance(args.brightness)

    output_path = os.path.join(args.bin_dir, args.output if args.output else "RECONSTRUCTED.png")
    final_output_img.save(output_path)
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    import multiprocessing
    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass
    main()