"""
ONNX-based Satellite Image Decompression Tool (Final Fix for String Type)

Fixes:
1. **Invalid `strings` parameter type**: Wraps byte strings in lists explicitly.
2. Debug Mode active.
3. Multiprocessing Optimized.

Usage:
    python decompress.py ./output_onnx/hualien_tile_r0_c0 --dec decoder.onnx --hyper hyper.onnx --workers 4
"""
import argparse
import os
import sys
import glob
import time
import struct
import zlib
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageEnhance
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try to import MS-SSIM
try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# ==============================================================================
# 0. Global Setup for Workers
# ==============================================================================
worker_dec_sess = None
worker_hyper_sess = None
worker_entropy_models = None

def init_worker(dec_path, hyper_path, use_cuda, cdf_module_path):
    global worker_dec_sess, worker_hyper_sess, worker_entropy_models
    
    if cdf_module_path not in sys.path:
        sys.path.insert(0, cdf_module_path)
    
    try:
        import fixed_cdfs
        import importlib
        importlib.reload(fixed_cdfs)
        HAS_FIXED_CDFS = True
    except ImportError:
        HAS_FIXED_CDFS = False
        print(f"[Worker {os.getpid()}] ⚠️ Cannot find fixed_cdfs.py")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    
    try:
        worker_dec_sess = ort.InferenceSession(dec_path, sess_options=opts, providers=providers)
        worker_hyper_sess = ort.InferenceSession(hyper_path, sess_options=opts, providers=providers)
    except Exception as e:
        print(f"[Worker {os.getpid()}] ❌ Failed to load ONNX models: {e}")
        return

    if HAS_FIXED_CDFS:
        N = len(fixed_cdfs.FIXED_EB_MEDIANS)
        eb = EntropyBottleneck(N)
        gc = GaussianConditional(None)
        
        dev = torch.device("cpu")
        
        eb._quantized_cdf.resize_(torch.tensor(fixed_cdfs.FIXED_EB_CDF).shape).copy_(
            torch.tensor(fixed_cdfs.FIXED_EB_CDF, device=dev, dtype=torch.int32))
        eb._offset.resize_(torch.tensor(fixed_cdfs.FIXED_EB_OFFSET).shape).copy_(
            torch.tensor(fixed_cdfs.FIXED_EB_OFFSET, device=dev, dtype=torch.int32))
        eb._cdf_length.resize_(torch.tensor(fixed_cdfs.FIXED_EB_LENGTH).shape).copy_(
            torch.tensor(fixed_cdfs.FIXED_EB_LENGTH, device=dev, dtype=torch.int32))
        eb.quantiles.data[:, 0, 1] = torch.tensor(fixed_cdfs.FIXED_EB_MEDIANS, device=dev).squeeze()

        gc._quantized_cdf.resize_(torch.tensor(fixed_cdfs.FIXED_GC_CDF).shape).copy_(
            torch.tensor(fixed_cdfs.FIXED_GC_CDF, device=dev, dtype=torch.int32))
        gc._offset.resize_(torch.tensor(fixed_cdfs.FIXED_GC_OFFSET).shape).copy_(
            torch.tensor(fixed_cdfs.FIXED_GC_OFFSET, device=dev, dtype=torch.int32))
        gc._cdf_length.resize_(torch.tensor(fixed_cdfs.FIXED_GC_LENGTH).shape).copy_(
            torch.tensor(fixed_cdfs.FIXED_GC_LENGTH, device=dev, dtype=torch.int32))
        gc.scale_table = torch.tensor(fixed_cdfs.FIXED_GC_SCALE_TABLE, device=dev)
        
        worker_entropy_models = (eb, gc)
    else:
        worker_entropy_models = None

# ==============================================================================
# 1. Packet Parsing (FIXED)
# ==============================================================================
def load_packet(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 22: return None
        
        if struct.unpack('<I', data[-4:])[0] != (zlib.crc32(data[:-4]) & 0xffffffff):
            print(f"❌ CRC Mismatch: {os.path.basename(path)}")
            return None 
        
        magic, iid, r, c, h, w, ly, lz = struct.unpack('<3sBBBHHII', data[:18])
        if magic != b'TIC': return None
        
        # [CRITICAL FIX] Wrap bytes in a list: [b'...']
        y_payload = [data[18 : 18 + ly]]
        z_payload = [data[18 + ly : 18 + ly + lz]]
        
        return {
            "row": r, "col": c, "id": iid, 
            "str": [y_payload, z_payload], 
            "sh": (h, w)
        }
    except Exception as e:
        print(f"❌ Load Error {path}: {e}")
        return None

def process_task(path, tid):
    if not worker_entropy_models: return None
        
    pkt = load_packet(path)
    if not pkt: return None
    if tid is not None and pkt['id'] != tid: return None
    
    eb, gc = worker_entropy_models
    try:
        # Entropy Decode Z
        z_hat = eb.decompress(pkt["str"][1], pkt["sh"])
        
        # Hyper Decoder (ONNX)
        scales, means = worker_hyper_sess.run(None, {"z_hat": z_hat.numpy()})
        
        # Entropy Decode Y
        idx = gc.build_indexes(torch.from_numpy(scales))
        y_hat = gc.decompress(pkt["str"][0], idx, means=torch.from_numpy(means))
        
        # Main Decoder (ONNX)
        x_hat = torch.from_numpy(worker_dec_sess.run(None, {"y_hat": y_hat.numpy()})[0]).clamp(0, 1)
        
        if x_hat.size(2) != 256 or x_hat.size(3) != 256:
            x_hat = x_hat[:, :, :256, :256]
            
        return (pkt['row'], pkt['col'], (x_hat.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    except Exception as e:
        # Debug info
        print(f"\n❌ Decoding Failed on {os.path.basename(path)}")
        print(f"   Reason: {e}")
        return None

# ==============================================================================
# Metrics & Main
# ==============================================================================
from torchvision import transforms

def calc_metrics(rec_img, orig_path):
    if not os.path.exists(orig_path): return
    print("Calculating Metrics on RAW reconstruction...")
    try:
        if orig_path.lower().endswith(('.tif', '.tiff')):
            import tifffile
            gt = tifffile.imread(orig_path)
            if gt.ndim == 3 and gt.shape[2] <= 4: gt = gt.transpose(2, 0, 1)
        else:
            gt = np.array(Image.open(orig_path).convert("RGB")).transpose(2, 0, 1)
        
        gt = torch.from_numpy(gt.astype(np.float32) / (255.0 if gt.dtype == np.uint8 else 1.0)).clamp(0, 1)
        gt = gt[:3] 
        rec = transforms.ToTensor()(rec_img)
        
        h, w = min(gt.shape[1], rec.shape[1]), min(gt.shape[2], rec.shape[2])
        gt, rec = gt[:, :h, :w], rec[:, :h, :w]
        
        mse = torch.mean((gt - rec) ** 2)
        psnr = -10 * torch.log10(mse)
        print(f"✅ PSNR:    {psnr:.4f} dB")
        
        if HAS_MSSSIM:
            val_msssim = ms_ssim(gt.unsqueeze(0), rec.unsqueeze(0), data_range=1.0).item()
            print(f"✅ MS-SSIM: {val_msssim:.4f}")
            
    except Exception as e:
        print(f"❌ Metric calculation error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_dir")
    parser.add_argument("--dec", required=True)
    parser.add_argument("--hyper", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--original", default=None)
    parser.add_argument("--target_id", type=int, default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--brightness", type=float, default=1.0)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.bin_dir, "*.bin"))
    if not files: return print("No files found.")
    
    print(f"🚀 Decoding {len(files)} packets with {args.workers} workers...")
    t0 = time.time()
    
    results = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(args.dec, args.hyper, not args.cpu, os.getcwd())) as ex:
        futures = {ex.submit(process_task, f, args.target_id): f for f in files}
        for i, fut in enumerate(as_completed(futures)):
            print(f"Progress: {i+1}/{len(files)}", end='\r')
            res = fut.result()
            if res: results.append(res)
    
    print(f"\n📥 Decoded {len(results)} / {len(files)} patches successfully.")
    
    if not results:
        print("\n❌ Critical Failure: All packets failed to decode.")
        return

    results.sort(key=lambda x: (x[0], x[1]))
    max_r, max_c = results[-1][0], results[-1][1]
    raw_img = Image.new('RGB', ((max_c+1)*256, (max_r+1)*256))
    
    seen = set()
    for r, c, arr in results:
        if (r,c) in seen: print(f"⚠️ Duplicate: ({r},{c})")
        seen.add((r,c))
        raw_img.paste(Image.fromarray(arr), (c*256, r*256))
        
    print(f"⏱️ Time: {time.time()-t0:.2f}s")
    
    if args.original: calc_metrics(raw_img, args.original)
    
    final_img = raw_img
    if args.brightness != 1.0:
        print(f"🌟 Applying brightness x{args.brightness}")
        final_img = ImageEnhance.Brightness(raw_img).enhance(args.brightness)
        
    out = os.path.join(args.bin_dir, args.output if args.output else "RECONSTRUCTED.png")
    final_img.save(out)
    print(f"💾 Saved: {out}")

if __name__ == "__main__":
    import multiprocessing
    try: multiprocessing.set_start_method('spawn')
    except: pass
    main()
    