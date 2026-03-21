"""
ONNX-based Satellite Image Compression Tool (Parallel Entropy Optimized)

Optimizations:
1. Dynamic CDFs + 2-Way Interleaved C++ ANS: 完美保留壓縮率，並利用 A53 雙發射管線極致加速。
2. Vectorized Patch Extraction.
3. Threading Control: 精準對齊 4 顆實體核心，避免 Context Switch 開銷。
4. Zero-Copy: Pybind11 Numpy Memory Buffer 傳遞。

Usage:
    python compress.py image.tif --enc encoder.onnx --hyper hyper.onnx --batch 16 --workers 4
"""
import argparse
import os
import sys

# 動態將專案的 src/ 目錄加入 Python 的搜尋路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import time
import struct
import zlib
import json
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# 限制 PyTorch 在 Main Process 的全域執行緒數，避免干擾 ONNX
torch.set_num_threads(4)

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai import ans 

try:
    import tifffile
except ImportError:
    tifffile = None

# ==============================================================================
# 0. Load Fixed CDFs
# ==============================================================================
sys.path.insert(0, os.getcwd()) 

try:
    import fixed_cdfs
    import importlib
    importlib.reload(fixed_cdfs)
    print("✅ Loaded fixed_cdfs.py")
except ImportError:
    print("\n[CRITICAL ERROR] Cannot find fixed_cdfs.py!")
    sys.exit(1)

# ==============================================================================
# 1. Parallel Worker Function (Zero-Copy C++ 2-Way ANS)
# ==============================================================================
def _compress_worker_dynamic(y_chunk, scales_chunk, means_chunk, cdf_list, offset_list, cdf_lengths, scale_table):
    # 強制鎖死 Worker 只能用單執行緒
    torch.set_num_threads(1)
    
    gc = GaussianConditional(None)
    gc.scale_table = scale_table
    indexes = gc.build_indexes(scales_chunk).to(torch.int32)
    
    symbols = torch.round(y_chunk - means_chunk).to(torch.int32)
    symbols = symbols.contiguous()
    indexes = indexes.contiguous()
    
    encoder = ans.DynamicRansEncoder2Way()
    strings = []
    
    for i in range(symbols.shape[0]):
        # O(1) 零拷貝提取指標
        sym_np = symbols[i].flatten().numpy()
        idx_np = indexes[i].flatten().numpy()
        
        bitstream = encoder.encode_with_indexes(
            sym_np, idx_np, cdf_list, cdf_lengths, offset_list
        )
        strings.append([bitstream]) 
        
    return strings

# ==============================================================================
# 2. Profiling & Analysis Tools
# ==============================================================================
def analyze_onnx_profile(profile_file, model_name="Model"):
    if not os.path.exists(profile_file) or os.path.getsize(profile_file) == 0: return
    try:
        with open(profile_file, 'r') as f: content = f.read()
        start, end = content.find('['), content.rfind(']') + 1
        data = json.loads(content[start:end] if start != -1 else content)
    except: return

    op_times = {}
    for e in data:
        if e.get("cat") == "Node":
            op = e.get("args", {}).get("op_name", "Unknown")
            if op in {"Pow", "Sqrt", "Reciprocal"}: op = f"GDN ({op})"
            op_times.setdefault(op, []).append(e.get("dur", 0))

    if not op_times: return
    print(f"\n{'='*60}\n🔬 {model_name} Profiling\n{'-'*60}")
    tot = sum(sum(t) for t in op_times.values())
    for op, times in sorted(op_times.items(), key=lambda x: sum(x[1]), reverse=True):
        print(f"{op:<20} | {sum(times)/1000:>10.2f} ms | {sum(times)/tot*100:>5.1f}%")
    try: os.remove(profile_file)
    except: pass

# ==============================================================================
# 3. Binary Packet & Patch Extraction
# ==============================================================================
def save_satellite_packet(out_enc, output_path, img_id, row, col):
    y_in, z_in = out_enc["strings"][0], out_enc["strings"][1]
    y_str = b''.join(y_in) if isinstance(y_in, list) else y_in
    z_str = b''.join(z_in) if isinstance(z_in, list) else z_in
    
    header = struct.pack('<3sBBBHHII', b'TIC', img_id, row, col, 
                         out_enc["shape"][0], out_enc["shape"][1], len(y_str), len(z_str))
    payload = header + y_str + z_str
    footer = struct.pack('<I', zlib.crc32(payload) & 0xffffffff)
    with open(output_path, "wb") as f: f.write(payload + footer)

def extract_patches_vectorized(full_image, patch_size=256):
    c, h, w = full_image.shape
    ph = (patch_size - h % patch_size) % patch_size
    pw = (patch_size - w % patch_size) % patch_size
    img = np.pad(full_image, ((0,0),(0,ph),(0,pw)), mode='constant')
    _, h_pad, w_pad = img.shape
    nr, nc = h_pad // patch_size, w_pad // patch_size
    patches = img.reshape(c, nr, patch_size, nc, patch_size).transpose(1,3,0,2,4).reshape(-1, c, patch_size, patch_size)
    return patches.astype(np.float32), [(r, c) for r in range(nr) for c in range(nc)], nr, nc

def pad_batch(batch, mult=64):
    h, w = batch.shape[2], batch.shape[3]
    nh, nw = ((h+mult-1)//mult)*mult, ((w+mult-1)//mult)*mult
    if nh==h and nw==w: return batch
    return np.pad(batch, ((0,0),(0,0),((nh-h)//2, nh-h-(nh-h)//2),((nw-w)//2, nw-w-(nw-w)//2)), mode='constant')

# ==============================================================================
# 4. Core Processing (Parallelized)
# ==============================================================================
def process_batch(sessions, models, executor, batch_x, batch_meta, save_dir, base, img_id, stats):
    enc, hyper = sessions
    eb, gc = models
    
    # 1. Prep
    t0 = time.time()
    padded = pad_batch(batch_x)
    stats['prep'] += time.time() - t0
    
    # 2. Encoder (ONNX - uses 4 intra_op threads in Main Process)
    t0 = time.time()
    enc_out = enc.run(None, {"input_image": padded})
    y_batch = torch.from_numpy(enc_out[0])
    z_batch = torch.from_numpy(enc_out[1])
    stats['encoder'] += time.time() - t0
    
    # 3. Entropy Z
    t0 = time.time()
    z_batch = z_batch.contiguous()
    z_strings_list = eb.compress(z_batch)
    medians = eb._get_medians().detach()
    spatial = len(z_batch.size()) - 2
    medians = eb._extend_ndims(medians, spatial).expand(z_batch.size(0), *([-1]*(spatial+1)))
    z_hat_batch = eb.quantize(z_batch, "dequantize", medians)
    stats['entropy_z'] += time.time() - t0
    
    # 4. HyperDecoder
    t0 = time.time()
    hyper_out = hyper.run(None, {"z_hat": z_hat_batch.numpy()})
    scales_batch = torch.from_numpy(hyper_out[0])
    means_batch = torch.from_numpy(hyper_out[1])
    stats['hyper'] += time.time() - t0
    
    # 5. Entropy Y (Parallelized Workers + Dynamic C++ 2-Way ANS)
    t0 = time.time()
    batch_size = len(batch_x)
    num_workers = executor._max_workers
    chunk_size = (batch_size + num_workers - 1) // num_workers
    
    gc_cdf_list = fixed_cdfs.FIXED_GC_CDF
    gc_offset_list = fixed_cdfs.FIXED_GC_OFFSET
    gc_lengths_list = fixed_cdfs.FIXED_GC_LENGTH

    futures = []
    for i in range(0, batch_size, chunk_size):
        y_chunk = y_batch[i : i + chunk_size]
        scales_chunk = scales_batch[i : i + chunk_size]
        means_chunk = means_batch[i : i + chunk_size]
        
        futures.append(executor.submit(
            _compress_worker_dynamic, 
            y_chunk, scales_chunk, means_chunk,
            gc_cdf_list, gc_offset_list, gc_lengths_list, gc.scale_table
        ))
    
    y_strings_list = []
    for f in futures:
        y_strings_list.extend(f.result())
        
    stats['entropy_y'] += time.time() - t0
    
    # 6. Save
    t0 = time.time()
    total_bits = 0
    z_shape = z_batch.shape[-2:]
    for i in range(batch_size):
        r, c = batch_meta[i]
        fpath = os.path.join(save_dir, f"{base}_r{r}_c{c}.bin")
        save_satellite_packet({"strings": [y_strings_list[i], z_strings_list[i]], "shape": z_shape}, fpath, img_id, r, c)
        total_bits += os.path.getsize(fpath) * 8
    stats['save'] += time.time() - t0
        
    return total_bits / (batch_x.shape[0] * batch_x.shape[2] * batch_x.shape[3])

def compress_image(sessions, models, executor, path, out_dir, img_id, batch, psize):
    fname = os.path.basename(path)
    sdir = os.path.join(out_dir, os.path.splitext(fname)[0])
    os.makedirs(sdir, exist_ok=True)
    
    stats = {k: 0.0 for k in ['load', 'extract', 'prep', 'encoder', 'entropy_z', 'hyper', 'entropy_y', 'save']}
    
    t0 = time.time()
    if tifffile and path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(path)
        if img.ndim==2: img=img[None]
        elif img.ndim==3 and img.shape[2]<=4: img=img.transpose(2,0,1)
    else:
        img = np.array(Image.open(path).convert("RGB")).transpose(2,0,1)
    img = img.astype(np.float32) / (255.0 if img.dtype==np.uint8 else 1.0)
    img = np.clip(img, 0, 1)[:3]
    stats['load'] = time.time() - t0
    
    t0 = time.time()
    patches, meta, _, _ = extract_patches_vectorized(img, psize)
    stats['extract'] = time.time() - t0
    
    print(f"\nProcessing {fname}: {img.shape[2]}x{img.shape[1]} -> {len(patches)} patches (Workers: {executor._max_workers})")
    
    bpp_sum = 0
    for i in range(0, len(patches), batch):
        end = min(i+batch, len(patches))
        bpp_sum += process_batch(sessions, models, executor, patches[i:end], meta[i:end], sdir, os.path.splitext(fname)[0], img_id, stats) * (end-i)
        print(f"Progress: {end}/{len(patches)}", end='\r')
        
    print(f"\n\n{'='*55}\n📊 Performance Report: {fname}\n{'='*55}")
    tot = sum(stats.values())
    for k, v in stats.items(): print(f"{k:<15} | {v:.4f}s    | {v/tot*100:.1f}%")
    print(f"{'-'*55}\nTotal Time      | {tot:.4f}s\nAverage BPP     | {bpp_sum/len(patches):.4f}\n{'='*55}\n")

# ==============================================================================
# 5. Init & Main
# ==============================================================================
def init_env(enc_path, hyp_path, cpu, profile):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 🚨 鎖死 ONNX 的執行緒配置，完美適配 4 顆 A53
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    
    if profile: opts.enable_profiling = True
    prov = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    enc_sess = ort.InferenceSession(enc_path, sess_options=opts, providers=prov)
    hyp_sess = ort.InferenceSession(hyp_path, sess_options=opts, providers=prov)
    
    try:
        dummy = np.zeros((1, 3, 64, 64), dtype=np.float32)
        N_onnx = enc_sess.run(None, {"input_image": dummy})[1].shape[1]
    except:
        N_onnx = len(fixed_cdfs.FIXED_EB_MEDIANS)
    
    N_cdf = len(fixed_cdfs.FIXED_EB_MEDIANS)
    if N_onnx != N_cdf:
        print(f"\n❌ Dimension Mismatch! N={N_onnx} vs CDF={N_cdf}. Run dump_cdfs.py!")
        sys.exit(1)
        
    eb = EntropyBottleneck(N_cdf)
    gc = GaussianConditional(None)
    
    dev = torch.device("cpu")
    for mod, prefix in [(eb, "FIXED_EB")]:
        mod._quantized_cdf.resize_(torch.tensor(getattr(fixed_cdfs, f"{prefix}_CDF")).shape).copy_(
            torch.tensor(getattr(fixed_cdfs, f"{prefix}_CDF"), device=dev, dtype=torch.int32))
        mod._offset.resize_(torch.tensor(getattr(fixed_cdfs, f"{prefix}_OFFSET")).shape).copy_(
            torch.tensor(getattr(fixed_cdfs, f"{prefix}_OFFSET"), device=dev, dtype=torch.int32))
        mod._cdf_length.resize_(torch.tensor(getattr(fixed_cdfs, f"{prefix}_LENGTH")).shape).copy_(
            torch.tensor(getattr(fixed_cdfs, f"{prefix}_LENGTH"), device=dev, dtype=torch.int32))
    
    eb.quantiles.data[:,0,1] = torch.tensor(fixed_cdfs.FIXED_EB_MEDIANS, device=dev).squeeze()
    gc.scale_table = torch.tensor(fixed_cdfs.FIXED_GC_SCALE_TABLE, device=dev)
    
    return (enc_sess, hyp_sess), (eb, gc), profile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", nargs='+')
    parser.add_argument("-o", "--output_dir", default="output_onnx")
    parser.add_argument("--enc", required=True)
    parser.add_argument("--hyper", required=True)
    # 預設 Batch 調降為 16，保護開發板記憶體
    parser.add_argument("--batch", type=int, default=16) 
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    
    sess, mods, prof = init_env(args.enc, args.hyper, args.cpu, args.profile)
    
    print(f"🚀 Initializing Process Pool with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for p in args.input_path:
            compress_image(sess, mods, executor, p, args.output_dir, 1, args.batch, 256)
        
    if prof:
        try:
            analyze_onnx_profile(sess[0].end_profiling(), "Encoder")
            analyze_onnx_profile(sess[1].end_profiling(), "HyperDecoder")
        except: pass

if __name__ == "__main__":
    main()