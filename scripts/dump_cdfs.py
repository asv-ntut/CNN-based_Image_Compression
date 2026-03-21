import torch
import sys
import os
import argparse
from compressai.models.tic import TIC

def dump_cdfs(checkpoint_path, output_filename):
    # 檢查路徑是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ 錯誤: 找不到 Checkpoint 檔案 '{checkpoint_path}'")
        return

    # 1. 載入模型權重
    print(f"🚀 正在讀取 Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # 移除 module. 前綴 (如果有的話)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 2. 自動偵測 N, M 參數
    N, M = 128, 192
    try:
        N = new_state_dict['g_a.0.weight'].size(0)
        keys = sorted([k for k in new_state_dict.keys() if 'g_a' in k and 'weight' in k])
        M = new_state_dict[keys[-1]].size(0)
        print(f"📊 偵測到模型架構: N={N}, M={M}")
    except Exception as e:
        print(f"⚠️ 無法自動偵測 N, M，將使用預設值 (128, 192). 錯誤: {e}")

    # 3. 初始化模型並套用權重
    model = TIC(N=N, M=M)
    model.load_state_dict(new_state_dict, strict=True)
    
    # 強制使用 CPU 執行以確保精度一致
    model = model.to('cpu')
    
    # 4. 更新 CDFs (此步驟會生成機率表)
    print("⏳ 正在更新 CDF 表 (這可能需要一點時間)...")
    model.update(force=True)
    
    eb = model.entropy_bottleneck
    gc = model.gaussian_conditional
    medians = eb._get_medians().detach()
    
    # 5. 寫入檔案
    with open(output_filename, "w") as f:
        # EntropyBottleneck CDFs
        f.write(f"FIXED_EB_CDF = {eb._quantized_cdf.tolist()}\n")
        f.write(f"FIXED_EB_OFFSET = {eb._offset.tolist()}\n")
        f.write(f"FIXED_EB_LENGTH = {eb._cdf_length.tolist()}\n")
        f.write(f"FIXED_EB_MEDIANS = {medians.tolist()}\n")
        
        # GaussianConditional CDFs
        f.write(f"FIXED_GC_CDF = {gc._quantized_cdf.tolist()}\n")
        f.write(f"FIXED_GC_OFFSET = {gc._offset.tolist()}\n")
        f.write(f"FIXED_GC_LENGTH = {gc._cdf_length.tolist()}\n")
        f.write(f"FIXED_GC_SCALE_TABLE = {gc.scale_table.tolist()}\n")
        
    print(f"✅ 成功產出: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIC CDF Dumper for FPGA/PetaLinux Consistency")
    
    # 添加命令行參數
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="輸入的 .pth.tar 模型路徑")
    parser.add_argument("-o", "--output", type=str, default="fixed_cdfs.py", 
                        help="輸出的 Python 檔名 (預設: fixed_cdfs.py)")
    
    args = parser.parse_args()

    dump_cdfs(args.input, args.output)