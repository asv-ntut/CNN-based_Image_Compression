# CNN-based Image Compression

本專案是一個基於深度學習的端到端影像壓縮系統，專為衛星影像與邊緣運算裝置所設計。本系統結合了 PyTorch 的訓練框架與高度最佳化的 C++ 擴充模組（支援 rANS 熵編碼），並透過 ONNX Runtime 進行高效能的推論。

本專案的底層 C++ 擴充模組已針對 **ARM Cortex-A53 (PetaLinux)** 進行指令集最佳化（包含 NEON SIMD 向量化與管線排程），以在資源受限的硬體上實現極致的壓縮與解壓縮速度。

---

## 系統需求 (Prerequisites)

* **作業系統：** Linux (推薦 Ubuntu 或 PetaLinux) / macOS / Windows
* **編譯器：** 支援 C++17 的編譯器 (GCC / Clang)
* **Python 環境：** Python >= 3.6
* **核心依賴：** `torch>=1.7.1`, `onnxruntime>=1.16.0`, `pybind11>=2.6.1`

---

## 環境建置 (Installation)

我們提供了一鍵安裝腳本，會自動幫你建立虛擬環境、安裝依賴套件，並編譯底層的 C++ 加速模組。

### 方法一：一鍵自動安裝 (推薦)

請在專案根目錄執行以下指令：

```bash
# 賦予執行權限
chmod +x setup.sh

# 執行腳本 (請務必使用 source 執行，以便腳本結束後保持虛擬環境啟動並進入 scripts 目錄)
source ./setup.sh
```

### 方法二：手動安裝

如果你偏好手動設定，請依照以下步驟執行：

```bash
# 1. 進入專案目錄
cd ~/CNN-based_Image_compression

# 2. 建立並啟動虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安裝環境所需的依賴套件
pip install --upgrade pip
pip install -r requirements.txt

# 4. 刪除舊有的 build 資料夾，並重新編譯 C++ 擴充模組
rm -rf build/
python setup.py build_ext --inplace
```

> **⚠️ 硬體編譯注意 (Hardware Optimization Note):**
> 本專案的 `setup.py` 預設開啟了針對 ARM Cortex-A53 的編譯最佳化參數 (`-mcpu=cortex-a53`, `-ftree-vectorize` 等)。如果你是在一般的 x86 電腦 (Intel/AMD) 上進行開發與測試，請修改 `setup.py` 將相關參數移除，或加上環境變數 `DEBUG_BUILD=1` 進行編譯，以免發生編譯器不認得該指令的錯誤。

---

## 使用教學 (Usage)

環境建置完成後，請確保你位於 `scripts/`（或對應的主程式）目錄下，並已啟動 `.venv` 虛擬環境。完整的壓縮與解壓縮流程如下：

### 1. 產生靜態 CDFs 權重檔
在進行 ONNX 推論前，需要先從 PyTorch 的 `.pth.tar` 權重檔中提取熵編碼所需的 CDFs（累計分配函數）並儲存為 Python 檔：

```bash
# 提取特定 lambda 版本的 CDFs
python cdfs/dump_cdfs.py -i ~/TIC_CPP/cdfs/pth/d_N32/lambda0.0108/distilled_N32_lambda_0.0108_checkpoint_best.pth.tar -o Inference/fixed_cdfs.py

# 或提取最佳權重的 CDFs
python cdfs/dump_cdfs.py -i ~/TIC_CPP/cdfs/pth/best.pth.tar -o Inference/fixed_cdfs.py
```

### 2. 執行影像壓縮 (Compress)
切換到 `Inference` 目錄下（視你的實際架構而定），將原始衛星影像（`.tif`）壓縮為二進位串流：

```bash
cd Inference

python compress.py Taiwan/hualien_RGB_Normalized_tile_r0_c0.tif \
    --enc onnx/onnx_models_d_N32_0108/tic_encoder.onnx \
    --hyper onnx/onnx_models_d_N32_0108/tic_hyper_decoder.onnx \
    --batch 64
```

### 3. 執行影像解壓縮 (Decompress)
將壓縮產生的檔案還原為影像，並可透過 `--original` 參數帶入原圖以計算 PSNR / MS-SSIM 等重建品質指標：

```bash
python decompress.py output/hualien_RGB_Normalized_tile_r0_c0 \
    --dec onnx/onnx_models_d_N32_0108/tic_decoder.onnx \
    --hyper onnx/onnx_models_d_N32_0108/tic_hyper_decoder.onnx \
    --original Taiwan/hualien_RGB_Normalized_tile_r0_c0.tif 
```

---

## 專案架構 (Project Structure)

* `src/compressai/`: 核心模型架構、熵編碼邏輯與 C++ 擴充模組原始碼 (`cpp_exts/`)。
* `src/ryg_rans/`: 外部依賴的高效能 rANS (Range Asymmetric Numeral System) C++ 函式庫。
* `scripts/` (或 `Inference/`): 執行模型壓縮、解壓縮與權重轉換的主程式腳本。
* `setup.sh` / `setup.py`: 負責自動化環境建置與 C++ 綁定編譯。
