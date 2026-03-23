# 1. 進入專案目錄
cd ~/CNN-based_Image_compression

# 2. 建立虛擬環境
python -m venv .venv
# 3. 啟動虛擬環境
source .venv/bin/activate

# 4. 安裝環境所需的依賴套件
pip install --upgrade pip
pip install -r requirements.txt

# 5. 刪除舊有的 build 資料夾，並重新編譯 C++ 擴充模組
rm -rf build/
python setup.py build_ext --inplace



# 其他方法
# 賦予執行權限
chmod +x setup.sh
# 一鍵執行
source ./setup.sh
# 或者是更簡短的寫法：
. ./setup.sh

# 產生fixed_cdfs.py
python cdfs/dump_cdfs.py -i ~/TIC_CPP/cdfs/pth/d_N32/lambda0.0108/distilled_N32_lambda_0.0108_checkpoint_best.pth.tar -o Inference/fixed_cdfs.py
python cdfs/dump_cdfs.py -i ~/TIC_CPP/cdfs/pth/best.pth.tar -o Inference/fixed_cdfs.py
cd Inference
# 運行"compress"
python compress.py Taiwan/hualien_RGB_Normalized_tile_r0_c0.tif --enc onnx/onnx_models_d_N32_0108/tic_encoder.onnx --hyper onnx/onnx_models_d_N32_0108/tic_hyper_decoder.onnx --batch 64
# 運行"decompress"
python decompress.py output/hualien_RGB_Normalized_tile_r0_c0 --dec onnx/onnx_models_d_N32_0108/tic_decoder.onnx --hyper onnx/onnx_models_d_N32_0108/tic_hyper_decoder.onnx --original Taiwan/hualien_RGB_Normalized_tile_r0_c0.tif 