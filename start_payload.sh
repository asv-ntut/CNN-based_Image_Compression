#!/bin/bash
echo "🚀 啟動 TIC 衛星影像壓縮酬載服務..."

# 1. 切換至專案絕對路徑 (使用 $HOME 確保背景執行時路徑正確)
PROJECT_DIR="$HOME/CNN-based_Image_compression"

# 檢查目錄是否存在，避免路徑錯誤導致腳本盲目執行
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
else
    echo "❌ 找不到專案目錄: $PROJECT_DIR"
    exit 1
fi

# 2. 啟用 Python 虛擬環境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 虛擬環境已啟用"
else
    echo "❌ 找不到 .venv 虛擬環境！請先執行 setup.sh 進行環境建置。"
    exit 1
fi

# 3. 於背景啟動 Python AI Worker (監聽本機 Port 6000)
# 註: 模型路徑指向 d_N32 版本,batch 設為 16 保護 A53 記憶體
echo "啟動背景 AI Worker..."
python ./scripts/compress.py \
    --enc "$PROJECT_DIR/scripts/onnx_models_d_N32/tic_encoder.onnx" \
    --hyper "$PROJECT_DIR/scripts/onnx_models_d_N32/tic_hyper_decoder.onnx" \
    --batch 64 \
    --workers 4 \
    --comm_ip 127.0.0.1 &

# 等待模型載入完成 (避免 C 程式先送指令但 Python 尚未就緒)
echo "⏳ 等待 AI Worker 載入 ONNX 模型 (約 5 秒)..."
sleep 5

# 4. 於前台啟動 C Dispatcher 監聽 OBC UART 訊號
echo "📡 啟動 UART 監聽接口 (Dispatcher)..."
# 注意：讀取 /dev/ttyUSB0 通常需要 sudo 權限
sudo ./dispatcher