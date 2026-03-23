#!/bin/bash

# 設定腳本只要遇到錯誤就立刻停止
set -e

echo "開始建置 CNN-Based Image Compression 專案環境..."

# 1. 切換到專案目錄
cd $(pwd)
echo "目前路徑: $(pwd)"

# 2. 建立虛擬環境
if [ ! -d ".venv" ]; then
    echo "尚未找到虛擬環境，正在建立 .venv..."
    python3 -m venv .venv
else
    echo "發現已存在的 .venv 虛擬環境。"
fi

# 3. 啟動虛擬環境
echo "正在啟動虛擬環境..."
source .venv/bin/activate

# 4. 安裝核心相依套件
echo "正在更新 pip 並安裝 requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. 清理舊的編譯產物 (編譯前預防性清理)
echo "正在清理舊的 build/ 資料夾..."
rm -rf build/

# 6. 重新編譯 C++ 擴充模組
echo "開始編譯 compressai 的 C++ 擴充模組..."
python setup.py build_ext --inplace

# 7. 清理編譯後殘留的無用檔案 (瘦身專案資料夾)
echo "正在清理安裝後不需要的暫存檔與 build/..."
rm -rf build/
rm -rf src/*.egg-info/
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "清理完成！"

# 8. 自動進入 scripts 資料夾
echo "切換至主程式目錄..."
cd scripts/

echo "========================================"
echo "環境建置與編譯大功告成！"
echo "目前的所在位置: $(pwd)"
echo "請確保你是使用 'source setup.sh' 執行的，這樣才能直接開始跑模型哦！"
echo "========================================"
