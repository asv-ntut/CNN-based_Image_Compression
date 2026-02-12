#!/bin/bash

# =========================================================
# 實驗配置區 (Experiment Configuration)
# =========================================================
MODEL="tic"           # 模型架構名稱
N=128                 # Hidden Channels
M=192                 # Latent Channels
QUALITY=3             # Quality Level (影響 Lambda)
LAMBDA=0.0972           # Rate-Distortion Trade-off
DATASET="/tmp/s2_combined"  # 訓練資料集路徑

# 實驗名稱 (自動生成，包含時間戳記，方便追蹤)
TIMESTAMP=$(date +"%Y%m%d_%H%M")
EXP_NAME="${MODEL}_N${N}_M${M}_q${QUALITY}_${TIMESTAMP}"

# GPU 設定
GPU_ID="0"

# =========================================================
# 執行指令
# =========================================================
echo "Starting Teacher Training: $EXP_NAME"
echo "Model: $MODEL | N: $N | M: $M"

python train.py \
  --model "$MODEL" \
  --N "$N" \
  --M "$M" \
  --dataset "$DATASET" \
  --epochs 200 \
  --learning-rate 1e-4 \
  --batch-size 192 \
  --num-workers 32 \
  --quality-level "$QUALITY" \
  --lambda "$LAMBDA" \
  --roi-factor 1.0 \
  --gpu-id "$GPU_ID" \
  --name "$EXP_NAME" \
  --cuda \
  --save
  "$@"
# 範例：如何恢復訓練 (Resume)
# 只需在上面加上 --checkpoint "path/to/checkpoint.pth.tar"