#!/bin/bash
#STAGE=3 LAMBDA=0.0108 TEACHER_CKPT="/home/asvserver/TIC/TIC/examples/onnx/onnx_models_d_N32_0108/checkpoint_best.pth.tar" ./run_student.sh

#RESUME_PATH="/home/asvserver/TIC/TIC/examples/pretrained/student/tic_student_N64/q3/L0.0324/checkpoint_best.pth.tar" EXP_NAME="Distill_S1_128to64_q3_20260212_0449" 
# =========================================================
# 蒸餾訓練腳本 (支援中斷續傳 Resume)
# =========================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1. 基礎設定
STAGE=${STAGE:-1} 
GPU_ID=${GPU_ID:-0}
DATASET="/tmp/s2_combined"  # 請確認你的資料路徑

# 2. 決定 Teacher / Student 模型參數
if [ "$STAGE" -eq 1 ]; then
    echo "=== Running Stage 1: 128 -> 64 ==="
    TEACHER_MODEL="cic"
    TEACHER_N=128
    STUDENT_MODEL="cic_student"
    STUDENT_N=64    
    # Stage 1 的 Teacher Checkpoint 預設路徑 (可被外部環境變數覆蓋)
elif [ "$STAGE" -eq 2 ]; then
    echo "=== Running Stage 2: 64 -> 32 ==="
    TEACHER_MODEL="cic_student"
    TEACHER_N=64
    STUDENT_MODEL="cic_tiny"
    STUDENT_N=32
    # Stage 2 的 Teacher 通常是 Stage 1 訓練出來的 Student
elif [ "$STAGE" -eq 3 ]; then
    echo "=== Running Stage 3: 32 -> 16 ==="
    TEACHER_MODEL="cic_student"
    TEACHER_N=32
    STUDENT_MODEL="cic_16"
    STUDENT_N=16
    # Stage 2 的 Teacher 通常是 Stage 1 訓練出來的 Student
else
    echo "Invalid Stage"
    exit 1
fi

# 使用外部傳入的 TEACHER_CKPT，如果沒傳則使用預設值
TEACHER_CKPT=${TEACHER_CKPT:-$DEFAULT_TEACHER}

# 3. 共同參數
M=192
QUALITY=3

# =========================================================
# [核心修改] 續傳邏輯 (Resume Logic)
# =========================================================
# 如果外部有設定 RESUME_PATH，表示要續傳
if [ ! -z "$RESUME_PATH" ]; then
    echo "⚠️  Resume Mode Detected!"
    echo "Loading Student Checkpoint: $RESUME_PATH"
    
    # 續傳時，我們通常希望 Log 繼續寫在原本的資料夾，或者使用原本的實驗名稱
    # 這裡假設外部會傳入原本的 EXP_NAME，如果沒傳，就用 timestamp 避免報錯
    if [ -z "$EXP_NAME" ]; then
        echo "Warning: Resuming without a specific EXP_NAME. Creating a new log folder."
        TIMESTAMP=$(date +"%Y%m%d_%H%M")
        EXP_NAME="Distill_S${STAGE}_Resume_${TIMESTAMP}"
    else
        echo "Resuming Experiment Log: $EXP_NAME"
    fi
    
    # 組合 Resume 參數
    RESUME_ARGS="--checkpoint $RESUME_PATH"
else
    # 全新訓練模式
    echo "🚀 New Training Mode"
    TIMESTAMP=$(date +"%Y%m%d_%H%M")
    EXP_NAME="Distill_S${STAGE}_${TEACHER_N}to${STUDENT_N}_q${QUALITY}_${TIMESTAMP}"
    RESUME_ARGS=""
fi

# =========================================================
# 執行指令
# =========================================================
echo "Experiment Name: $EXP_NAME"
echo "Teacher: $TEACHER_MODEL (N=$TEACHER_N) -> Student: $STUDENT_MODEL (N=$STUDENT_N)"
echo "Teacher Ckpt: $TEACHER_CKPT"

# 使用 exec 執行 python，並帶入 RESUME_ARGS
# 注意："$@" 放在最後，讓你可以在指令行臨時加參數 (例如改 batch size)
python student_train.py \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-n "$TEACHER_N" \
  --student-model "$STUDENT_MODEL" \
  --student-n "$STUDENT_N" \
  --m "$M" \
  --dataset "$DATASET" \
  --teacher-checkpoint "$TEACHER_CKPT" \
  --epochs 200 \
  --learning-rate 1e-4 \
  --aux-learning-rate 1e-3 \
  --batch-size 192 \
  --test-batch-size 1 \
  --num-workers 32 \
  --quality-level "$QUALITY" \
  --lambda "$LAMBDA" \
  --alpha 0.3 \
  --beta 0.3 \
  --gamma 0.4 \
  --roi-factor 1.0 \
  --gpu-id "$GPU_ID" \
  --name "$EXP_NAME" \
  --cuda \
  --save \
  $RESUME_ARGS \
  "$@"