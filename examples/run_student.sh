#!/bin/bash

# =========================================================
# 蒸餾階段設定 (Distillation Stage Configuration)
# =========================================================
# 設定你要跑哪個階段：
# 1 = Teacher(128) -> Student(64)
# 2 = Teacher(64)  -> Student(32)
STAGE=${STAGE:-1} 
GPU_ID=${GPU_ID:-0}

DATASET="/tmp/s2_combined" # 訓練資料集路徑

if [ "$STAGE" -eq 1 ]; then
    echo "=== Running Stage 1: 128 -> 64 ==="
    TEACHER_MODEL="tic"
    TEACHER_N=128
    STUDENT_MODEL="tic_student"
    STUDENT_N=64    
elif [ "$STAGE" -eq 2 ]; then
    echo "=== Running Stage 2: 64 -> 32 ==="
    TEACHER_MODEL="tic_student"
    TEACHER_N=64
    STUDENT_MODEL="tic_tiny"
    STUDENT_N=32
else
    echo "Invalid Stage"
    exit 1
fi

# 共同參數
M=192
QUALITY=3
LAMBDA=0.0324

TIMESTAMP=$(date +"%Y%m%d_%H%M")
EXP_NAME="Distill_S${STAGE}_${TEACHER_N}to${STUDENT_N}_q${QUALITY}_${TIMESTAMP}"

# =========================================================
# 執行指令
# =========================================================
echo "Experiment Name: $EXP_NAME"
echo "Teacher: $TEACHER_MODEL (N=$TEACHER_N) -> Student: $STUDENT_MODEL (N=$STUDENT_N)"
echo "Loading Teacher form: $TEACHER_CKPT"

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
  --batch-size 288 \
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
  "$@"