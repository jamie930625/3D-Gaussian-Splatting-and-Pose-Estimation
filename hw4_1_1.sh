#!/bin/bash

###############################################
# HW4_1_1.sh  —  Wide-baseline (Original Pairs)
###############################################

# TA will run:
#   bash hw4_1_1.sh $1 $2 $3 $4
#   $1 = index txt
#   $2 = image root
#   $3 = model checkpoint
#   $4 = output prediction npy

INDEX_TXT_PATH="$1"
DATA_ROOT="$2"
MODEL_PATH="$3"
SAVE_POSE_PATH="$4"

echo "----------------------------------------"
echo " Running HW4_1_1.sh (Wide-Baseline) "
echo "----------------------------------------"
echo "Index TXT      : ${INDEX_TXT_PATH}"
echo "Image Root     : ${DATA_ROOT}"
echo "Model Checkpt  : ${MODEL_PATH}"
echo "Output Pose NPY: ${SAVE_POSE_PATH}"
echo ""

# -------------------------
#  FIXED (You must supply)
# -------------------------

# Ground-truth path (public dataset only)
GT_NPY_PATH="hw_4_1_data/public/gt.npy"

# Output directory for logs / tmp files
OUTPUT_DIR="output/hw4_1_1_wide_baseline"
mkdir -p ${OUTPUT_DIR}

# Do NOT use interpolated frames
INTERP_DIR=""

# Use ONLY original endpoints
USE_ORIGINAL_FLAG=""

# Evaluation mode (Rotation only)
EVAL_MODE="R"

# -------------------------
# Run Dust3R Inference
# -------------------------
python3 dust3r_inference.py \
    --index_txt_path "${INDEX_TXT_PATH}" \
    --gt_npy_path "${GT_NPY_PATH}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --eval_mode "${EVAL_MODE}" \
    --batch_size 4 \
    --num_workers 4 \
    --seed 0 \
    --save_pose_path "${SAVE_POSE_PATH}"

echo ""
echo "----------------------------------------"
echo " Wide-baseline inference finished. "
echo " Predicted poses saved to: ${SAVE_POSE_PATH}"
echo "----------------------------------------"
