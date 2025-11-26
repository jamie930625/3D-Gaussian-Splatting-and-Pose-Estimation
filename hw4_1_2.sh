#!/bin/bash

###############################################
# HW4_1_2.sh  —  Interpolated Sequences
###############################################

# TA will run:
#   bash hw4_1_2.sh $1 $2 $3 $4 $5
#
#   $1 = index txt
#   $2 = original image root
#   $3 = interpolated sequence directory (root)
#   $4 = model checkpoint
#   $5 = output prediction npy

INDEX_TXT_PATH="$1"
DATA_ROOT="$2"
INTERP_DIR="$3"
MODEL_PATH="$4"
SAVE_POSE_PATH="$5"

echo "----------------------------------------"
echo " Running HW4_1_2.sh (Interpolated seq.) "
echo "----------------------------------------"
echo "Index TXT       : ${INDEX_TXT_PATH}"
echo "Original Images : ${DATA_ROOT}"
echo "Interpolated Dir: ${INTERP_DIR}"
echo "Model Checkpt   : ${MODEL_PATH}"
echo "Output Pose NPY : ${SAVE_POSE_PATH}"
echo ""

# -------------------------
#  FIXED CONFIGURATION
# -------------------------

# Public ground truth file
GT_NPY_PATH="hw_4_1_data/public/gt.npy"

# Output directory (for logs, temp files)
OUTPUT_DIR="output/hw4_1_2_interpolated"
mkdir -p ${OUTPUT_DIR}

# Evaluation: rotation only
EVAL_MODE="R"

# -------------------------
# Run Dust3R Inference
# -------------------------
python3 dust3r_inference.py \
    --index_txt_path "${INDEX_TXT_PATH}" \
    --gt_npy_path "${GT_NPY_PATH}" \
    --data_root "${DATA_ROOT}" \
    --interpolated_dir "${INTERP_DIR}" \
    --use_original_endpoints \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --eval_mode "${EVAL_MODE}" \
    --batch_size 4 \
    --num_workers 4 \
    --seed 0 \
    --save_pose_path "${SAVE_POSE_PATH}"

echo ""
echo "----------------------------------------"
echo " Interpolated sequence inference finished. "
echo " Predicted poses saved to: ${SAVE_POSE_PATH}"
echo "----------------------------------------"
