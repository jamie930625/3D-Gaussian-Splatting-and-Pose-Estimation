#!/bin/bash
set -e

DATA_ROOT="$1"
OUT_DIR="$2"

echo "Data root: $DATA_ROOT"
echo "Output dir: $OUT_DIR"

# 超參數（你可以自己微調這三個數）
INIT_NITER=1          # init_geo 裡面其實沒用到，但留著無妨
TRAIN_ITER=100        # train.py 的 iterations
RENDER_ITER=100       # render 時載入的 checkpoint 迭代數
POSE_OPT_ITER=100     # <<< 這個才是原本顯示 500/500 的那個

# =========================
# CKPT (ABSOLUTE PATH)
# =========================
CKPT="$(pwd)/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
echo "Using CKPT: $CKPT"

# =========================
# Detect sparse_3 (n_views)
# =========================
SPARSE_DIR=$(find "$DATA_ROOT" -maxdepth 1 -type d -name "sparse_*" | head -n 1)
BASENAME=$(basename "$SPARSE_DIR")
N_VIEWS=${BASENAME#sparse_}
echo "[INFO] Detected $BASENAME → n_views=$N_VIEWS"

# =========================
# Create TEMP dirs
# =========================
TMP_TRAIN=$(mktemp -d)
TMP_TEST=$(mktemp -d)

mkdir -p "$TMP_TRAIN/sparse_$N_VIEWS/0"
mkdir -p "$TMP_TEST/sparse_$N_VIEWS/0"
mkdir -p "$TMP_TEST/sparse_$N_VIEWS/1"

echo "[INFO] Using TMP_TRAIN=$TMP_TRAIN"
echo "[INFO] Using TMP_TEST=$TMP_TEST"

# ===========================================
# TRAINING SOURCE  (3 train images from 0/)
# ===========================================
ln -sf "$DATA_ROOT/images"                         "$TMP_TRAIN/images"
ln -sf "$DATA_ROOT/sparse_$N_VIEWS/0/images.txt"   "$TMP_TRAIN/sparse_$N_VIEWS/0/images.txt"
ln -sf "$DATA_ROOT/sparse_$N_VIEWS/0/cameras.txt"  "$TMP_TRAIN/sparse_$N_VIEWS/0/cameras.txt"

# train needs point cloud
for f in points3D.ply points3D.bin points3D.txt points3D_all.npy pointsColor_all.npy non_scaled_focals.npy confidence_dsp.npy; do
    if [ -e "$DATA_ROOT/sparse_$N_VIEWS/0/$f" ]; then
        ln -sf "$DATA_ROOT/sparse_$N_VIEWS/0/$f" "$TMP_TRAIN/sparse_$N_VIEWS/0/$f"
    fi
done

# ===========================================
# TESTING SOURCE  (4 test cameras from 1/)
# ===========================================
# test still needs train metadata (required by COLMAP loader!)
ln -sf "$DATA_ROOT/sparse_$N_VIEWS/0/images.txt"   "$TMP_TEST/sparse_$N_VIEWS/0/images.txt"
ln -sf "$DATA_ROOT/sparse_$N_VIEWS/0/cameras.txt"  "$TMP_TEST/sparse_$N_VIEWS/0/cameras.txt"

for f in points3D.ply points3D.bin points3D.txt points3D_all.npy pointsColor_all.npy non_scaled_focals.npy confidence_dsp.npy; do
    if [ -e "$DATA_ROOT/sparse_$N_VIEWS/0/$f" ]; then
        ln -sf "$DATA_ROOT/sparse_$N_VIEWS/0/$f" "$TMP_TEST/sparse_$N_VIEWS/0/$f"
    fi
done

# only the 4 test images/cameras
ln -sf "$DATA_ROOT/images"                         "$TMP_TEST/images"
ln -sf "$DATA_ROOT/sparse_$N_VIEWS/1/images.txt"   "$TMP_TEST/sparse_$N_VIEWS/1/images.txt"
ln -sf "$DATA_ROOT/sparse_$N_VIEWS/1/cameras.txt"  "$TMP_TEST/sparse_$N_VIEWS/1/cameras.txt"

# ===============================
# 1. INIT  (fast MASt3R → Colmap)
# ===============================
python3 InstantSplat/init_geo.py \
    --source_path "$TMP_TRAIN" \
    --model_path "$OUT_DIR" \
    --ckpt_path "$CKPT" \
    --n_views "$N_VIEWS" \
    --niter "$INIT_NITER"

# ===============================
# 2. TRAIN  (fast GS training)
# ===============================
python3 InstantSplat/train.py \
    --source_path "$TMP_TRAIN" \
    --model_path "$OUT_DIR" \
    --n_views "$N_VIEWS" \
    --iterations "$TRAIN_ITER"

# ===============================
# 3. RENDER (ONLY 4 TEST IMAGES)
# ===============================
python3 InstantSplat/render.py \
    --source_path "$TMP_TEST" \
    --model_path "$OUT_DIR" \
    --n_views "$N_VIEWS" \
    --eval \
    --skip_train \
    --iterations "$RENDER_ITER" \
    --optim_test_pose_iter "$POSE_OPT_ITER"

echo "[HW4_2] Done. Test PNG saved in:"
echo "   $OUT_DIR/test/ours_${RENDER_ITER}/renders/"
