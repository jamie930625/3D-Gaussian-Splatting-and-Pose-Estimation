import os
import sys
import torch
import numpy as np
import logging
import warnings
import gc
from tqdm import tqdm
import glob
from utils import closed_form_inverse_se3

# --- Dust3R imports & Local Imports ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DUST3R_PARENT_DIR = os.path.join(SCRIPT_DIR, 'dust3r')
if DUST3R_PARENT_DIR not in sys.path:
    sys.path.append(DUST3R_PARENT_DIR)


from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from utils import get_c2w_rotation_from_gt_dict, get_c2w_translation_from_gt_dict

from args import parse_args
from metrics import se3_to_relative_pose_error, print_summary_report
from utils import set_random_seeds


def load_dust3r_model(device, model_path_arg):
    print(f"Initializing Dust3R model from {model_path_arg}...")
    model_name = model_path_arg
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    return model


# =====================================================================
# load_samples_from_files (你原本的版本已經正確，我保持不動)
# =====================================================================
import os
import sys
import torch
import numpy as np
import logging
import warnings
import gc
from tqdm import tqdm
import glob

def load_samples_from_files(index_txt_path, gt_npy_path, data_root, interpolated_dir, use_original_endpoints, test_only=False):
    samples = []
    gt_data_dict = {}

    if not test_only:
        print(f"Loading Ground Truth from: {gt_npy_path}")
        if not gt_npy_path or not os.path.exists(gt_npy_path):
            raise FileNotFoundError(f"GT .npy file not found: {gt_npy_path}. (Use --test_only if no GT available)")
        gt_data_dict = np.load(gt_npy_path, allow_pickle=True).item()
        print(f"Loaded {len(gt_data_dict)} GT entries.")
    else:
        print("Running in --test_only mode. Ground truth will not be loaded.")

    # Parse index txt
    print(f"Loading samples from: {index_txt_path}")
    if not os.path.exists(index_txt_path):
        raise FileNotFoundError(f"Index .txt file not found: {index_txt_path}")

    with open(index_txt_path, 'r') as f:
        lines = f.readlines()

    data_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('idx img1 img2'):
            data_started = True
            continue
        if not data_started or line.startswith('---'):
            continue

        parts = line.split()
        if len(parts) >= 3:
            idx_key_str = parts[0]

            try:
                idx_key_int = int(idx_key_str)
                gt_raw = gt_data_dict.get(idx_key_int, None)

                img1_rel = parts[1]
                img2_rel = parts[2]

                # Determine image paths
                image_paths = []

                # MODE 3 — Combined: Original endpoints + interpolated middle frames
                if use_original_endpoints:
                    if not data_root or not interpolated_dir:
                        raise ValueError("--use_original_endpoints requires both --data_root and --interpolated_dir")

                    p1 = os.path.join(data_root, img1_rel)
                    p2 = os.path.join(data_root, img2_rel)
                    if not os.path.exists(p1) or not os.path.exists(p2):
                        raise FileNotFoundError(f"Original image not found: {p1} or {p2}")

                    interp_dir = os.path.join(interpolated_dir, idx_key_str, "dynamicrafter")
                    all_pngs = glob.glob(os.path.join(interp_dir, "*.png"))

                    def fnum(p):
                        try: return int(os.path.basename(p).split("frame")[-1].split(".")[0])
                        except: return 999999

                    sorted_pngs = sorted(all_pngs, key=fnum)
                    mid_frames = sorted_pngs[1:-1]

                    image_paths = [p1] + mid_frames + [p2]

                # MODE 2 — Interpolated only
                elif interpolated_dir:
                    interp_dir = os.path.join(interpolated_dir, idx_key_str, "dynamicrafter")
                    all_pngs = glob.glob(os.path.join(interp_dir, "*.png"))

                    def fnum(p):
                        try: return int(os.path.basename(p).split("frame")[-1].split(".")[0])
                        except: return 999999

                    sorted_pngs = sorted(all_pngs, key=fnum)
                    image_paths = sorted_pngs

                # MODE 1 — Original pair only
                elif data_root:
                    p1 = os.path.join(data_root, img1_rel)
                    p2 = os.path.join(data_root, img2_rel)
                    if not os.path.exists(p1) or not os.path.exists(p2):
                        raise FileNotFoundError(f"Original image not found: {p1} or {p2}")
                    image_paths = [p1, p2]

                samples.append({
                    'scene_name': idx_key_str,
                    'image_paths': image_paths,
                    'gt': gt_raw
                })

            except Exception as e:
                print(f"Warning: error processing idx {idx_key_str}: {e}")

    print(f"Loaded {len(samples)} valid samples.")
    return samples



# =====================================================================
#  GT 轉換 utility
# =====================================================================

def read_gt_dict(gt_raw):
    if gt_raw is None:
        return None

    gt_rot1 = get_c2w_rotation_from_gt_dict(gt_raw['img1'])
    gt_rot2 = get_c2w_rotation_from_gt_dict(gt_raw['img2'])
    rot_pair = np.stack([gt_rot1, gt_rot2], axis=0)

    t1 = get_c2w_translation_from_gt_dict(gt_raw['img1'], gt_rot1)
    t2 = get_c2w_translation_from_gt_dict(gt_raw['img2'], gt_rot2)

    if t1 is not None and t2 is not None:
        t_pair = np.stack([t1, t2])
        t_tensor = torch.from_numpy(t_pair).double()
    else:
        t_tensor = torch.full((2, 3), float('nan'), dtype=torch.double)

    rot_tensor = torch.from_numpy(rot_pair).double()
    t_tensor = t_tensor.unsqueeze(-1)

    gt_se3 = torch.cat([rot_tensor, t_tensor], dim=2)
    return gt_se3



# =====================================================================
#  ⭐⭐⭐⭐⭐ 這裡是最重要的：完成 TODO
# =====================================================================

def inference_one_scene(model, sample, device, args):
    """
    Performs inference for one scene/sample using Dust3R model.
    sample: dict with keys:
        'scene_name': str
        'image_paths': list[str]
        'gt': dict or None
    """
    image_paths = sample['image_paths']
    scene_name = sample['scene_name']
    gt_data_raw = sample['gt']

    # ---------------------------
    # 1. Load images
    # ---------------------------
    images = load_images(image_paths, size=args.dust3r_image_size)  # list of np arrays

    # ---------------------------
    # 2. Create DUSt3R image pairs
    # ---------------------------
    # For 2-image wide-baseline → 2 symmetric pairs
    # For multi-frame interpolated → N(N−1) pairs
    pairs = make_pairs(
        images,
        scene_graph='complete',
        prefilter=None,
        symmetrize=True
    )

    # ---------------------------
    # 3. Run DUSt3R model
    # ---------------------------
    output = inference(
        pairs,
        model,
        device,
        batch_size=args.batch_size
    )

    # ---------------------------
    # 4. Global Alignment
    # ---------------------------
    if len(images) == 2:
        # Wide-baseline: Only convert outputs into poses
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

    else:
        # Interpolated sequence: Need full BA alignment
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

        _ = scene.compute_global_alignment(
            init="mst",
            niter=args.dust3r_align_niter,
            schedule=args.dust3r_align_schedule,
            lr=args.dust3r_align_lr
        )

    # ---------------------------
    # 5. Extract aligned camera poses (C2W)
    # ---------------------------
    poses_from_scene = scene.get_im_poses()           # list of (4x4) tensors
    poses_np_cpu = [p.detach().cpu().numpy() for p in poses_from_scene]

    if len(poses_np_cpu) != len(images):
        raise ValueError(
            f"[{scene_name}] Expected {len(images)} pose matrices, "
            f"but Dust3R returned {len(poses_np_cpu)}"
        )

    pred_extrinsic = np.stack(poses_np_cpu, axis=0)   # (N, 4, 4)

    # ---------------------------
    # 6. Read GT if exists
    # ---------------------------
    gt_pose = read_gt_dict(gt_data_raw)

    return pred_extrinsic, gt_pose


# =====================================================================
# main():
# =====================================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_random_seeds(args.seed)
    model = load_dust3r_model(device, args.model_path)

    samples_to_process = load_samples_from_files(
        args.index_txt_path,
        args.gt_npy_path,
        args.data_root,
        args.interpolated_dir,
        args.use_original_endpoints,
        args.test_only
    )

    all_r_errors = []
    all_t_errors = []
    all_predicted_poses = {}

    for sample in tqdm(samples_to_process, desc="Processing Samples"):
        pred_extrinsic, gt_c2w_se3_pair = inference_one_scene(model, sample, device, args)
        pred_extrinsic = torch.from_numpy(pred_extrinsic).to(device)

        pred_c2w_pair = pred_extrinsic[[0, -1]]

        if gt_c2w_se3_pair is None:
            gt_c2w_se3_pair = torch.full(
                (2, 3, 4), float('nan'),
                dtype=pred_extrinsic.dtype,
                device=device
            )
        else:
            gt_c2w_se3_pair = gt_c2w_se3_pair.to(device)

        add_row = torch.tensor([0,0,0,1], dtype=gt_c2w_se3_pair.dtype, device=device).expand(2,1,4)
        gt_c2w_4x4 = torch.cat([gt_c2w_se3_pair, add_row], dim=1)

        r_err, t_err = se3_to_relative_pose_error(pred_c2w_pair, gt_c2w_4x4, 2)

        all_r_errors.append(r_err.item())
        all_t_errors.append(t_err.item())

        all_predicted_poses[sample['scene_name']] = pred_c2w_pair.detach().cpu().numpy()

    # Save
    if args.save_pose_path:
        out_dir = os.path.dirname(args.save_pose_path)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nSaving predicted poses to {args.save_pose_path}...")
        np.save(args.save_pose_path, all_predicted_poses)
        print("Saved OK.")

    print_summary_report(all_r_errors, all_t_errors, args.eval_mode)
    print("\nEvaluation finished.")


if __name__ == "__main__":
    for pkg in ['numpy', 'torch', 'tqdm']:
        try: __import__(pkg)
        except ImportError: sys.exit(1)
    args = parse_args()
    main()
