#!/usr/bin/env python3

import argparse
import json
import os
import sys
from os.path import join

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

cv2 = None
np = None
torch = None
logger = None
DataLoader = None
tqdm = None
HAMMERDataset = None
PriorDepthAnything = None
log_img = None


def load_runtime_dependencies():
    global cv2, np, torch, logger, DataLoader, tqdm
    global HAMMERDataset, PriorDepthAnything, log_img

    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    from loguru import logger as _logger
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm import tqdm as _tqdm

    from dataset import HAMMERDataset as _HAMMERDataset
    from prior_depth_anything import PriorDepthAnything as _PriorDepthAnything
    from prior_depth_anything.utils import log_img as _log_img

    cv2 = _cv2
    np = _np
    torch = _torch
    logger = _logger
    DataLoader = _DataLoader
    tqdm = _tqdm
    HAMMERDataset = _HAMMERDataset
    PriorDepthAnything = _PriorDepthAnything
    log_img = _log_img


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def optional_path(value):
    if value is None:
        return None
    if value.lower() in ("", "none", "null", "auto"):
        return None
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prior-Depth-Anything inference for HAMMER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="HAMMER JSONL path")
    parser.add_argument("--output", default="output_dir", help="Output directory")
    parser.add_argument(
        "--raw-type",
        required=True,
        choices=["d435", "l515", "tof"],
        help="HAMMER raw depth field used as Prior-Depth-Anything prior",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=optional_path,
        default=None,
        help="Directory containing prior_depth_anything_<size>[_1_1].pth; use auto/none for HF download",
    )
    parser.add_argument(
        "--mde-dir",
        type=optional_path,
        default=None,
        help="Directory containing depth_anything_v2_<size>.pth; use auto/none for HF download",
    )
    parser.add_argument(
        "--frozen-model-size",
        choices=["vits", "vitb", "vitl"],
        default="vitl",
        help="Depth Anything V2 backbone size for coarse stage",
    )
    parser.add_argument(
        "--conditioned-model-size",
        choices=["vits", "vitb"],
        default="vitb",
        help="Prior-Depth-Anything conditioned model size",
    )
    parser.add_argument("--version", choices=["1.0", "1.1"], default="1.1")
    parser.add_argument("--coarse-only", type=str2bool, default=False)
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional Prior-Depth-Anything sparse sampling pattern; default uses raw depth directly",
    )
    parser.add_argument("--double-global", type=str2bool, default=False)
    parser.add_argument("--prior-cover", type=str2bool, default=False)
    parser.add_argument(
        "--down-fill-mode",
        choices=["linear", "global", "knn"],
        default="linear",
        help="Fill mode used only by downscale_* sparse patterns",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--image-min", type=float, default=0.1)
    parser.add_argument("--image-max", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-vis",
        nargs="?",
        const=True,
        type=str2bool,
        default=True,
        help="Save *_pred_depth.png visualizations",
    )
    parser.add_argument("--clamp-to-depth-range", type=str2bool, default=False)
    return parser.parse_args()


def load_depth_meters(depth_path, depth_scale, max_depth=None):
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not load depth from {depth_path}")
        depth = np.asarray(depth).astype(np.float32) / depth_scale

    depth = np.squeeze(depth).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    if max_depth is not None:
        depth[depth > max_depth] = 0.0
    return depth


def read_depth_shape(depth_path):
    if depth_path.endswith(".npy"):
        return np.squeeze(np.load(depth_path)).shape[-2:]
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not load depth from {depth_path}")
    return depth.shape[:2]


def sample_id_from_rgb_path(rgb_path):
    parts = rgb_path.split("/")
    scene_name = parts[-4]
    stem = os.path.splitext(os.path.basename(rgb_path))[0]
    return f"{scene_name}#{stem}"


def build_model(args):
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        logger.warning(
            "Prior-Depth-Anything uses torch_cluster KNN in the completion stage; "
            "CUDA is strongly recommended for full inference."
        )

    model = PriorDepthAnything(
        device=device,
        version=args.version,
        mde_dir=args.mde_dir,
        ckpt_dir=args.ckpt_dir,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        coarse_only=args.coarse_only,
    )
    model.eval()
    return model, device


def save_visualization(pred, output_dir, name, image_min, image_max):
    vis = pred.copy()
    path = join(output_dir, f"{name}_pred_depth.png")
    log_img(
        vis,
        path,
        valids=vis > 0.0001,
        scale=image_max - image_min,
        shift=image_min,
    )


def inference(args):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file does not exist: {args.dataset}")
    load_runtime_dependencies()
    os.makedirs(args.output, exist_ok=True)

    dataset = HAMMERDataset(args.dataset, args.raw_type)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, device = build_model(args)
    args.resolved_model_module = "prior_depth_anything"
    args.resolved_model_class = "PriorDepthAnything"
    args.output_kind = "metric_depth_meter"
    args.device = device

    with open(join(args.output, "args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    min_depth, max_depth = dataset.depth_range
    logger.info(
        "Running HAMMER inference with PriorDepthAnything "
        f"version={args.version}, frozen={args.frozen_model_size}, "
        f"conditioned={args.conditioned_model_size}, raw_type={args.raw_type}"
    )

    for batch_items in tqdm(dataloader, desc="Processing HAMMER samples"):
        rgb_paths, raw_depth_paths, gt_depth_paths = batch_items

        for rgb_path, raw_depth_path, gt_depth_path in zip(
            rgb_paths, raw_depth_paths, gt_depth_paths
        ):
            rgb_path = str(rgb_path)
            raw_depth_path = str(raw_depth_path)
            gt_depth_path = str(gt_depth_path)
            name = sample_id_from_rgb_path(rgb_path)

            raw_depth = load_depth_meters(
                raw_depth_path, args.depth_scale, max_depth=args.max_depth
            )

            pred = model.infer_one_sample(
                image=rgb_path,
                prior=raw_depth,
                geometric=None,
                pattern=args.pattern,
                double_global=args.double_global,
                prior_cover=args.prior_cover,
                visualize=False,
                down_fill_mode=args.down_fill_mode,
            )
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred).astype(np.float32)
            pred[~np.isfinite(pred)] = 0.0

            gt_shape = read_depth_shape(gt_depth_path)
            if pred.shape != gt_shape:
                pred = cv2.resize(
                    pred,
                    (gt_shape[1], gt_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)

            if args.clamp_to_depth_range:
                pred = np.clip(pred, min_depth, max_depth).astype(np.float32)

            np.save(join(args.output, f"{name}.npy"), pred)

            if args.save_vis:
                save_visualization(
                    pred, args.output, name, args.image_min, args.image_max
                )


if __name__ == "__main__":
    inference(parse_arguments())
