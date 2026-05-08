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

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

cv2 = None
np = None
torch = None
logger = None
DataLoader = None
tqdm = None
load_test_dataset = None
PriorDepthAnything = None
log_img = None
sample_name_for_sample = None

DEFAULT_INTRINSICS_PATH = "data/TRansPose/sequences/intrinsics.txt"


def load_runtime_dependencies():
    global cv2, np, torch, logger, DataLoader, tqdm
    global load_test_dataset, sample_name_for_sample, PriorDepthAnything, log_img

    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    from loguru import logger as _logger
    from torch.utils.data import DataLoader as _DataLoader
    from tqdm import tqdm as _tqdm

    from dataset import load_test_dataset as _load_test_dataset
    from dataset import sample_name_for_sample as _sample_name_for_sample
    from prior_depth_anything import PriorDepthAnything as _PriorDepthAnything
    from prior_depth_anything.utils import log_img as _log_img

    cv2 = _cv2
    np = _np
    torch = _torch
    logger = _logger
    DataLoader = _DataLoader
    tqdm = _tqdm
    load_test_dataset = _load_test_dataset
    sample_name_for_sample = _sample_name_for_sample
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
        description="Prior-Depth-Anything inference for HAMMER, ClearPose, DREDS, or TRansPose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="HAMMER, ClearPose, DREDS, or TRansPose JSONL path")
    parser.add_argument("--output", default="output_dir", help="Run metadata output directory")
    parser.add_argument(
        "--prediction-dir",
        default=None,
        help="Directory for .npy predictions; defaults to --output/predictions",
    )
    parser.add_argument(
        "--visualization-dir",
        default=None,
        help="Directory for visualization files; defaults to --output/visualizations",
    )
    parser.add_argument(
        "--raw-type",
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw depth source used as Prior-Depth-Anything prior; ClearPose/DREDS use d435",
    )
    parser.add_argument(
        "--priorda-ckpt",
        type=optional_path,
        default=None,
        help="Prior-Depth-Anything checkpoint file; use auto/none for HF download",
    )
    parser.add_argument(
        "--mde-ckpt",
        type=optional_path,
        default=None,
        help="Depth Anything V2 checkpoint file; use auto/none for HF download",
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of dataset samples to process; 0 means all samples",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-vis",
        nargs="?",
        const=True,
        type=str2bool,
        default=True,
        help="Save *_promptda_vis.jpg visualizations",
    )
    parser.add_argument(
        "--intrinsics-path",
        default=DEFAULT_INTRINSICS_PATH,
        help="Camera intrinsics text file for TRansPose point cloud visualization",
    )
    parser.add_argument(
        "--pc-rot-x-deg",
        type=float,
        default=25.0,
        help="Point cloud view rotation around X axis in degrees",
    )
    parser.add_argument(
        "--pc-rot-y-deg",
        type=float,
        default=15.0,
        help="Point cloud view rotation around Y axis in degrees",
    )
    parser.add_argument(
        "--pc-knn-k",
        type=int,
        default=16,
        help="KNN neighbors for predicted point cloud floater filtering",
    )
    parser.add_argument(
        "--pc-knn-std-ratio",
        type=float,
        default=2.0,
        help="Mean-distance std ratio threshold for predicted point cloud filtering",
    )
    parser.add_argument(
        "--disable-pc-knn-filter",
        action="store_true",
        help="Disable KNN filtering for predicted point cloud visualization",
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
        mde_path=args.mde_ckpt,
        ckpt_path=args.priorda_ckpt,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        coarse_only=args.coarse_only,
    )
    model.eval()
    return model, device


def save_visualization(pred, output_dir, name, image_min, image_max):
    vis = pred.copy()
    path = join(output_dir, f"{name}_promptda_vis.jpg")
    log_img(
        vis,
        path,
        valids=vis > 0.0001,
        scale=image_max - image_min,
        shift=image_min,
    )


def _ensure_image_runtime():
    global cv2, np
    if cv2 is None:
        import cv2 as _cv2

        cv2 = _cv2
    if np is None:
        import numpy as _np

        np = _np
    return cv2, np


def read_rgb_image(rgb_path):
    _cv2, _np = _ensure_image_runtime()
    rgb_bgr = _cv2.imread(rgb_path, _cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    return _np.asarray(rgb_bgr[:, :, ::-1])


def image_grid(images, rows, cols):
    _cv2, _np = _ensure_image_runtime()
    if len(images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(images)}")

    target_h, target_w = images[0].shape[:2]
    canvas = _np.zeros((rows * target_h, cols * target_w, 3), dtype=_np.uint8)

    for idx, image in enumerate(images):
        image = _np.asarray(image)
        if image.ndim == 2:
            image = _np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[:2] != (target_h, target_w):
            image = _cv2.resize(image, (target_w, target_h), interpolation=_cv2.INTER_LINEAR)
        row = idx // cols
        col = idx % cols
        canvas[
            row * target_h : (row + 1) * target_h,
            col * target_w : (col + 1) * target_w,
        ] = image.astype(_np.uint8, copy=False)

    return canvas


def colorize_depth(depth, max_depth):
    _cv2, _np = _ensure_image_runtime()
    depth = _np.asarray(depth, dtype=_np.float32).squeeze()
    valid = (depth > 0.0) & _np.isfinite(depth)
    if max_depth <= 0:
        max_depth = max(float(depth[valid].max()), 1.0) if valid.any() else 1.0

    norm = _np.clip(depth / max_depth, 0.0, 1.0)
    gray = (norm * 255.0).astype(_np.uint8)
    color_map = getattr(_cv2, "COLORMAP_TURBO", _cv2.COLORMAP_JET)
    colored_bgr = _cv2.applyColorMap(gray, color_map)
    colored = colored_bgr[:, :, ::-1]
    colored[~valid] = 0
    return colored


def load_intrinsics(path):
    _, _np = _ensure_image_runtime()
    intrinsics = _np.loadtxt(path, dtype=_np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics matrix must have shape (3, 3), got {intrinsics.shape}")
    return intrinsics


def scale_intrinsics(intrinsics, orig_hw, new_hw):
    scaled = intrinsics.copy()
    sy = new_hw[0] / orig_hw[0]
    sx = new_hw[1] / orig_hw[1]
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    return scaled


def resize_to(image, target_hw, interpolation=None):
    _cv2, _np = _ensure_image_runtime()
    image = _np.asarray(image)
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image
    if interpolation is None:
        interpolation = _cv2.INTER_LINEAR
    return _cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def filter_pointcloud_knn(points, colors, k=16, std_ratio=2.0):
    _, _np = _ensure_image_runtime()
    if k < 1 or points.shape[0] <= k:
        return points, colors

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return points, colors

    neighbor_count = min(k + 1, points.shape[0])
    try:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=neighbor_count, workers=-1)
    except Exception:
        return points, colors

    if distances.ndim == 1:
        return points, colors

    mean_distances = distances[:, 1:].mean(axis=1)
    finite = _np.isfinite(mean_distances)
    if not finite.any():
        return points, colors

    valid_mean_distances = mean_distances[finite]
    threshold = valid_mean_distances.mean() + std_ratio * valid_mean_distances.std()
    keep = finite & (mean_distances <= threshold)
    if not keep.any():
        return points, colors
    return points[keep], colors[keep]


def render_pointcloud_reproject(
    depth_map,
    intrinsics,
    rgb_img,
    rot_x_deg=25.0,
    rot_y_deg=15.0,
    bg_color=(255, 255, 255),
    knn_filter=True,
    knn_k=16,
    knn_std_ratio=2.0,
):
    _cv2, _np = _ensure_image_runtime()
    depth_map = _np.asarray(depth_map, dtype=_np.float32).squeeze()
    height, width = depth_map.shape
    rgb_img = resize_to(rgb_img, (height, width)).astype(_np.uint8, copy=False)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = _np.meshgrid(_np.arange(width), _np.arange(height))
    valid = (depth_map > 1e-8) & _np.isfinite(depth_map)
    if not valid.any():
        return _np.full((height, width, 3), bg_color, dtype=_np.uint8)

    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = _np.stack([x, y, z], axis=-1).astype(_np.float32, copy=False)
    colors = _np.clip(rgb_img, 0, 255).astype(_np.uint8)[valid]
    if knn_filter:
        points, colors = filter_pointcloud_knn(
            points,
            colors,
            k=knn_k,
            std_ratio=knn_std_ratio,
        )
        if points.shape[0] == 0:
            return _np.full((height, width, 3), bg_color, dtype=_np.uint8)

    center = points.mean(axis=0)
    points_centered = points - center

    rx = _np.radians(rot_x_deg)
    ry = _np.radians(rot_y_deg)
    cos_x, sin_x = _np.cos(rx), _np.sin(rx)
    cos_y, sin_y = _np.cos(ry), _np.sin(ry)

    x1 = points_centered[:, 0]
    y1 = points_centered[:, 1] * cos_x - points_centered[:, 2] * sin_x
    z1 = points_centered[:, 1] * sin_x + points_centered[:, 2] * cos_x
    x2 = x1 * cos_y + z1 * sin_y
    y2 = y1
    z2 = -x1 * sin_y + z1 * cos_y
    points_rot = _np.stack([x2, y2, z2], axis=-1) + center
    z_new = points_rot[:, 2]
    keep = z_new > 1e-4
    if not keep.any():
        return _np.full((height, width, 3), bg_color, dtype=_np.uint8)

    u_proj = points_rot[keep, 0] * fx / z_new[keep] + cx
    v_proj = points_rot[keep, 1] * fy / z_new[keep] + cy
    z_buf = z_new[keep]
    c_buf = colors[keep]

    pad = int(max(height, width) * 0.3)
    canvas_h, canvas_w = height + 2 * pad, width + 2 * pad
    ui = _np.round(u_proj + pad).astype(_np.int32)
    vi = _np.round(v_proj + pad).astype(_np.int32)

    in_bounds = (ui >= 0) & (ui < canvas_w) & (vi >= 0) & (vi < canvas_h)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z_buf = z_buf[in_bounds]
    c_buf = c_buf[in_bounds]
    if ui.size == 0:
        return _np.full((height, width, 3), bg_color, dtype=_np.uint8)

    order = _np.argsort(-z_buf)
    ui = ui[order]
    vi = vi[order]
    c_buf = c_buf[order]

    canvas = _np.full((canvas_h, canvas_w, 3), bg_color, dtype=_np.uint8)
    canvas[vi, ui] = c_buf

    filled = _np.zeros((canvas_h, canvas_w), dtype=_np.uint8)
    filled[vi, ui] = 255
    kernel = _np.ones((3, 3), dtype=_np.uint8)
    filled_dilated = _cv2.dilate(filled, kernel, iterations=1)
    holes = (filled_dilated > 0) & (filled == 0)
    if holes.any():
        for channel_idx in range(3):
            blurred = _cv2.blur(canvas[:, :, channel_idx].astype(_np.float32), (3, 3))
            canvas[:, :, channel_idx][holes] = blurred[holes].astype(_np.uint8)

    rows = _np.any(filled_dilated > 0, axis=1)
    cols = _np.any(filled_dilated > 0, axis=0)
    if rows.any() and cols.any():
        row_min, row_max = _np.where(rows)[0][[0, -1]]
        col_min, col_max = _np.where(cols)[0][[0, -1]]
        margin = 10
        row_min = max(0, row_min - margin)
        row_max = min(canvas_h - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(canvas_w - 1, col_max + margin)
        canvas = canvas[row_min : row_max + 1, col_min : col_max + 1]

    return resize_to(canvas, (height, width))


def create_visualizationv2(
    rgb_src,
    raw_depth,
    pred_depth,
    gt_depth,
    image_max,
    intrinsics,
    pc_rot_x_deg=25.0,
    pc_rot_y_deg=15.0,
    pc_knn_k=16,
    pc_knn_std_ratio=2.0,
    disable_pc_knn_filter=False,
):
    _, _np = _ensure_image_runtime()
    rgb_display = rgb_src.astype(_np.uint8, copy=False)
    raw_depth_colored = colorize_depth(raw_depth, image_max)
    pred_colored = colorize_depth(pred_depth, image_max)
    gt_depth_colored = colorize_depth(gt_depth, image_max)
    pred_pointcloud = render_pointcloud_reproject(
        pred_depth,
        intrinsics,
        rgb_display,
        rot_x_deg=pc_rot_x_deg,
        rot_y_deg=pc_rot_y_deg,
        knn_filter=not disable_pc_knn_filter,
        knn_k=pc_knn_k,
        knn_std_ratio=pc_knn_std_ratio,
    )
    gt_pointcloud = render_pointcloud_reproject(
        gt_depth,
        intrinsics,
        rgb_display,
        rot_x_deg=pc_rot_x_deg,
        rot_y_deg=pc_rot_y_deg,
        knn_filter=False,
    )
    target_hw = rgb_display.shape[:2]
    pred_pointcloud = resize_to(pred_pointcloud, target_hw)
    gt_pointcloud = resize_to(gt_pointcloud, target_hw)

    return image_grid(
        [
            rgb_display,
            raw_depth_colored,
            pred_colored,
            gt_depth_colored,
            pred_pointcloud,
            gt_pointcloud,
        ],
        3,
        2,
    )


def validate_transpose_visualization_args(args):
    if not os.path.exists(args.intrinsics_path):
        raise FileNotFoundError(f"Intrinsics path does not exist: {args.intrinsics_path}")
    if args.pc_knn_k < 1:
        raise ValueError(f"--pc-knn-k must be greater than 0, got {args.pc_knn_k}")
    if args.pc_knn_std_ratio < 0:
        raise ValueError(
            f"--pc-knn-std-ratio must be non-negative, got {args.pc_knn_std_ratio}"
        )


def save_transpose_visualization(
    rgb_path,
    raw_depth,
    pred,
    gt_depth_path,
    output_dir,
    name,
    args,
    intrinsics,
):
    _cv2, _np = _ensure_image_runtime()
    rgb_src = read_rgb_image(rgb_path)
    target_hw = rgb_src.shape[:2]
    raw_depth = resize_to(raw_depth, target_hw)
    pred = resize_to(pred, target_hw)
    gt_depth = load_depth_meters(gt_depth_path, args.depth_scale, max_depth=args.max_depth)
    gt_depth = resize_to(gt_depth, target_hw)
    scaled_intrinsics = scale_intrinsics(intrinsics, rgb_src.shape[:2], target_hw)

    grid_vis = create_visualizationv2(
        rgb_src,
        raw_depth,
        pred,
        gt_depth,
        args.image_max,
        scaled_intrinsics,
        pc_rot_x_deg=args.pc_rot_x_deg,
        pc_rot_y_deg=args.pc_rot_y_deg,
        pc_knn_k=args.pc_knn_k,
        pc_knn_std_ratio=args.pc_knn_std_ratio,
        disable_pc_knn_filter=args.disable_pc_knn_filter,
    )
    out_path = join(output_dir, f"{name}_grid_vis.jpg")
    _cv2.imwrite(out_path, _np.asarray(grid_vis)[:, :, ::-1])


def inference(args):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file does not exist: {args.dataset}")
    if args.max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {args.max_samples}")
    load_runtime_dependencies()
    args.prediction_dir = args.prediction_dir or join(args.output, "predictions")
    args.visualization_dir = args.visualization_dir or join(args.output, "visualizations")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.prediction_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(args.visualization_dir, exist_ok=True)

    dataset, dataset_kind = load_test_dataset(
        args.dataset, args.raw_type, max_samples=args.max_samples
    )
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale
    if args.save_vis and dataset_kind == "transpose":
        validate_transpose_visualization_args(args)
        transpose_intrinsics = load_intrinsics(args.intrinsics_path)
    else:
        transpose_intrinsics = None

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
        "Running HAMMER/ClearPose/DREDS/TRansPose inference with PriorDepthAnything "
        f"version={args.version}, frozen={args.frozen_model_size}, "
        f"conditioned={args.conditioned_model_size}, raw_type={args.raw_type}"
    )

    for batch_items in tqdm(dataloader, desc="Processing dataset samples"):
        if len(batch_items) == 4:
            rgb_paths, raw_depth_paths, gt_depth_paths, sample_names = batch_items
        else:
            rgb_paths, raw_depth_paths, gt_depth_paths = batch_items
            sample_names = None

        for sample_idx, (rgb_path, raw_depth_path, gt_depth_path) in enumerate(zip(
            rgb_paths, raw_depth_paths, gt_depth_paths
        )):
            rgb_path = str(rgb_path)
            raw_depth_path = str(raw_depth_path)
            gt_depth_path = str(gt_depth_path)
            if sample_names is None:
                sample = (rgb_path, raw_depth_path, gt_depth_path)
            else:
                sample = (
                    rgb_path,
                    raw_depth_path,
                    gt_depth_path,
                    sample_names[sample_idx],
                )
            name = sample_name_for_sample(dataset_kind, sample)

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
            if pred.shape != gt_shape and dataset_kind != "dreds":
                pred = cv2.resize(
                    pred,
                    (gt_shape[1], gt_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)

            if args.clamp_to_depth_range:
                pred = np.clip(pred, min_depth, max_depth).astype(np.float32)

            np.save(join(args.prediction_dir, f"{name}.npy"), pred)

            if args.save_vis:
                if dataset_kind == "transpose":
                    save_transpose_visualization(
                        rgb_path,
                        raw_depth,
                        pred,
                        gt_depth_path,
                        args.visualization_dir,
                        name,
                        args,
                        transpose_intrinsics,
                    )
                else:
                    save_visualization(
                        pred,
                        args.visualization_dir,
                        name,
                        args.image_min,
                        args.image_max,
                    )


if __name__ == "__main__":
    inference(parse_arguments())
