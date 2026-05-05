#!/usr/bin/env python3
"""Run Prior-Depth-Anything iBims inference and save official *_results.mat files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IBIMS_DEPTH_MAX_M = 50.0
IBIMS_DEPTH_SCALE = 65535.0 / IBIMS_DEPTH_MAX_M
SYNTHETIC_RAW_DIR_NAME = "ibims1_synthetic_raw_depth"
EXPECTED_SHAPE = (480, 640)
DEFAULT_PRIORDA_CKPT = "ckpts/prior_depth_anything_vitb_1_1.pth"
DEFAULT_MDE_CKPT = "ckpts/depth_anything_v2_vitl.pth"

cv2 = None
np = None
torch = None
savemat = None
tqdm = None
PriorDepthAnything = None


def load_runtime_dependencies() -> None:
    global cv2, np, torch, savemat, tqdm, PriorDepthAnything

    if PriorDepthAnything is not None:
        return

    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    from scipy.io import savemat as _savemat
    from tqdm import tqdm as _tqdm

    from prior_depth_anything import PriorDepthAnything as _PriorDepthAnything

    cv2 = _cv2
    np = _np
    torch = _torch
    savemat = _savemat
    tqdm = _tqdm
    PriorDepthAnything = _PriorDepthAnything


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in ("yes", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value.lower() in ("", "none", "null", "auto"):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prior-Depth-Anything inference for iBims and write official MAT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="iBims JSONL manifest path")
    parser.add_argument(
        "--priorda-ckpt",
        type=optional_path,
        default=DEFAULT_PRIORDA_CKPT,
        help="Prior-Depth-Anything checkpoint file; use auto/none for HF download",
    )
    parser.add_argument(
        "--mde-ckpt",
        type=optional_path,
        default=DEFAULT_MDE_CKPT,
        help="Depth Anything V2 checkpoint file; use auto/none for HF download",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction directory; defaults to evaluation_ibims/output/ibims_<ckpt_label>_<timestamp>/predictions/<level>",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Manifest batch size")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0. Defaults to cuda:0 when available, otherwise cpu",
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
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Depth clamp for raw input; defaults to manifest depth-range max",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Raw depth scale; defaults to each manifest row depth_scale",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use only the first N manifest rows for smoke testing; 0 means all samples",
    )
    parser.add_argument(
        "--clamp-to-depth-range",
        type=str2bool,
        default=False,
        help="Clip predictions to the manifest depth-range before saving",
    )
    return parser.parse_args()


def normalize_max_samples(max_samples: Optional[int]) -> Optional[int]:
    if max_samples is None or max_samples == 0:
        return None
    if max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    return max_samples


def resolve_root(path: str) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj
    return path_obj.resolve()


def resolve_optional_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    resolved = resolve_root(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return str(resolved)


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def select_device(device_name: Optional[str]) -> str:
    load_runtime_dependencies()
    if device_name is None or device_name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is not available: {device_name}")
    return device_name


def build_model(args: argparse.Namespace):
    load_runtime_dependencies()
    device = select_device(args.device)
    if device == "cpu":
        print(
            "WARNING: Prior-Depth-Anything uses torch_cluster KNN in the completion stage; "
            "CUDA is strongly recommended for full inference.",
            file=sys.stderr,
        )

    priorda_ckpt = resolve_optional_path(args.priorda_ckpt)
    mde_ckpt = resolve_optional_path(args.mde_ckpt)
    model = PriorDepthAnything(
        device=device,
        version=args.version,
        mde_path=mde_ckpt,
        ckpt_path=priorda_ckpt,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size,
        coarse_only=args.coarse_only,
    )
    model.eval()
    return model, device


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("dataset") != "ibims":
                raise ValueError(f"{manifest_path}:{line_number} is not an iBims row")
            for key in ("sample_id", "rgb", "raw_depth"):
                if key not in row:
                    raise ValueError(f"{manifest_path}:{line_number} missing required key: {key}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def infer_difficulty(manifest_path: Path, rows: List[Dict[str, Any]]) -> str:
    difficulty = rows[0].get("difficulty")
    if difficulty:
        return str(difficulty)
    stem = manifest_path.stem
    return stem[len("ibims_") :] if stem.startswith("ibims_") else stem


def checkpoint_stem(path: Optional[str], fallback: str) -> str:
    if path is None:
        return fallback
    return Path(path).stem


def sanitize_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "auto"


def checkpoint_label(priorda_ckpt: Optional[str], mde_ckpt: Optional[str]) -> str:
    mde_name = checkpoint_stem(mde_ckpt, "mde_auto")
    priorda_name = checkpoint_stem(priorda_ckpt, "priorda_auto")
    return sanitize_label(f"{mde_name}__{priorda_name}")


def default_output_dir(
    manifest_path: Path,
    rows: List[Dict[str, Any]],
    priorda_ckpt: Optional[str],
    mde_ckpt: Optional[str],
) -> Path:
    difficulty = infer_difficulty(manifest_path, rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = checkpoint_label(priorda_ckpt, mde_ckpt)
    return PROJECT_ROOT / "evaluation_ibims" / "output" / f"ibims_{label}_{timestamp}" / "predictions" / difficulty


def row_depth_scale(row: Dict[str, Any], cli_depth_scale: Optional[float]) -> float:
    if cli_depth_scale is not None:
        return cli_depth_scale
    return float(row.get("depth_scale", IBIMS_DEPTH_SCALE))


def row_depth_range(row: Dict[str, Any]) -> Tuple[float, float]:
    depth_range = row.get("depth-range", [0.01, IBIMS_DEPTH_MAX_M])
    return float(depth_range[0]), float(depth_range[1])


def row_max_depth(row: Dict[str, Any], cli_max_depth: Optional[float]) -> float:
    if cli_max_depth is not None:
        return cli_max_depth
    return row_depth_range(row)[1]


def read_single_channel(path: Path):
    load_runtime_dependencies()
    if path.suffix.lower() == ".npy":
        image = np.load(str(path))
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read depth image: {path}")
    if image.ndim == 3:
        image = image[:, :, 0]
    return np.squeeze(image)


def read_rgb_shape(path: Path) -> Tuple[int, int]:
    load_runtime_dependencies()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RGB image: {path}")
    return image.shape[:2]


def load_raw_depth(raw_depth_path: Path, depth_scale: float, max_depth: float):
    depth = read_single_channel(raw_depth_path).astype(np.float32) / depth_scale
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return depth


def normalize_prediction(pred_depth: Any, target_shape: Tuple[int, int], depth_range: Tuple[float, float], clamp: bool):
    load_runtime_dependencies()
    if torch is not None and isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    pred = np.asarray(pred_depth, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Expected HxW prediction, got shape {pred.shape}")
    if pred.shape != target_shape:
        pred = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = pred.astype(np.float32, copy=False)
    invalid = ~np.isfinite(pred) | (pred <= 0.0)
    if clamp:
        min_depth, max_depth = depth_range
        pred = np.clip(pred, min_depth, max_depth).astype(np.float32, copy=False)
    pred[invalid] = np.nan
    return pred


def iter_batches(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(vars(args))
    return out


def run_manifest_inference(
    manifest_path: str,
    output_dir: str,
    model: Any,
    device: str,
    batch_size: int = 1,
    depth_scale: Optional[float] = None,
    max_depth: Optional[float] = None,
    max_samples: Optional[int] = 0,
    pattern: Optional[str] = None,
    double_global: bool = False,
    prior_cover: bool = False,
    down_fill_mode: str = "linear",
    clamp_to_depth_range: bool = False,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    load_runtime_dependencies()
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    manifest = resolve_root(str(manifest_path))
    output = resolve_root(str(output_dir))
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = load_manifest(manifest)
    normalized_max_samples = normalize_max_samples(max_samples)
    if normalized_max_samples is not None:
        rows = rows[:normalized_max_samples]
    if not rows:
        raise ValueError(f"No samples selected from manifest: {manifest}")

    difficulty = infer_difficulty(manifest, rows)
    output.mkdir(parents=True, exist_ok=True)

    written = 0
    progress = tqdm(total=len(rows), desc=f"iBims {difficulty} inference")
    try:
        for batch_rows in iter_batches(rows, batch_size):
            for row in batch_rows:
                sample_id = str(row["sample_id"])
                rgb_path = resolve_path(manifest.parent, row["rgb"])
                raw_depth_path = resolve_path(manifest.parent, row["raw_depth"])

                raw_depth = load_raw_depth(
                    raw_depth_path,
                    row_depth_scale(row, depth_scale),
                    row_max_depth(row, max_depth),
                )
                rgb_shape = read_rgb_shape(rgb_path)
                if raw_depth.shape != rgb_shape:
                    raise ValueError(
                        f"RGB/depth shape mismatch for {sample_id}: "
                        f"rgb={rgb_shape}, depth={raw_depth.shape}"
                    )
                if raw_depth.shape != EXPECTED_SHAPE:
                    raise ValueError(f"{sample_id}: expected raw depth shape {EXPECTED_SHAPE}, got {raw_depth.shape}")

                pred_depth = model.infer_one_sample(
                    image=str(rgb_path),
                    prior=raw_depth,
                    geometric=None,
                    pattern=pattern,
                    double_global=double_global,
                    prior_cover=prior_cover,
                    visualize=False,
                    down_fill_mode=down_fill_mode,
                )
                pred_depth = normalize_prediction(
                    pred_depth,
                    raw_depth.shape,
                    row_depth_range(row),
                    clamp_to_depth_range,
                )
                if pred_depth.shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"{sample_id}: expected prediction shape {EXPECTED_SHAPE}, got {pred_depth.shape}"
                    )

                savemat(
                    output / f"{sample_id}_results.mat",
                    {"pred_depths": pred_depth.astype(np.float32, copy=False)},
                )
                written += 1
                progress.update(1)
    finally:
        progress.close()

    stats = {
        "difficulty": difficulty,
        "manifest": str(manifest),
        "output_dir": str(output),
        "num_predictions": written,
        "device_resolved": device,
    }
    metadata = dict(run_metadata or {})
    metadata.update(stats)
    with open(output / "infer_args.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, sort_keys=True, default=str)

    return stats


def main() -> None:
    args = parse_args()
    manifest_path = resolve_root(args.manifest)
    rows = load_manifest(manifest_path)
    output_dir = (
        resolve_root(args.output_dir)
        if args.output_dir
        else default_output_dir(manifest_path, rows, args.priorda_ckpt, args.mde_ckpt)
    )
    model, device = build_model(args)

    stats = run_manifest_inference(
        str(manifest_path),
        str(output_dir),
        model,
        device,
        batch_size=args.batch_size,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        pattern=args.pattern,
        double_global=args.double_global,
        prior_cover=args.prior_cover,
        down_fill_mode=args.down_fill_mode,
        clamp_to_depth_range=args.clamp_to_depth_range,
        run_metadata={
            **serializable_args(args),
            "priorda_ckpt": args.priorda_ckpt,
            "mde_ckpt": args.mde_ckpt,
            "resolved_model_module": "prior_depth_anything",
            "resolved_model_class": "PriorDepthAnything",
            "output_kind": "metric_depth_meter",
            "alignment": "none",
        },
    )
    print(f"Wrote {stats['num_predictions']} official iBims predictions to: {output_dir}")


if __name__ == "__main__":
    main()
