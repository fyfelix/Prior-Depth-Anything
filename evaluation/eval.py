import argparse
import json
import os
from datetime import datetime
from os.path import exists, join

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

cv2 = None
np = None
pd = None
torch = None
tqdm = None
DataLoader = None
DEVICE = "cpu"
abs_relative_difference = None
rmse_linear = None
delta1_acc = None
mae_linear = None
delta4_acc_105 = None
delta5_acc110 = None
load_test_dataset = None
sample_name_for_sample = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prior-Depth-Anything depth evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HAMMER, ClearPose, DREDS, or TRansPose JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_dir",
        help="Metric and eval metadata output directory",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Directory containing .npy predictions; defaults to --output/predictions",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw type; ClearPose/DREDS use d435, TRansPose uses l515",
    )
    parser.add_argument(
        "--input-size", type=int, default=518, help="Input size for inference"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument(
        "--max-depth", type=float, default=6.0, help="Maximum valid depth value"
    )
    parser.add_argument(
        "--image-min", type=float, default=0.1, help="Minimum valid depth value"
    )
    parser.add_argument(
        "--image-max", type=float, default=5.0, help="Maximum valid depth value"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device used for metric computation. Defaults to cuda when available, otherwise cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of dataset samples to evaluate; 0 means all samples",
    )
    return parser.parse_args()


def load_runtime_dependencies():
    global cv2, np, pd, torch, tqdm, DataLoader, DEVICE
    global abs_relative_difference, rmse_linear, delta1_acc, mae_linear
    global delta4_acc_105, delta5_acc110
    global load_test_dataset, sample_name_for_sample

    import cv2 as _cv2
    import numpy as _np
    import pandas as _pd
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import Dataset as _Dataset
    from tqdm import tqdm as _tqdm

    from dataset import load_test_dataset as _load_test_dataset
    from dataset import sample_name_for_sample as _sample_name_for_sample
    from utils.metric import abs_relative_difference as _abs_relative_difference
    from utils.metric import delta1_acc as _delta1_acc
    from utils.metric import delta4_acc_105 as _delta4_acc_105
    from utils.metric import delta5_acc110 as _delta5_acc110
    from utils.metric import mae_linear as _mae_linear
    from utils.metric import rmse_linear as _rmse_linear

    cv2 = _cv2
    np = _np
    pd = _pd
    torch = _torch
    tqdm = _tqdm
    DataLoader = _DataLoader
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    abs_relative_difference = _abs_relative_difference
    rmse_linear = _rmse_linear
    delta1_acc = _delta1_acc
    mae_linear = _mae_linear
    delta4_acc_105 = _delta4_acc_105
    delta5_acc110 = _delta5_acc110
    load_test_dataset = _load_test_dataset
    sample_name_for_sample = _sample_name_for_sample
    return _Dataset


def load_gt_depth(depth_path, depth_scale, max_depth, min_depth):
    depth_GT = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_GT is None:
        raise ValueError(f"Could not load GT depth from {depth_path}")
    depth_GT = np.asarray(depth_GT).astype(np.float32) / depth_scale
    valid_mask = (depth_GT >= min_depth) & (depth_GT <= max_depth)
    depth_GT[~valid_mask] = min_depth
    return depth_GT, valid_mask


def align_prediction_shape(pred, gt_shape, dataset_kind, name):
    if pred.shape == gt_shape:
        return pred

    if dataset_kind != "dreds":
        raise ValueError(
            f"Prediction/GT shape mismatch for {name}: "
            f"dataset_kind={dataset_kind}, pred_shape={pred.shape}, gt_shape={gt_shape}"
        )

    if pred.ndim != 2 or len(gt_shape) != 2:
        raise ValueError(
            f"DREDS evaluation expects 2D depth maps for {name}: "
            f"pred_shape={pred.shape}, gt_shape={gt_shape}"
        )

    gt_height, gt_width = gt_shape
    return cv2.resize(
        pred.astype(np.float32, copy=False),
        (gt_width, gt_height),
        interpolation=cv2.INTER_NEAREST,
    )


def build_eval_dataset_class(DatasetBase):
    class EvalDataset(DatasetBase):
        def __init__(self, dataset, output_path, args, depth_scale, align=False):
            self.dataset = dataset
            self.prediction_path = args.prediction_dir or join(output_path, "predictions")
            self.legacy_prediction_path = output_path
            self.args = args
            self.depth_scale = depth_scale
            self.align = align

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            depth_GT, valid_mask = load_gt_depth(
                sample[2],
                self.depth_scale,
                self.args.max_depth,
                self.args.min_depth,
            )
            name = sample_name_for_sample(self.args.dataset_kind, sample)

            pred_path = join(self.prediction_path, name + ".npy")
            if not exists(pred_path) and self.args.dataset_kind != "transpose":
                pred_path = join(self.legacy_prediction_path, name + ".npy")
            if not exists(pred_path):
                if self.args.dataset_kind == "transpose":
                    raise FileNotFoundError(
                        f"TRansPose prediction for {name} not found in "
                        f"{self.prediction_path}"
                    )
                raise FileNotFoundError(
                    f"Prediction for {name} not found in "
                    f"{self.prediction_path} or {self.legacy_prediction_path}"
                )

            pred = np.load(pred_path)
            pred = align_prediction_shape(
                pred, depth_GT.shape, self.args.dataset_kind, name
            )

            pred_invalid_mask = np.logical_or(np.isnan(pred), np.isinf(pred))
            if pred_invalid_mask.sum() > 0:
                valid_mask = valid_mask & ~pred_invalid_mask

            if self.align:
                depth_GT_reshaped = depth_GT[valid_mask].reshape((-1, 1))
                pred_reshaped = pred[valid_mask].reshape((-1, 1))

                _ones = np.ones_like(pred_reshaped)
                A = np.concatenate([pred_reshaped, _ones], axis=-1)
                X = np.linalg.lstsq(A, depth_GT_reshaped, rcond=None)[0]
                scale, shift = X
                pred_reshaped = scale * pred_reshaped + shift
                pred_reshaped = np.clip(
                    pred_reshaped, a_min=self.args.min_depth, a_max=None
                )

                return {
                    "name": name,
                    "pred": pred_reshaped.astype(np.float32),
                    "gt": depth_GT_reshaped.astype(np.float32),
                    "mask": np.ones_like(pred_reshaped, dtype=bool),
                    "is_aligned": True,
                }

            return {
                "name": name,
                "pred": pred.astype(np.float32),
                "gt": depth_GT.astype(np.float32),
                "mask": valid_mask.astype(bool),
                "is_aligned": False,
            }

    return EvalDataset


def main():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()

    if args.max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {args.max_samples}")

    DatasetBase = load_runtime_dependencies()

    output_dir = args.output
    args.prediction_dir = args.prediction_dir or join(args.output, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    eval_device = args.device or DEVICE
    args.device = eval_device

    dataset, dataset_kind = load_test_dataset(
        args.dataset, args.raw_type, max_samples=args.max_samples
    )
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale
    depth_scale = args.depth_scale

    with open(join(output_dir, "eval_args.json"), "w") as f:
        json.dump(vars(args), f)

    min_depth = dataset.depth_range[0]
    max_depth = dataset.depth_range[1]

    args.min_depth = min_depth
    args.max_depth = max_depth

    print(
        "min depth is updated and set to ",
        min_depth,
        "and max depth is updated and set to ",
        max_depth,
    )

    all_metrics = []
    ALIGN = False

    EvalDataset = build_eval_dataset_class(DatasetBase)
    eval_dataset = EvalDataset(dataset, output_dir, args, depth_scale, align=ALIGN)

    batch_size = 1 if ALIGN else 32
    num_workers = 0 if ALIGN or not eval_device.startswith("cuda") else 8

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=eval_device.startswith("cuda"),
    )

    for batch in tqdm(loader):
        names = batch["name"]

        pred_depth_ts = batch["pred"].to(eval_device, non_blocking=True)
        gt_depth_ts = batch["gt"].to(eval_device, non_blocking=True)
        mask_ts = batch["mask"].to(eval_device, non_blocking=True)

        l1 = mae_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        rmse = rmse_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        abs_rel = abs_relative_difference(
            pred_depth_ts, gt_depth_ts, mask_ts, reduction="none"
        )
        d4 = delta4_acc_105(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        d5 = delta5_acc110(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        d1 = delta1_acc(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")

        batch_len = len(names)
        l1_cpu = l1.detach().cpu().numpy()
        rmse_cpu = rmse.detach().cpu().numpy()
        abs_rel_cpu = abs_rel.detach().cpu().numpy()
        d4_cpu = d4.detach().cpu().numpy()
        d5_cpu = d5.detach().cpu().numpy()
        d1_cpu = d1.detach().cpu().numpy()
        for i in range(batch_len):
            metrics = {
                "name": names[i],
                "L1": l1_cpu[i],
                "rmse_linear": rmse_cpu[i],
                "abs_relative_difference": abs_rel_cpu[i],
                "delta4_acc_105": d4_cpu[i],
                "delta5_acc110": d5_cpu[i],
                "delta1_acc": d1_cpu[i],
            }
            all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    all_metrics_mean = all_metrics.mean(numeric_only=True).to_frame().T

    all_metrics.to_csv(
        join(output_dir, f"all_metrics_{current_time}_{ALIGN}.csv"), index=False
    )
    all_metrics_mean.to_json(
        join(output_dir, f"mean_metrics_{current_time}_{ALIGN}.json"),
        orient="records",
        lines=True,
        force_ascii=False,
    )

    from loguru import logger

    logger.info(f"save dir: {output_dir}")


if __name__ == "__main__":
    main()
