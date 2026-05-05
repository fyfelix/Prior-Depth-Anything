#!/usr/bin/env python3
"""One-shot Prior-Depth-Anything iBims inference and official evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, PIPELINE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval_official import prepare_workspace, resolve_root, run_official_eval  # noqa: E402
from infer_to_mat import (  # noqa: E402
    DEFAULT_MDE_CKPT,
    DEFAULT_PRIORDA_CKPT,
    SYNTHETIC_RAW_DIR_NAME,
    build_model,
    checkpoint_label,
    normalize_max_samples,
    optional_path,
    run_manifest_inference,
    str2bool,
)


ALL_LEVELS = ["easy", "medium", "hard", "extreme"]
RESULT_METRIC_KEYS = [
    "rel",
    "sq_rel",
    "rms",
    "log10",
    "thr1",
    "thr2",
    "thr3",
    "dde_0",
    "dde_p",
    "dde_m",
    "pe_fla",
    "pe_ori",
    "dbe_acc",
    "dbe_com",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prior-Depth-Anything iBims inference and official eval across difficulty levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument("--ibims-root", default="data/ibims1", help="iBims dataset root")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=ALL_LEVELS,
        default=ALL_LEVELS,
        help="Difficulty levels to process",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PIPELINE_DIR / "output"),
        help="Base output root used when --run-dir is not set",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Exact output run directory; defaults to <output-dir>/ibims_<ckpt_label>_<timestamp>",
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
        help="Use only the first N samples per level for smoke testing; 0 means all samples",
    )
    parser.add_argument(
        "--clamp-to-depth-range",
        type=str2bool,
        default=False,
        help="Clip predictions to the manifest depth-range before saving",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference and use existing predictions in --run-dir",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip official evaluation and only run inference",
    )
    return parser.parse_args()


def default_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = checkpoint_label(args.priorda_ckpt, args.mde_ckpt)
    return resolve_root(args.output_dir) / f"ibims_{label}_{timestamp}"


def manifest_for_level(ibims_root: Path, level: str) -> Path:
    return ibims_root / SYNTHETIC_RAW_DIR_NAME / "manifests" / f"ibims_{level}.jsonl"


def parse_eval_stdout(text: str) -> Dict[str, float]:
    results = {}
    in_block = False
    for line in text.splitlines():
        if not in_block:
            if line.strip() == "Results:":
                in_block = True
            continue
        if line.strip() == "":
            continue
        match = re.match(r"(\S+)\s*=\s*([\d.eE+\-]+)", line.strip())
        if match:
            results[match.group(1)] = float(match.group(2))
        else:
            break
    return results


def run_inference(args: argparse.Namespace, run_dir: Path) -> None:
    ibims_root = resolve_root(args.ibims_root)
    model, device = build_model(args)
    print("Model: prior_depth_anything.PriorDepthAnything")
    print(f"Device: {device}")

    for level in args.levels:
        manifest_path = manifest_for_level(ibims_root, level)
        if not manifest_path.is_file():
            print(f"[skip infer] manifest not found: {manifest_path}")
            continue

        pred_dir = run_dir / "predictions" / level
        stats = run_manifest_inference(
            str(manifest_path),
            str(pred_dir),
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
                **vars(args),
                "resolved_model_module": "prior_depth_anything",
                "resolved_model_class": "PriorDepthAnything",
                "output_kind": "metric_depth_meter",
                "alignment": "none",
            },
        )
        print(f"[infer] {level}: wrote {stats['num_predictions']} predictions to {pred_dir}")


def run_evaluation(args: argparse.Namespace, run_dir: Path) -> None:
    ibims_root = resolve_root(args.ibims_root)
    all_metrics = {}

    for level in args.levels:
        pred_dir = run_dir / "predictions" / level
        if not pred_dir.is_dir():
            print(f"[skip eval] prediction dir not found: {pred_dir}")
            continue

        workspace = run_dir / "official_eval" / level / "workspace"
        log_path = run_dir / "official_eval" / level / "official_eval_stdout.txt"
        print(f"[eval] {level}: preparing workspace {workspace}")
        eval_script, names = prepare_workspace(
            ibims_root,
            pred_dir,
            workspace,
            normalize_max_samples(args.max_samples),
        )
        print(f"[eval] {level}: validated {len(names)} predictions")
        print(f"[eval] {level}: running official eval")

        result = run_official_eval(eval_script, workspace, log_path, check=False, echo=False)
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            print(f"[eval] {level}: official eval failed, log saved to {log_path}", file=sys.stderr)
            raise SystemExit(result.returncode)

        metrics = parse_eval_stdout(result.stdout)
        all_metrics[level] = metrics
        if metrics:
            print(f"[eval] {level}: extracted {len(metrics)} metrics")
        else:
            print(f"[eval] {level}: WARNING - no metrics parsed from output")
            print(result.stdout[-500:] if result.stdout else "(empty stdout)")

    if all_metrics:
        summary_path = run_dir / "eval_summary.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["level"] + RESULT_METRIC_KEYS)
            writer.writeheader()
            for level in [item for item in ALL_LEVELS if item in all_metrics]:
                writer.writerow(
                    {"level": level, **{key: all_metrics[level].get(key) for key in RESULT_METRIC_KEYS}}
                )
        with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as file:
            json.dump(all_metrics, file, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"Eval summary saved to: {summary_path}")
        print_metrics_table(all_metrics)
    else:
        print("[eval] No metrics collected.")


def print_metrics_table(all_metrics: Dict[str, Dict[str, float]]) -> None:
    levels = [level for level in ALL_LEVELS if level in all_metrics]
    all_keys: List[str] = []
    for metrics in all_metrics.values():
        for key in metrics:
            if key not in all_keys:
                all_keys.append(key)

    col_width = 10
    header = f"{'metric':<12}" + "".join(f"{level:>{col_width}}" for level in levels)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for key in all_keys:
        row = f"{key:<12}"
        for level in levels:
            value = all_metrics[level].get(key)
            row += f"{value:{col_width}.4f}" if value is not None else f"{'-':>{col_width}}"
        print(row)
    print(sep)


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    run_dir = resolve_root(args.run_dir) if args.run_dir else default_run_dir(args).resolve()

    if args.skip_infer and not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist (needed when --skip-infer): {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if not args.skip_infer:
        run_inference(args, run_dir)

    if not args.skip_eval:
        run_evaluation(args, run_dir)


if __name__ == "__main__":
    main()
