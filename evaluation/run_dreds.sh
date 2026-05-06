#!/usr/bin/env bash

set -euo pipefail

export OPENCV_IO_ENABLE_OPENEXR=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

DEFAULT_PRIORDA_CKPT="${REPO_ROOT}/ckpts/prior_depth_anything_vitb_1_1.pth"
DEFAULT_MDE_CKPT="${REPO_ROOT}/ckpts/depth_anything_v2_vitl.pth"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_dreds.sh [priorda_ckpt=ckpts/prior_depth_anything_vitb_1_1.pth] [mde_ckpt=ckpts/depth_anything_v2_vitl.pth] [variant=all] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]

Arguments:
  variant               catknown | catnovel | all. Default: all

Environment overrides:
  PRIORDA_CKPT          Prior-Depth-Anything checkpoint file.
  MDE_CKPT              Depth Anything V2 checkpoint file.
  DREDS_KNOWN_JSONL     DREDS catknown JSONL. Default: data/DREDS/test_std_catknown.jsonl
  DREDS_NOVEL_JSONL     DREDS catnovel JSONL. Default: data/DREDS/test_std_catnovel.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory for a single variant.
  OUTPUT_ROOT           Root directory for default per-variant outputs. Default: checkpoint directory
  BATCH_SIZE            Inference batch size. Default: 1
  NUM_WORKERS           Inference DataLoader workers. Default: 0
  MAX_SAMPLES           Maximum samples to evaluate. 0 means all samples. Default: 0
  DEVICE                Torch device, e.g. cuda:0.
  PATTERN               Optional Prior-Depth-Anything sparse pattern.
  SAVE_VIS              Save *_promptda_vis.jpg visualizations. Default: false
  COARSE_ONLY           Use coarse stage only. Default: false
  DOUBLE_GLOBAL         Use double-global conditioning. Default: false
  PRIOR_COVER           Preserve all prior pixels for sparse patterns. Default: false
  DOWN_FILL_MODE        linear/global/knn for downscale_* patterns. Default: linear
  CLAMP_TO_DEPTH_RANGE  Clip predictions to dataset depth-range. Default: false
  PYTHON_BIN            Python executable. Default: python, falling back to python3

DREDS uses EXR floating-point depth in meters. raw-type is passed as d435 only
to satisfy the shared Python CLI and is ignored by the DREDS dataset loader.
Use auto/none/null for either checkpoint to let Prior-Depth-Anything download
weights from Hugging Face.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

priorda_ckpt="${1:-${PRIORDA_CKPT:-${DEFAULT_PRIORDA_CKPT}}}"
mde_ckpt="${2:-${MDE_CKPT:-${DEFAULT_MDE_CKPT}}}"
variant="${3:-all}"
frozen_size="${4:-vitl}"
conditioned_size="${5:-vitb}"
version="${6:-1.1}"
cleanup_npy="${7:-false}"
camera_type="d435"

dreds_known_jsonl="${DREDS_KNOWN_JSONL:-${REPO_ROOT}/data/DREDS/test_std_catknown.jsonl}"
dreds_novel_jsonl="${DREDS_NOVEL_JSONL:-${REPO_ROOT}/data/DREDS/test_std_catnovel.jsonl}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-false}"
coarse_only="${COARSE_ONLY:-false}"
double_global="${DOUBLE_GLOBAL:-false}"
prior_cover="${PRIOR_COVER:-false}"
down_fill_mode="${DOWN_FILL_MODE:-linear}"
clamp_to_depth_range="${CLAMP_TO_DEPTH_RANGE:-false}"

if [[ "${variant}" == "all" && -n "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR can only be used with variant=catknown or variant=catnovel; use OUTPUT_ROOT for variant=all." >&2
    exit 2
fi

if ! [[ "${max_samples}" =~ ^[0-9]+$ ]]; then
    echo "MAX_SAMPLES must be a non-negative integer, got: ${max_samples}" >&2
    exit 1
fi

if [[ "${priorda_ckpt}" == "none" || "${priorda_ckpt}" == "null" ]]; then
    priorda_ckpt="auto"
fi
if [[ "${mde_ckpt}" == "none" || "${mde_ckpt}" == "null" ]]; then
    mde_ckpt="auto"
fi

if [[ "${priorda_ckpt}" != "auto" && ! -f "${priorda_ckpt}" ]]; then
    echo "missing Prior-Depth-Anything checkpoint: ${priorda_ckpt}" >&2
    exit 1
fi
if [[ "${mde_ckpt}" != "auto" && ! -f "${mde_ckpt}" ]]; then
    echo "missing Depth Anything V2 checkpoint: ${mde_ckpt}" >&2
    exit 1
fi

model_name="$(basename "${priorda_ckpt}")"
model_stub="${model_name%%.*}"
model_dir="$(dirname "${priorda_ckpt}")"
if [[ "${priorda_ckpt}" == "auto" ]]; then
    model_stub="auto"
    model_dir="${SCRIPT_DIR}/output"
fi
output_root="${OUTPUT_ROOT:-${model_dir}}"
ckpt_label="${mde_ckpt}\\${priorda_ckpt}"
save_vis_arg=(--save-vis "${save_vis}")

run_one_variant() {
    local label="$1"
    local jsonl_path="$2"
    local output_dir="${OUTPUT_DIR:-${output_root}/dreds_${label}_${model_stub}}"

    if [[ ! -f "${jsonl_path}" ]]; then
        echo "missing DREDS ${label} dataset JSONL: ${jsonl_path}" >&2
        exit 1
    fi

    echo "[${label}] model class: prior_depth_anything.PriorDepthAnything"
    echo "[${label}] version: ${version}"
    echo "[${label}] frozen model size: ${frozen_size}"
    echo "[${label}] conditioned model size: ${conditioned_size}"
    echo "[${label}] prior checkpoint: ${priorda_ckpt}"
    echo "[${label}] mde checkpoint: ${mde_ckpt}"
    echo "[${label}] dataset path: ${jsonl_path}"
    echo "[${label}] camera type: ${camera_type}"
    echo "[${label}] output dir: ${output_dir}"
    echo "[${label}] batch size: ${batch_size}"
    echo "[${label}] max samples: ${max_samples}"
    echo "[${label}] save vis: ${save_vis}"
    echo "[${label}] cleanup npy: ${cleanup_npy}"

    local infer_cmd=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py"
        --dataset "${jsonl_path}"
        --raw-type "${camera_type}"
        --output "${output_dir}"
        --frozen-model-size "${frozen_size}"
        --conditioned-model-size "${conditioned_size}"
        --version "${version}"
        --batch-size "${batch_size}"
        --num-workers "${num_workers}"
        --max-samples "${max_samples}"
        --priorda-ckpt "${priorda_ckpt}"
        --mde-ckpt "${mde_ckpt}"
        --coarse-only "${coarse_only}"
        --double-global "${double_global}"
        --prior-cover "${prior_cover}"
        --down-fill-mode "${down_fill_mode}"
        --clamp-to-depth-range "${clamp_to_depth_range}"
        "${save_vis_arg[@]}"
    )

    if [[ -n "${DEVICE:-}" ]]; then
        infer_cmd+=(--device "${DEVICE}")
    fi
    if [[ -n "${PATTERN:-}" ]]; then
        infer_cmd+=(--pattern "${PATTERN}")
    fi

    "${infer_cmd[@]}"

    echo "[${label}] evaluating the model on DREDS"
    local eval_cmd=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py"
        --encoder "${conditioned_size}"
        --model-path "${ckpt_label}"
        --dataset "${jsonl_path}"
        --output "${output_dir}"
        --raw-type "${camera_type}"
        --max-samples "${max_samples}"
    )

    if [[ -n "${DEVICE:-}" ]]; then
        eval_cmd+=(--device "${DEVICE}")
    fi

    time "${eval_cmd[@]}"

    if [[ "${cleanup_npy}" == "true" ]]; then
        echo "[${label}] cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
        if [[ -d "${output_dir}/predictions" ]]; then
            find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
        fi
    fi
}

case "${variant}" in
    catknown)
        run_one_variant catknown "${dreds_known_jsonl}"
        ;;
    catnovel)
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    all)
        run_one_variant catknown "${dreds_known_jsonl}"
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    *)
        echo "unknown DREDS variant: ${variant} (expected: catknown | catnovel | all)" >&2
        exit 1
        ;;
esac
