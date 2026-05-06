#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

DEFAULT_PRIORDA_CKPT="${REPO_ROOT}/ckpts/prior_depth_anything_vitb_1_1.pth"
DEFAULT_MDE_CKPT="${REPO_ROOT}/ckpts/depth_anything_v2_vitl.pth"
DEFAULT_DATASET_PATH="${REPO_ROOT}/data/clearpose/test.jsonl"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_clearpose.sh [priorda_ckpt=ckpts/prior_depth_anything_vitb_1_1.pth] [mde_ckpt=ckpts/depth_anything_v2_vitl.pth] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]

Environment overrides:
  PRIORDA_CKPT          Prior-Depth-Anything checkpoint file.
  MDE_CKPT              Depth Anything V2 checkpoint file.
  DATASET_PATH          ClearPose JSONL path. Default: data/clearpose/test.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory.
  BATCH_SIZE            Inference batch size. Default: 1
  NUM_WORKERS           Inference DataLoader workers. Default: 0
  MAX_SAMPLES           Maximum samples to evaluate. 0 means all samples. Default: 0
  DEVICE                Torch device, e.g. cuda:0.
  PATTERN               Optional Prior-Depth-Anything sparse pattern.
  SAVE_VIS              Save *_promptda_vis.jpg visualizations. Default: true
  COARSE_ONLY           Use coarse stage only. Default: false
  DOUBLE_GLOBAL         Use double-global conditioning. Default: false
  PRIOR_COVER           Preserve all prior pixels for sparse patterns. Default: false
  DOWN_FILL_MODE        linear/global/knn for downscale_* patterns. Default: linear
  CLAMP_TO_DEPTH_RANGE  Clip predictions to dataset depth-range. Default: false
  PYTHON_BIN            Python executable. Default: python, falling back to python3

ClearPose is fixed to raw-type=d435. Use auto/none/null for either checkpoint
to let Prior-Depth-Anything download weights from Hugging Face.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

priorda_ckpt="${1:-${PRIORDA_CKPT:-${DEFAULT_PRIORDA_CKPT}}}"
mde_ckpt="${2:-${MDE_CKPT:-${DEFAULT_MDE_CKPT}}}"
frozen_size="${3:-vitl}"
conditioned_size="${4:-vitb}"
version="${5:-1.1}"
cleanup_npy="${6:-false}"
camera_type="d435"

dataset_path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-true}"
coarse_only="${COARSE_ONLY:-false}"
double_global="${DOUBLE_GLOBAL:-false}"
prior_cover="${PRIOR_COVER:-false}"
down_fill_mode="${DOWN_FILL_MODE:-linear}"
clamp_to_depth_range="${CLAMP_TO_DEPTH_RANGE:-false}"

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

if [[ ! -f "${dataset_path}" ]]; then
    echo "missing ClearPose dataset JSONL: ${dataset_path}" >&2
    exit 1
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
output_dir="${OUTPUT_DIR:-${model_dir}/clearpose_${model_stub}_data_${camera_type}}"
ckpt_label="${mde_ckpt}\\${priorda_ckpt}"

save_vis_arg=(--save-vis "${save_vis}")

echo "model class: prior_depth_anything.PriorDepthAnything"
echo "version: ${version}"
echo "frozen model size: ${frozen_size}"
echo "conditioned model size: ${conditioned_size}"
echo "prior checkpoint: ${priorda_ckpt}"
echo "mde checkpoint: ${mde_ckpt}"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "output dir: ${output_dir}"
echo "batch size: ${batch_size}"
echo "max samples: ${max_samples}"
echo "save vis: ${save_vis}"
echo "cleanup npy: ${cleanup_npy}"

infer_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py"
    --dataset "${dataset_path}"
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

echo "evaluating the model on ClearPose"
eval_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py"
    --encoder "${conditioned_size}"
    --model-path "${ckpt_label}"
    --dataset "${dataset_path}"
    --output "${output_dir}"
    --raw-type "${camera_type}"
    --max-samples "${max_samples}"
)

if [[ -n "${DEVICE:-}" ]]; then
    eval_cmd+=(--device "${DEVICE}")
fi

time "${eval_cmd[@]}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
    if [[ -d "${output_dir}/predictions" ]]; then
        find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
    fi
fi
