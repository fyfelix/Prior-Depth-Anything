#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

DEFAULT_CKPT_DIR="${REPO_ROOT}/ckpts"
DEFAULT_DATASET_PATH="${REPO_ROOT}/data/HAMMER/test.jsonl"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/output"

usage() {
    cat <<'EOF'
Usage:
  ./evaluation/run_eval.sh [ckpt_dir=ckpts] [camera_type=d435] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]

Environment overrides:
  CKPT_DIR              Directory containing prior_depth_anything_<size>[_1_1].pth. Default: ckpts
  DATASET_PATH          HAMMER JSONL path. Default: data/HAMMER/test.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory. Default: evaluation/output
  MDE_DIR               Directory containing depth_anything_v2_<size>.pth. Default: ckpts
  BATCH_SIZE            Inference path batch size. Default: 1
  NUM_WORKERS           Inference DataLoader workers. Default: 0
  DEVICE                Torch device, e.g. cuda:0.
  PATTERN               Optional Prior-Depth-Anything sparse pattern.
  SAVE_VIS              Save *_pred_depth.png visualizations. Default: true
  COARSE_ONLY           Use coarse stage only. Default: false
  PRIOR_COVER           Preserve all prior pixels for sparse patterns. Default: false
  DOWN_FILL_MODE        linear/global/knn for downscale_* patterns. Default: linear
  CLAMP_TO_DEPTH_RANGE  Clip predictions to HAMMER depth-range. Default: false
  PYTHON_BIN            Python executable. Default: python, falling back to python3

ckpt_dir should contain prior_depth_anything_<conditioned_size>[_1_1].pth.
MDE_DIR should contain depth_anything_v2_<frozen_size>.pth.
Use auto/none/null to let Prior-Depth-Anything download weights from Hugging Face.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ckpt_dir="${1:-${CKPT_DIR:-${DEFAULT_CKPT_DIR}}}"
camera_type="${2:-d435}"
frozen_size="${3:-vitl}"
conditioned_size="${4:-vitb}"
version="${5:-1.1}"
cleanup_npy="${6:-false}"

dataset_path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
output_dir="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
mde_dir="${MDE_DIR:-${DEFAULT_CKPT_DIR}}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
save_vis="${SAVE_VIS:-true}"
coarse_only="${COARSE_ONLY:-false}"
prior_cover="${PRIOR_COVER:-false}"
down_fill_mode="${DOWN_FILL_MODE:-linear}"
clamp_to_depth_range="${CLAMP_TO_DEPTH_RANGE:-false}"

ckpt_label="${ckpt_dir}"
if [[ "${ckpt_label}" == "auto" || "${ckpt_label}" == "none" || "${ckpt_label}" == "null" ]]; then
    ckpt_label="hf_auto"
fi
priorda_ckpt="hf_auto"
mde_ckpt="hf_auto"

if [[ "${mde_dir}" == "none" || "${mde_dir}" == "null" ]]; then
    mde_dir="auto"
fi

if [[ ! -f "${dataset_path}" ]]; then
    echo "missing HAMMER dataset JSONL: ${dataset_path}" >&2
    exit 1
fi

if [[ "${ckpt_dir}" != "auto" && "${ckpt_dir}" != "none" && "${ckpt_dir}" != "null" ]]; then
    priorda_ckpt="${ckpt_dir}/prior_depth_anything_${conditioned_size}.pth"
    if [[ "${version}" == "1.1" ]]; then
        priorda_ckpt="${ckpt_dir}/prior_depth_anything_${conditioned_size}_1_1.pth"
    fi
    if [[ ! -f "${priorda_ckpt}" ]]; then
        echo "missing Prior-Depth-Anything checkpoint: ${priorda_ckpt}" >&2
        exit 1
    fi
fi

if [[ "${mde_dir}" != "auto" ]]; then
    mde_ckpt="${mde_dir}/depth_anything_v2_${frozen_size}.pth"
    if [[ ! -f "${mde_ckpt}" ]]; then
        echo "missing Depth Anything V2 checkpoint: ${mde_ckpt}" >&2
        exit 1
    fi
fi

echo "model class: prior_depth_anything.PriorDepthAnything"
echo "version: ${version}"
echo "frozen model size: ${frozen_size}"
echo "conditioned model size: ${conditioned_size}"
echo "ckpt dir: ${ckpt_dir}"
echo "mde dir: ${mde_dir}"
echo "prior checkpoint: ${priorda_ckpt}"
echo "mde checkpoint: ${mde_ckpt}"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "output dir: ${output_dir}"
echo "batch size: ${batch_size}"
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
    --save-vis "${save_vis}"
    --coarse-only "${coarse_only}"
    --prior-cover "${prior_cover}"
    --down-fill-mode "${down_fill_mode}"
    --clamp-to-depth-range "${clamp_to_depth_range}"
)

if [[ "${ckpt_dir}" != "auto" && "${ckpt_dir}" != "none" && "${ckpt_dir}" != "null" ]]; then
    infer_cmd+=(--ckpt-dir "${ckpt_dir}")
fi

if [[ "${mde_dir}" != "auto" ]]; then
    infer_cmd+=(--mde-dir "${mde_dir}")
fi

if [[ -n "${DEVICE:-}" ]]; then
    infer_cmd+=(--device "${DEVICE}")
fi

if [[ -n "${PATTERN:-}" ]]; then
    infer_cmd+=(--pattern "${PATTERN}")
fi

"${infer_cmd[@]}"

echo "evaluating predictions"
eval_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py"
    --encoder "${conditioned_size}"
    --model-path "${ckpt_label}"
    --dataset "${dataset_path}"
    --output "${output_dir}"
    --raw-type "${camera_type}"
)

if [[ -n "${DEVICE:-}" ]]; then
    eval_cmd+=(--device "${DEVICE}")
fi

time "${eval_cmd[@]}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}"
    find "${output_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi
