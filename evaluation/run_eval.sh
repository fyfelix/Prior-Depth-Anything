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
DEFAULT_DATASET_PATH="${REPO_ROOT}/data/HAMMER/test_filled_d435.jsonl"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/output"

usage() {
    cat <<'EOF'
Usage:
  ./evaluation/run_eval.sh [priorda_ckpt=ckpts/prior_depth_anything_vitb_1_1.pth] [mde_ckpt=ckpts/depth_anything_v2_vitl.pth] [camera_type=d435] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]

Environment overrides:
  PRIORDA_CKPT          Prior-Depth-Anything checkpoint file.
                        Default: ckpts/prior_depth_anything_vitb_1_1.pth
  MDE_CKPT              Depth Anything V2 checkpoint file.
                        Default: ckpts/depth_anything_v2_vitl.pth
  DATASET_PATH          HAMMER JSONL path. Default: data/HAMMER/test_filled_d435.jsonl
  OUTPUT_DIR            Base output root. Default: evaluation/output
                        Each run writes to OUTPUT_DIR/YYYY-mm-dd_HH-MM-SS/
                        with predictions/ and visualizations/ subdirectories.
  BATCH_SIZE            Inference path batch size. Default: 1
  NUM_WORKERS           Inference DataLoader workers. Default: 0
  MAX_SAMPLES           Maximum samples to evaluate. 0 means all samples. Default: 0
  DEVICE                Torch device, e.g. cuda:0.
  PATTERN               Optional Prior-Depth-Anything sparse pattern.
  SAVE_VIS              Save *_promptda_vis.jpg visualizations. Default: true
  COARSE_ONLY           Use coarse stage only. Default: false
  PRIOR_COVER           Preserve all prior pixels for sparse patterns. Default: false
  DOWN_FILL_MODE        linear/global/knn for downscale_* patterns. Default: linear
  CLAMP_TO_DEPTH_RANGE  Clip predictions to HAMMER depth-range. Default: false
  PYTHON_BIN            Python executable. Default: python, falling back to python3

Default checkpoint label:
  ckpts/depth_anything_v2_vitl.pth\prior_depth_anything_vitb_1_1.pth

Use auto/none/null to let Prior-Depth-Anything download weights from Hugging Face.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

priorda_ckpt="${1:-${PRIORDA_CKPT:-${DEFAULT_PRIORDA_CKPT}}}"
mde_ckpt="${2:-${MDE_CKPT:-${DEFAULT_MDE_CKPT}}}"
camera_type="${3:-d435}"
frozen_size="${4:-vitl}"
conditioned_size="${5:-vitb}"
version="${6:-1.1}"
cleanup_npy="${7:-false}"

dataset_path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
output_dir="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
save_vis="${SAVE_VIS:-true}"
coarse_only="${COARSE_ONLY:-false}"
prior_cover="${PRIOR_COVER:-false}"
down_fill_mode="${DOWN_FILL_MODE:-linear}"
clamp_to_depth_range="${CLAMP_TO_DEPTH_RANGE:-false}"
max_samples="${MAX_SAMPLES:-0}"

if ! [[ "${max_samples}" =~ ^[0-9]+$ ]]; then
    echo "MAX_SAMPLES must be a non-negative integer, got: ${max_samples}" >&2
    exit 1
fi

timestamp="$(date +'%Y-%m-%d_%H-%M-%S')"
run_output_dir="${output_dir}/${timestamp}"
prediction_dir="${run_output_dir}/predictions"
visualization_dir="${run_output_dir}/visualizations"

if [[ "${priorda_ckpt}" == "none" || "${priorda_ckpt}" == "null" ]]; then
    priorda_ckpt="auto"
fi

if [[ "${mde_ckpt}" == "none" || "${mde_ckpt}" == "null" ]]; then
    mde_ckpt="auto"
fi

if [[ ! -f "${dataset_path}" ]]; then
    echo "missing HAMMER dataset JSONL: ${dataset_path}" >&2
    exit 1
fi

if [[ "${priorda_ckpt}" != "auto" ]]; then
    if [[ ! -f "${priorda_ckpt}" ]]; then
        echo "missing Prior-Depth-Anything checkpoint: ${priorda_ckpt}" >&2
        exit 1
    fi
fi

if [[ "${mde_ckpt}" != "auto" ]]; then
    if [[ ! -f "${mde_ckpt}" ]]; then
        echo "missing Depth Anything V2 checkpoint: ${mde_ckpt}" >&2
        exit 1
    fi
fi

ckpt_label="${mde_ckpt}\\${priorda_ckpt}"

mkdir -p "${prediction_dir}" "${visualization_dir}"

echo "model class: prior_depth_anything.PriorDepthAnything"
echo "version: ${version}"
echo "frozen model size: ${frozen_size}"
echo "conditioned model size: ${conditioned_size}"
echo "prior checkpoint: ${priorda_ckpt}"
echo "mde checkpoint: ${mde_ckpt}"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "output root: ${output_dir}"
echo "run output dir: ${run_output_dir}"
echo "prediction dir: ${prediction_dir}"
echo "visualization dir: ${visualization_dir}"
echo "batch size: ${batch_size}"
echo "max samples: ${max_samples}"
echo "save vis: ${save_vis}"
echo "cleanup npy: ${cleanup_npy}"

infer_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py"
    --dataset "${dataset_path}"
    --raw-type "${camera_type}"
    --output "${run_output_dir}"
    --prediction-dir "${prediction_dir}"
    --visualization-dir "${visualization_dir}"
    --frozen-model-size "${frozen_size}"
    --conditioned-model-size "${conditioned_size}"
    --version "${version}"
    --batch-size "${batch_size}"
    --num-workers "${num_workers}"
    --max-samples "${max_samples}"
    --priorda-ckpt "${priorda_ckpt}"
    --mde-ckpt "${mde_ckpt}"
    --save-vis "${save_vis}"
    --coarse-only "${coarse_only}"
    --prior-cover "${prior_cover}"
    --down-fill-mode "${down_fill_mode}"
    --clamp-to-depth-range "${clamp_to_depth_range}"
)

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
    --output "${run_output_dir}"
    --prediction-dir "${prediction_dir}"
    --raw-type "${camera_type}"
    --max-samples "${max_samples}"
)

if [[ -n "${DEVICE:-}" ]]; then
    eval_cmd+=(--device "${DEVICE}")
fi

time "${eval_cmd[@]}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${prediction_dir}"
    find "${prediction_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi
