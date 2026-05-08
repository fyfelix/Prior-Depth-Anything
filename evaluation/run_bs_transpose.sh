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
DEFAULT_DATASET_PATH="${REPO_ROOT}/data/TRansPose/sequences/dc_testset.jsonl"
DEFAULT_INTRINSICS_PATH="${REPO_ROOT}/data/TRansPose/sequences/intrinsics.txt"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_bs_transpose.sh [priorda_ckpt=ckpts/prior_depth_anything_vitb_1_1.pth] [mde_ckpt=ckpts/depth_anything_v2_vitl.pth] [camera_type=l515] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]

Environment overrides:
  PRIORDA_CKPT          Prior-Depth-Anything checkpoint file.
  MDE_CKPT              Depth Anything V2 checkpoint file.
  DATASET_PATH          TRansPose JSONL path. Default: data/TRansPose/sequences/dc_testset.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory.
  BATCH_SIZE            Inference batch size. Default: 1
  NUM_WORKERS           Inference DataLoader workers. Default: 0
  MAX_SAMPLES           Maximum samples to evaluate. 0 means all samples. Default: 0
  DEVICE                Torch device, e.g. cuda:0.
  PATTERN               Optional Prior-Depth-Anything sparse pattern.
  SAVE_VIS              Save 3x2 grid visualizations. Default: false
  INTRINSICS_PATH       Camera intrinsics text file. Default: data/TRansPose/sequences/intrinsics.txt
  PC_ROT_X_DEG          Point cloud view rotation around X axis. Default: 25.0
  PC_ROT_Y_DEG          Point cloud view rotation around Y axis. Default: 15.0
  PC_KNN_K              KNN neighbors for predicted point cloud filtering. Default: 16
  PC_KNN_STD_RATIO      KNN std ratio threshold. Default: 2.0
  DISABLE_PC_KNN_FILTER Disable predicted point cloud KNN filtering when true. Default: false
  COARSE_ONLY           Use coarse stage only. Default: false
  DOUBLE_GLOBAL         Use double-global conditioning. Default: false
  PRIOR_COVER           Preserve all prior pixels for sparse patterns. Default: false
  DOWN_FILL_MODE        linear/global/knn for downscale_* patterns. Default: linear
  CLAMP_TO_DEPTH_RANGE  Clip predictions to dataset depth-range. Default: false
  PYTHON_BIN            Python executable. Default: python, falling back to python3

TRansPose is fixed to raw-type=l515. Use auto/none/null for either checkpoint
to let Prior-Depth-Anything download weights from Hugging Face.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

priorda_ckpt="${1:-${PRIORDA_CKPT:-${DEFAULT_PRIORDA_CKPT}}}"
mde_ckpt="${2:-${MDE_CKPT:-${DEFAULT_MDE_CKPT}}}"
camera_type="${3:-l515}"
frozen_size="${4:-vitl}"
conditioned_size="${5:-vitb}"
version="${6:-1.1}"
cleanup_npy="${7:-false}"

dataset_path="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
intrinsics_path="${INTRINSICS_PATH:-${DEFAULT_INTRINSICS_PATH}}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-false}"
coarse_only="${COARSE_ONLY:-false}"
double_global="${DOUBLE_GLOBAL:-false}"
prior_cover="${PRIOR_COVER:-false}"
down_fill_mode="${DOWN_FILL_MODE:-linear}"
clamp_to_depth_range="${CLAMP_TO_DEPTH_RANGE:-false}"
pc_rot_x_deg="${PC_ROT_X_DEG:-25.0}"
pc_rot_y_deg="${PC_ROT_Y_DEG:-15.0}"
pc_knn_k="${PC_KNN_K:-16}"
pc_knn_std_ratio="${PC_KNN_STD_RATIO:-2.0}"
disable_pc_knn_filter="${DISABLE_PC_KNN_FILTER:-false}"

if [[ "${camera_type}" != "l515" ]]; then
    echo "TRansPose dataset only supports l515 raw type, got: ${camera_type}" >&2
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

if [[ ! -f "${dataset_path}" ]]; then
    echo "missing TRansPose dataset JSONL: ${dataset_path}" >&2
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
dataset_name="$(basename "${dataset_path}")"
dataset_stub="${dataset_name%%.*}"
output_dir="${OUTPUT_DIR:-${model_dir}/transpose_${dataset_stub}_${model_stub}_data_${camera_type}}"
ckpt_label="${mde_ckpt}\\${priorda_ckpt}"

save_vis_arg=(--save-vis "${save_vis}")
pc_knn_filter_arg=()
if [[ "${disable_pc_knn_filter}" == "true" ]]; then
    pc_knn_filter_arg=(--disable-pc-knn-filter)
fi

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
echo "intrinsics path: ${intrinsics_path}"
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
    --intrinsics-path "${intrinsics_path}"
    --pc-rot-x-deg "${pc_rot_x_deg}"
    --pc-rot-y-deg "${pc_rot_y_deg}"
    --pc-knn-k "${pc_knn_k}"
    --pc-knn-std-ratio "${pc_knn_std_ratio}"
    "${save_vis_arg[@]}"
    "${pc_knn_filter_arg[@]}"
)

if [[ -n "${DEVICE:-}" ]]; then
    infer_cmd+=(--device "${DEVICE}")
fi
if [[ -n "${PATTERN:-}" ]]; then
    infer_cmd+=(--pattern "${PATTERN}")
fi

"${infer_cmd[@]}"

echo "evaluating the model on TRansPose"
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
