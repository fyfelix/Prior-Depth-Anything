#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PRIORDA_CKPT="${PROJECT_ROOT}/ckpts/prior_depth_anything_vitb_1_1.pth"
DEFAULT_MDE_CKPT="${PROJECT_ROOT}/ckpts/depth_anything_v2_vitl.pth"
DEFAULT_IBIMS_ROOT="${PROJECT_ROOT}/data/ibims1"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/output"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

usage() {
    cat <<'EOF'
Usage:
  ./evaluation_ibims/run_all.sh [priorda_ckpt=ckpts/prior_depth_anything_vitb_1_1.pth] [mde_ckpt=ckpts/depth_anything_v2_vitl.pth] [extra run_all.py args...]

Environment overrides:
  PRIORDA_CKPT          Prior-Depth-Anything checkpoint file.
  MDE_CKPT              Depth Anything V2 checkpoint file.
  IBIMS_ROOT            iBims dataset root. Default: data/ibims1
  OUTPUT_DIR            Base output root. Default: evaluation_ibims/output
  BATCH_SIZE            Inference batch size. Default: 1
  MAX_SAMPLES           Maximum samples per difficulty. 0 means all. Default: 0
  DEVICE                Torch device, e.g. cuda:0
  PATTERN               Optional Prior-Depth-Anything sparse pattern.
  COARSE_ONLY           Use coarse stage only. Default: false
  PRIOR_COVER           Preserve all prior pixels for sparse patterns. Default: false
  DOUBLE_GLOBAL         Use double-global conditioning. Default: false
  DOWN_FILL_MODE        linear/global/knn for downscale_* patterns. Default: linear
  CLAMP_TO_DEPTH_RANGE  Clip predictions to manifest depth-range. Default: false
  PYTHON_BIN            Python executable. Default: .venv/bin/python if present, otherwise python

Use auto/none/null for checkpoint args to let Prior-Depth-Anything download weights from Hugging Face.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
    priorda_ckpt="${1}"
    shift
else
    priorda_ckpt="${PRIORDA_CKPT:-${DEFAULT_PRIORDA_CKPT}}"
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
    mde_ckpt="${1}"
    shift
else
    mde_ckpt="${MDE_CKPT:-${DEFAULT_MDE_CKPT}}"
fi

ibims_root="${IBIMS_ROOT:-${DEFAULT_IBIMS_ROOT}}"
output_dir="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
batch_size="${BATCH_SIZE:-1}"
max_samples="${MAX_SAMPLES:-0}"
coarse_only="${COARSE_ONLY:-false}"
prior_cover="${PRIOR_COVER:-false}"
double_global="${DOUBLE_GLOBAL:-false}"
down_fill_mode="${DOWN_FILL_MODE:-linear}"
clamp_to_depth_range="${CLAMP_TO_DEPTH_RANGE:-false}"
frozen_model_size="${FROZEN_MODEL_SIZE:-vitl}"
conditioned_model_size="${CONDITIONED_MODEL_SIZE:-vitb}"
version="${VERSION:-1.1}"

if ! [[ "${max_samples}" =~ ^[0-9]+$ ]]; then
    echo "MAX_SAMPLES must be a non-negative integer, got: ${max_samples}" >&2
    exit 1
fi

skip_infer=false
for arg in "$@"; do
    if [[ "${arg}" == "--skip-infer" ]]; then
        skip_infer=true
        break
    fi
done

if [[ "${priorda_ckpt}" == "none" || "${priorda_ckpt}" == "null" ]]; then
    priorda_ckpt="auto"
fi

if [[ "${mde_ckpt}" == "none" || "${mde_ckpt}" == "null" ]]; then
    mde_ckpt="auto"
fi

if [[ "${skip_infer}" != "true" && "${priorda_ckpt}" != "auto" && ! -f "${priorda_ckpt}" ]]; then
    echo "missing Prior-Depth-Anything checkpoint: ${priorda_ckpt}" >&2
    exit 1
fi

if [[ "${skip_infer}" != "true" && "${mde_ckpt}" != "auto" && ! -f "${mde_ckpt}" ]]; then
    echo "missing Depth Anything V2 checkpoint: ${mde_ckpt}" >&2
    exit 1
fi

echo "model class: prior_depth_anything.PriorDepthAnything"
echo "version: ${version}"
echo "frozen model size: ${frozen_model_size}"
echo "conditioned model size: ${conditioned_model_size}"
echo "prior checkpoint: ${priorda_ckpt}"
echo "mde checkpoint: ${mde_ckpt}"
echo "ibims root: ${ibims_root}"
echo "output root: ${output_dir}"
echo "batch size: ${batch_size}"
echo "max samples: ${max_samples}"

cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all.py"
    --priorda-ckpt "${priorda_ckpt}"
    --mde-ckpt "${mde_ckpt}"
    --ibims-root "${ibims_root}"
    --output-dir "${output_dir}"
    --batch-size "${batch_size}"
    --max-samples "${max_samples}"
    --frozen-model-size "${frozen_model_size}"
    --conditioned-model-size "${conditioned_model_size}"
    --version "${version}"
    --coarse-only "${coarse_only}"
    --prior-cover "${prior_cover}"
    --double-global "${double_global}"
    --down-fill-mode "${down_fill_mode}"
    --clamp-to-depth-range "${clamp_to_depth_range}"
)

if [[ -n "${DEVICE:-}" ]]; then
    cmd+=(--device "${DEVICE}")
fi

if [[ -n "${PATTERN:-}" ]]; then
    cmd+=(--pattern "${PATTERN}")
fi

cd "${PROJECT_ROOT}"
exec "${cmd[@]}" "$@"
