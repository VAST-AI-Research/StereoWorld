#!/usr/bin/env bash
# Single-GPU fixed-left view-inpainting inference.
#
# Usage:
#   bash run_view_inpaint_single.sh
#   bash run_view_inpaint_single.sh --pipeline_dir /path/to/StereoWorldInpaintModel
#   bash run_view_inpaint_single.sh --eval_json /path/to/eval.json
#   bash run_view_inpaint_single.sh --dry-run

set -euo pipefail

PIPELINE_DIR="weights/StereoWorldInpaintModel"
EVAL_JSON="ExpData/view_inpaint_eval.json"
OUTPUT_DIR="output_view_inpaint"
PYTHON_BIN="python"
GPU_ID=0
HEIGHT=480
WIDTH=832
FPS=16
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=3.0
FLOW_SHIFT=3.0
BOUNDARY=0.875
SEED=42
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash run_view_inpaint_single.sh [options]

Run fixed-left view inpainting for all jobs in one eval JSON on one GPU.

Options:
  --pipeline_dir PATH          Inpainting pipeline directory
                               (default: weights/StereoWorldInpaintModel)
  --eval_json PATH             Batch eval JSON
                               (default: ExpData/view_inpaint_eval.json)
  --output_dir PATH            Generated-video root
                               (default: output_view_inpaint)
  --gpu_id ID                  Physical CUDA GPU to use (default: 0)
  --python PATH                Python executable (default: python)
  --H INT --W INT              Output resolution (default: 480x832)
  --fps INT                    Output FPS (default: 16)
  --num_inference_steps INT    Sampling steps (default: 50)
  --steps INT                  Alias for --num_inference_steps
  --guidance_scale FLOAT       CFG scale (default: 3.0)
  --flow_shift FLOAT           Flow scheduler shift (default: 3.0)
  --boundary FLOAT             Transformer boundary (default: 0.875)
  --seed INT                   Sampling seed (default: 42)
  --dry-run                    Print the command without checking files or writing logs
  -h, --help                   Show this help
EOF
}

require_value() {
    if [[ $# -lt 2 || -z ${2:-} ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline_dir) require_value "$@"; PIPELINE_DIR="$2"; shift 2 ;;
        --eval_json) require_value "$@"; EVAL_JSON="$2"; shift 2 ;;
        --output_dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
        --gpu_id) require_value "$@"; GPU_ID="$2"; shift 2 ;;
        --python) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --H) require_value "$@"; HEIGHT="$2"; shift 2 ;;
        --W) require_value "$@"; WIDTH="$2"; shift 2 ;;
        --fps) require_value "$@"; FPS="$2"; shift 2 ;;
        --num_inference_steps|--steps)
            require_value "$@"; NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --guidance_scale) require_value "$@"; GUIDANCE_SCALE="$2"; shift 2 ;;
        --flow_shift) require_value "$@"; FLOW_SHIFT="$2"; shift 2 ;;
        --boundary) require_value "$@"; BOUNDARY="$2"; shift 2 ;;
        --seed) require_value "$@"; SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

COMMAND=(
    "$PYTHON_BIN" inference_view_inpainting.py
    --pipeline_dir "$PIPELINE_DIR"
    --eval_json "$EVAL_JSON"
    --output_dir "$OUTPUT_DIR"
    --H "$HEIGHT" --W "$WIDTH"
    --fps "$FPS"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --guidance_scale "$GUIDANCE_SCALE"
    --flow_shift "$FLOW_SHIFT"
    --boundary "$BOUNDARY"
    --seed "$SEED"
)

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_ID"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
}

if [[ $DRY_RUN -eq 1 ]]; then
    print_command
    exit 0
fi

LOG_DIR="shell_logs"
LOG_PATH="$LOG_DIR/inference_view_inpaint_single.log"
mkdir -p -- "$LOG_DIR"

echo "Pipeline:   $PIPELINE_DIR"
echo "Eval JSON:  $EVAL_JSON"
echo "Output:     $OUTPUT_DIR"
echo "GPU:        $GPU_ID"
echo "Resolution: ${HEIGHT}x${WIDTH}, ${FPS} fps"
echo "Sampling:   ${NUM_INFERENCE_STEPS} steps, cfg=${GUIDANCE_SCALE}, shift=${FLOW_SHIFT}, boundary=${BOUNDARY}, seed=${SEED}"
echo "Log:        $LOG_PATH"
echo

CUDA_VISIBLE_DEVICES="$GPU_ID" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    "${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
