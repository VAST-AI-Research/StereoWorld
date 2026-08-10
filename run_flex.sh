#!/bin/bash
# Multi-GPU flex stereo video inference (sequence parallel via torchrun).
#
# Usage:
#   bash run_flex.sh                                             # 4 GPUs, defaults
#   bash run_flex.sh --num_gpus 8                                # 8 GPUs
#   bash run_flex.sh --eval_json /path/to/eval.json              # custom eval json
#   bash run_flex.sh --pipeline_dir /path/to/pipeline            # custom pipeline

set -e

# ── Defaults ──────────────────────────────────────────────────────────
PIPELINE_DIR="weights/StereoWorldFlexModel"

EVAL_JSON="./ExpData/flex_demo_custom_eval.json"
OUTPUT_DIR="output_flex"
NUM_GPUS=4
H=480
W=832
NUM_FRAMES=81
FPS=16
SEED=42
MASTER_PORT=${MASTER_PORT:-29501}

# ── Parse arguments ───────────────────────────────────────────────────
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline_dir)   PIPELINE_DIR="$2"; shift 2;;
        --eval_json)      EVAL_JSON="$2"; shift 2;;
        --output_dir)     OUTPUT_DIR="$2"; shift 2;;
        --num_gpus)       NUM_GPUS="$2"; shift 2;;
        --H)              H="$2"; shift 2;;
        --W)              W="$2"; shift 2;;
        --num_frames)     NUM_FRAMES="$2"; shift 2;;
        --fps)            FPS="$2"; shift 2;;
        --seed)           SEED="$2"; shift 2;;
        *) EXTRA_ARGS+=("$1"); shift 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p shell_logs

echo "Pipeline:    $PIPELINE_DIR"
echo "Eval JSON:   $EVAL_JSON"
echo "Output:      $OUTPUT_DIR"
echo "GPUs:        $NUM_GPUS"
echo "Resolution:  ${H}x${W}, ${NUM_FRAMES} frames, ${FPS} fps"
echo ""

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT inference_flex.py \
    --pipeline_dir "$PIPELINE_DIR" \
    --eval_json "$EVAL_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --ulysses_degree "$NUM_GPUS" --ring_degree 1 \
    --H "$H" --W "$W" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee shell_logs/inference_flex.log
