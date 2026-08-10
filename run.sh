#!/bin/bash
# Stereo video inference script.
#
# Usage:
#   bash run.sh                                          # defaults
#   bash run.sh --eval_json /path/to/eval.json           # custom eval json
#   bash run.sh --num_gpus 8 --H 704 --W 1280           # 8 GPUs, higher res

set -e

# ── Defaults ───────────────────────────────────────────────────────────
PIPELINE_DIR="weights/StereoWorldModel"
EVAL_JSON="./ExpData/demo_custom_eval.json"
OUTPUT_DIR="output"
NUM_GPUS=4
H=480
W=832
NUM_FRAMES=81
FPS=16
BASELINE=0.2
SEED=42
USE_RAYMAP=""
MASTER_PORT=${MASTER_PORT:-29500}

# ── Parse arguments ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline_dir) PIPELINE_DIR="$2"; shift 2;;
        --eval_json)    EVAL_JSON="$2"; shift 2;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2;;
        --num_gpus)     NUM_GPUS="$2"; shift 2;;
        --H)            H="$2"; shift 2;;
        --W)            W="$2"; shift 2;;
        --num_frames)   NUM_FRAMES="$2"; shift 2;;
        --fps)          FPS="$2"; shift 2;;
        --baseline)     BASELINE="$2"; shift 2;;
        --seed)         SEED="$2"; shift 2;;
        --use_raymap)   USE_RAYMAP="--use_raymap"; shift 1;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p shell_logs

echo "Pipeline:   $PIPELINE_DIR"
echo "Eval JSON:  $EVAL_JSON"
echo "Output:     $OUTPUT_DIR"
echo "GPUs:       $NUM_GPUS"
echo "Resolution: ${H}x${W}, ${NUM_FRAMES} frames, ${FPS} fps"
echo "Baseline:   $BASELINE"
echo ""

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT inference.py \
    --pipeline_dir "$PIPELINE_DIR" \
    --eval_json "$EVAL_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --baseline "$BASELINE" \
    $USE_RAYMAP \
    --ulysses_degree "$NUM_GPUS" --ring_degree 1 \
    --H "$H" --W "$W" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --seed "$SEED" \
    2>&1 | tee shell_logs/inference.log
