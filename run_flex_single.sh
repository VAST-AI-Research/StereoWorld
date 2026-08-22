#!/bin/bash
# Single-GPU flex stereo video inference.
#
# Usage:
#   bash run_flex_single.sh                                          # defaults
#   bash run_flex_single.sh --eval_json /path/to/eval.json           # custom eval json
#   bash run_flex_single.sh --pipeline_dir /path/to/pipeline         # custom pipeline

set -e

# ── Defaults ──────────────────────────────────────────────────────────
PIPELINE_DIR="weights/StereoWorldFlexModel"
EVAL_JSON="./ExpData/flex_demo_custom_eval.json"
OUTPUT_DIR="output_flex"
H=480
W=832
NUM_FRAMES=81
FPS=16
SEED=42

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline_dir) PIPELINE_DIR="$2"; shift 2;;
        --eval_json)    EVAL_JSON="$2"; shift 2;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2;;
        --H)            H="$2"; shift 2;;
        --W)            W="$2"; shift 2;;
        --num_frames)   NUM_FRAMES="$2"; shift 2;;
        --fps)          FPS="$2"; shift 2;;
        --seed)         SEED="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p shell_logs

echo "Pipeline:   $PIPELINE_DIR"
echo "Eval JSON:  $EVAL_JSON"
echo "Output:     $OUTPUT_DIR"
echo "Resolution: ${H}x${W}, ${NUM_FRAMES} frames, ${FPS} fps"
echo ""

python inference_flex.py \
    --pipeline_dir "$PIPELINE_DIR" \
    --eval_json "$EVAL_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --H "$H" --W "$W" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --seed "$SEED" \
    2>&1 | tee shell_logs/inference_flex_single.log
