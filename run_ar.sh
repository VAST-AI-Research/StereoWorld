#!/bin/bash
# Single-GPU autoregressive long-video inference.
#
# Usage:
#   bash run_ar.sh --checkpoint /path/to/student.pt
#   bash run_ar.sh --checkpoint /path/to/student.pt --limit 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p shell_logs
export TOKENIZERS_PARALLELISM=false

python inference_ar.py \
    --pipeline_dir "${PIPELINE_DIR:-weights/StereoWorldModel}" \
    --checkpoint "${CHECKPOINT:-weights/student_stage3.generator.pt}" \
    --eval_json "${EVAL_JSON:-./ExpData/demo_custom_eval.json}" \
    --output_dir "${OUTPUT_DIR:-output_ar}" \
    --height "${HEIGHT:-480}" \
    --width "${WIDTH:-832}" \
    --num_frames "${NUM_FRAMES:-153}" \
    --fps "${FPS:-16}" \
    --baseline "${BASELINE:-0.2}" \
    --seed "${SEED:-1234}" \
    "$@" \
    2>&1 | tee shell_logs/inference_ar.log
