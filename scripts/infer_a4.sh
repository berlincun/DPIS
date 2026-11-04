#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_IMAGE="${INPUT_IMAGE:?Set INPUT_IMAGE}"
PROMPT="${PROMPT:?Set PROMPT}"
MODEL_PATH="${MODEL_PATH:-$ROOT/checkpoints/model_best.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/inference}"

export SD_TURBO_PATH="${SD_TURBO_PATH:-stabilityai/sd-turbo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

test -f "$INPUT_IMAGE"
test -f "$MODEL_PATH"

"$PYTHON_BIN" "$ROOT/inference_paired.py" \
  --input_image "$INPUT_IMAGE" \
  --prompt "$PROMPT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42
