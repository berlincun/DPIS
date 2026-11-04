#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${DATASET_DIR:?Please set DATASET_DIR}"
MODEL_PATH="$ROOT/checkpoints/model_best.pkl"
PROMPT_JSON="$ROOT/data/test_prompts.json"

export SD_TURBO_PATH="${SD_TURBO_PATH:-stabilityai/sd-turbo}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-$ROOT/outputs/test3000_$(date +%Y%m%d_%H%M%S)}"

echo "========== DPIS A4 TEST3000 =========="
echo "Python: $PYTHON_BIN"
echo "Dataset: $DATASET_DIR"
echo "SD-Turbo: $SD_TURBO_PATH"
echo "A4 Checkpoint: $MODEL_PATH"
echo "Output: $OUT_DIR"

command -v "$PYTHON_BIN" >/dev/null 2>&1
test -f "$MODEL_PATH"
test -f "$PROMPT_JSON"
test -d "$DATASET_DIR/test_A"
test -d "$DATASET_DIR/test_B"

mkdir -p "$OUT_DIR"

echo "========== 1. GENERATE 3000 IMAGES =========="

"$PYTHON_BIN" "$ROOT/evaluation/a4_center_crop_generate.py" \
    --model_path "$MODEL_PATH" \
    --dataset_folder "$DATASET_DIR" \
    --split test \
    --prompt_json "$PROMPT_JSON" \
    --output_dir "$OUT_DIR" \
    --resolution 512 \
    --num_images 3000 \
    --seed 42

echo "========== 2. RESULTS =========="

METRICS="$OUT_DIR/test_metrics_exact_fid.json"

if [[ ! -s "$METRICS" ]]; then
    echo "[ERROR] Metrics file missing: $METRICS"
    exit 1
fi

cat "$METRICS"

echo
echo "[SUCCESS] DPIS A4 Test3000 evaluation completed."
echo "[RESULT DIRECTORY] $OUT_DIR"
