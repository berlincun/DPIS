#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${DATASET_DIR:?Please set DATASET_DIR}"
export SD_TURBO_PATH="${SD_TURBO_PATH:-stabilityai/sd-turbo}"

TEV_CKPT="${TEV_CKPT:-$ROOT/checkpoints/kaist-tevnet.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/a4_training}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes 1 \
  --mixed_precision fp16 \
  "$ROOT/train_dpis.py" \
  --dataset_folder "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$SD_TURBO_PATH" \
  --tev_ckpt_path "$TEV_CKPT" \
  --train_image_prep center_crop_resize_512 \
  --test_image_prep center_crop_resize_512 \
  --resolution 512 \
  --train_batch_size 2 \
  --max_train_steps 9000 \
  --seed 42 \
  --learning_rate 5e-6 \
  --lora_rank_unet 8 \
  --lora_rank_vae 4 \
  --lambda_tev_rec 0.008 \
  --lambda_tev_tev 0.0005 \
  --phys_start_step 3000 \
  --phys_warmup_steps 2500 \
  --phys_max_factor 0.25 \
  --eval_freq 500 \
  --track_val_fid \
  --num_samples_eval 3000 \
  --checkpointing_steps 50 \
  --early_stop_patience_eval 3
