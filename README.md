# DPIS: Dynamic Physics-Informed Supervision

DPIS is a one-step visible-to-infrared image translation
method based on pix2pix-Turbo.

The final A4 configuration combines TeVNet-based physical
supervision, delayed activation, linear ramp-up, and
component ratio scaling. EMA is not used.

## Environment

- Python 3.10
- PyTorch 2.6.0
- torchvision 0.21.0
- CUDA 12.4 in the original environment

Install a compatible CUDA-enabled PyTorch build and
the dependencies in requirements.txt.

## Required Models

Download the SD-Turbo pretrained model separately.

Set its location:

    export SD_TURBO_PATH=/path/to/sd-turbo

The trained DPIS A4 checkpoint is model_best.pkl.
The TeVNet checkpoint is kaist-tevnet.pth.

Place both checkpoints inside checkpoints/.

Their SHA256 hashes are listed in checkpoints/SHA256SUMS.

## Dataset

Prepare paired KAIST visible and infrared images:

    train_A/   train_B/
    val_A/     val_B/
    test_A/    test_B/

Matching A/B images must share filenames.

Prompt JSON files are provided in data/.

The full dataset and training/validation split files
are not distributed in this repository.

## Training

    export DATASET_DIR=/path/to/KAIST_resample6000
    export SD_TURBO_PATH=/path/to/sd-turbo
    bash scripts/train_a4.sh

The  A4 training uses center-crop training.

## Single-Image Inference

    export INPUT_IMAGE=/path/to/visible.jpg
    export PROMPT="a thermal infrared image of a road scene"
    export MODEL_PATH=/path/to/model_best.pkl
    bash scripts/infer_a4.sh

## Test3000 Evaluation

    export DATASET_DIR=/path/to/KAIST_resample6000
    export SD_TURBO_PATH=/path/to/sd-turbo
    bash scripts/run_test3000.sh

The evaluation uses the fixed Test3000 sample mapping
and center-crop preprocessing at 512x512.

Historical A4 results (not recomputed for this corrected package):

| Metric | Result |
|--------|--------|
| FID | 88.8305 |
| LPIPS | 0.4054 |
| SSIM | 0.7239 |
| PSNR | 22.2394 dB |

The corrected evaluation applies the same grayscale conversion, center crop,
and resolution resize to the FID reference images as to the paired metric
references. Re-run Test3000 to obtain results for this corrected preprocessing.

Validation uses up to 3000 available paired samples. There is no separate
validation preflight; see data/README.md for the actual-count requirement.



## Attribution

This implementation builds on pix2pix-Turbo and
uses TeVNet-based physical supervision from PID.

Third-party licenses and notices are included.
