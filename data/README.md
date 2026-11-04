# KAIST Dataset Preparation

The original A4 experiment uses paired visible (A)
and infrared (B) images.

Dataset structure:

KAIST_resample6000/
  train_A/  6000 images
  train_B/  6000 images
  val_A/    3000 images
  val_B/    3000 images
  test_A/   3000 images
  test_B/   3000 images

Training:
- Center crop followed by resize to 512 x 512
- Fixed training and validation splits
- Text conditioning from the provided prompt JSON files

Evaluation:
- Fixed Test3000 image pairs
- Center crop followed by resize to 512 x 512
- Paired ground-truth infrared references

The KAIST images themselves are not included.

Users must obtain the source dataset and reconstruct
the corresponding data split. The existing filenames
and prompt files alone do not recreate the original
6,000-image training subset.

## Train6000
- 6000
- set00 - set05

## Val3000
- 3000
- set00 - set05

## Test3000
- 3000
- set06 - set11

Copy `data/val_prompts.json` into the dataset root as `val_prompts.json`.
Prepare 3000 actual validation image pairs with filenames matching this JSON.
There is no separate validation preflight in `scripts/train_a4.sh`.
The dataset loader uses the intersection of A filenames, B filenames, and
prompt keys. `--num_samples_eval 3000` is an evaluation upper limit, not a
check that 3000 pairs are available. Confirm the actual loaded validation
count in the training log before treating a run as Val3000.
