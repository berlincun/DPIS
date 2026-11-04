import os
import json
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import lpips
from cleanfid.fid import get_files_features, build_feature_extractor, fid_from_feats
from pix2pix_turbo import Pix2Pix_Turbo


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def numeric_key(name):
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else stem


def count_images(folder):
    folder = Path(folder)
    return len([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def unique_image_files(folder):
    """Return each image once, avoiding clean-fid's Windows case-insensitive glob duplication."""
    folder = Path(folder)
    files = sorted(
        (p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS),
        key=lambda p: str(p).casefold(),
    )
    unique = {}
    for path in files:
        unique.setdefault(str(path).casefold(), str(path))
    return list(unique.values())


def clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)



def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def pil_to_input_tensor(img, size):
    img = img.convert("RGB")
    img = center_crop_square(img)
    img = img.resize((size, size), Image.LANCZOS)
    return transforms.ToTensor()(img)


def prepare_ir_target_pil(img, size):
    """Share the same reference image geometry between paired metrics and FID."""
    img = img.convert("L").convert("RGB")
    img = center_crop_square(img)
    return img.resize((size, size), Image.LANCZOS)


def pil_ir_to_target_tensor(img, size):
    img = prepare_ir_target_pil(img, size)
    t = transforms.ToTensor()(img)
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(t)


def to_gray01_from_minus1_1(x):
    x01 = (x.float().clamp(-1, 1) * 0.5 + 0.5)[0]
    gray = x01.mean(dim=0)
    return gray.detach().cpu().numpy().astype(np.float32)


def compute_psnr(pred_gray, target_gray, eps=1e-10):
    mse = float(np.mean((pred_gray - target_gray) ** 2))
    if mse <= eps:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim_skimage(pred_gray, target_gray):
    try:
        from skimage.metrics import structural_similarity
    except Exception as e:
        raise ImportError("scikit-image is required for SSIM. Install it with: pip install scikit-image") from e
    return float(structural_similarity(target_gray, pred_gray, data_range=1.0))


def validate_eval_pairs(prompts, input_dir, target_dir, expected):
    """Require exactly `expected` prompted, paired test samples."""
    if expected <= 0:
        raise ValueError(f"--num_images must be positive; got {expected}")
    if not isinstance(prompts, dict):
        raise TypeError("Prompt JSON must be a mapping of image filenames to text prompts")
    if len(prompts) != expected:
        raise RuntimeError(
            f"Test sample count mismatch: expected exactly {expected} prompt entries, "
            f"found {len(prompts)}. No evaluation was started."
        )

    names = sorted(prompts.keys(), key=numeric_key)
    missing_a = [n for n in names if not (input_dir / n).is_file()]
    missing_b = [n for n in names if not (target_dir / n).is_file()]
    if missing_a or missing_b:
        raise RuntimeError(
            f"Test sample count mismatch: expected exactly {expected} image pairs; "
            f"missing inputs={len(missing_a)}, missing references={len(missing_b)}. "
            f"Examples A={missing_a[:5]}, B={missing_b[:5]}. "
            "No evaluation was started."
        )
    return names


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This eval script expects CUDA because Pix2Pix_Turbo uses .cuda() internally.")

    dataset_root = Path(args.dataset_folder)
    input_dir = dataset_root / f"{args.split}_A"
    target_dir = dataset_root / f"{args.split}_B"
    prompt_json = Path(args.prompt_json) if args.prompt_json else dataset_root / f"{args.split}_prompts.json"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target dir not found: {target_dir}")
    if not prompt_json.is_file():
        raise FileNotFoundError(f"Prompt json not found: {prompt_json}")

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "pred"
    ref_exact_dir = out_dir / "_fid_ref_exact"
    pred_exact_dir = out_dir / "_fid_pred_exact"

    with open(prompt_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # The requested evaluation size is an exact requirement, not an upper bound.
    # For Test3000 the launcher passes --num_images 3000.
    valid_names = validate_eval_pairs(prompts, input_dir, target_dir, args.num_images)

    # Reset only after data validation succeeds, avoiding accidental loss of
    # existing outputs when the input dataset is incomplete.
    clean_dir(pred_dir)
    clean_dir(ref_exact_dir)
    clean_dir(pred_exact_dir)

    print(f"[INFO] split={args.split}")
    print(f"[INFO] seed={args.seed}")
    print(f"[INFO] matched images={len(valid_names)}")
    print(f"[INFO] source A dir={input_dir}")
    print(f"[INFO] target B dir={target_dir}")
    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] pred_dir={pred_dir}")
    print(f"[INFO] ref_exact_dir={ref_exact_dir}")
    print(f"[INFO] pred_exact_dir={pred_exact_dir}")

    model = Pix2Pix_Turbo(pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    l_lpips, l_l2, l_ssim, l_psnr = [], [], [], []

    with torch.no_grad():
        for name in tqdm(valid_names, desc="Generating images"):
            src_pil = Image.open(input_dir / name)
            tgt_pil = Image.open(target_dir / name)

            x_src = pil_to_input_tensor(src_pil, args.resolution).unsqueeze(0).cuda()
            x_tgt = pil_ir_to_target_tensor(tgt_pil, args.resolution).unsqueeze(0).cuda()

            x_src_model = x_src.half() if args.use_fp16 else x_src
            pred = model(x_src_model, prompt=prompts[name], deterministic=True)
            pred = pred.float()
            x_tgt = x_tgt.float()

            l2 = F.mse_loss(pred, x_tgt, reduction="mean").item()
            lp = net_lpips(pred, x_tgt).mean().item()

            pred_gray = to_gray01_from_minus1_1(pred)
            tgt_gray = to_gray01_from_minus1_1(x_tgt)
            ssim = compute_ssim_skimage(pred_gray, tgt_gray)
            psnr = compute_psnr(pred_gray, tgt_gray)

            l_l2.append(l2)
            l_lpips.append(lp)
            l_ssim.append(ssim)
            l_psnr.append(psnr)

            out_pil = transforms.ToPILImage()((pred[0].cpu().clamp(-1, 1) * 0.5 + 0.5))
            out_pil = out_pil.convert("L").convert("RGB")

            # Save to visible output and exact FID folder.
            out_pil.save(pred_dir / name)
            out_pil.save(pred_exact_dir / name)

            # Use the same grayscale, center crop, and resize as paired metrics.
            ref_pil = prepare_ir_target_pil(tgt_pil, args.resolution)
            ref_pil.save(ref_exact_dir / name)

    print("[GENERATION_COMPLETED]", len(valid_names), flush=True)

    n_pred = count_images(pred_dir)
    n_ref_exact = count_images(ref_exact_dir)
    n_pred_exact = count_images(pred_exact_dir)
    print(f"[CHECK] pred_dir image count={n_pred}")
    print(f"[CHECK] ref_exact_dir image count={n_ref_exact}")
    print(f"[CHECK] pred_exact_dir image count={n_pred_exact}")

    expected = len(valid_names)
    if n_pred != expected or n_ref_exact != expected or n_pred_exact != expected:
        raise RuntimeError(
            f"Image count mismatch before FID: expected={expected}, "
            f"pred={n_pred}, ref_exact={n_ref_exact}, pred_exact={n_pred_exact}"
        )

    feat_model = build_feature_extractor("legacy_tensorflow", "cuda", use_dataparallel=False)

    def fid_transform(x):
        x_pil = Image.fromarray(x).convert("L").convert("RGB")
        x_pil = x_pil.resize((299, 299), Image.LANCZOS)
        return np.array(x_pil)

    ref_files = unique_image_files(ref_exact_dir)
    pred_files = unique_image_files(pred_exact_dir)
    if len(ref_files) != expected or len(pred_files) != expected:
        raise RuntimeError(
            f"Unique FID file count mismatch: expected={expected}, "
            f"ref={len(ref_files)}, pred={len(pred_files)}"
        )
    print(f"[CHECK] unique reference FID files={len(ref_files)}")
    print(f"[CHECK] unique generated FID files={len(pred_files)}")

    print("[INFO] Computing exact reference FID features...")
    ref_stats = get_files_features(
        ref_files,
        model=feat_model,
        num_workers=args.num_workers_fid,
        batch_size=args.fid_batch_size,
        device=torch.device("cuda"),
        mode="legacy_tensorflow",
        custom_image_tranform=fid_transform,
        description="",
        fdir=None,
        verbose=True,
    )

    print("[INFO] Computing exact generated FID features...")
    pred_stats = get_files_features(
        pred_files,
        model=feat_model,
        num_workers=args.num_workers_fid,
        batch_size=args.fid_batch_size,
        device=torch.device("cuda"),
        mode="legacy_tensorflow",
        custom_image_tranform=fid_transform,
        description="",
        fdir=None,
        verbose=True,
    )

    if ref_stats.shape[0] != expected or pred_stats.shape[0] != expected:
        raise RuntimeError(
            f"FID feature count mismatch: expected={expected}, "
            f"ref={ref_stats.shape[0]}, pred={pred_stats.shape[0]}"
        )

    fid = float(fid_from_feats(ref_stats, pred_stats))

    metrics = {
        "split": args.split,
        "seed": args.seed,
        "num_images": expected,
        "model_path": args.model_path,
        "fid": fid,
        "l2": float(np.mean(l_l2)),
        "lpips": float(np.mean(l_lpips)),
        "ssim": float(np.mean(l_ssim)),
        "psnr": float(np.mean(l_psnr)),
        "pred_count": n_pred,
        "fid_ref_exact_count": n_ref_exact,
        "fid_pred_exact_count": n_pred_exact,
        "fid_ref_feature_count": int(ref_stats.shape[0]),
        "fid_pred_feature_count": int(pred_stats.shape[0]),
        "fid_file_enumeration": "explicit_unique_paths_windows_fix",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_metrics_exact_fid.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--prompt_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_images", type=int, default=3000)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--fid_batch_size", type=int, default=8)
    parser.add_argument("--num_workers_fid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
