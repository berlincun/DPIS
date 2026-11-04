import os
import gc
import json
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset, build_transform
from tevnet import TeVloss, load_tevnet_from_checkpoint, TeVNetNormalized


def ste_clamp(x, min_val=-1.0, max_val=1.0):
    """
    Straight-through clamp.
    Forward: clamp to valid image range.
    Backward: keep identity gradient.
    """
    x_clamped = x.clamp(min_val, max_val)
    return x + (x_clamped - x).detach()




def rgb_minus1_to_gray01(x):
    """
    Convert a batch of images from [-1, 1] RGB to [0, 1] grayscale.
    Shape: [B,3,H,W] -> [B,1,H,W]
    """
    x01 = x.float().clamp(-1.0, 1.0) * 0.5 + 0.5
    gray = x01.mean(dim=1, keepdim=True)
    return gray


def compute_batch_psnr(pred_gray, target_gray, eps=1e-10):
    """
    pred_gray/target_gray: [B,1,H,W] in [0,1]
    return scalar float tensor = mean PSNR over batch
    """
    mse = F.mse_loss(pred_gray, target_gray, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))
    return psnr.mean()


def compute_batch_ssim(pred_gray, target_gray, data_range=1.0, window_size=11, c1=0.01 ** 2, c2=0.03 ** 2):
    """
    Lightweight SSIM for validation logging.
    pred_gray/target_gray: [B,1,H,W] in [0,1]
    returns mean SSIM over batch.
    """
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred_gray, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target_gray, kernel_size=window_size, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred_gray * pred_gray, kernel_size=window_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(target_gray * target_gray, kernel_size=window_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(pred_gray * target_gray, kernel_size=window_size, stride=1, padding=pad) - mu_xy

    c1 = (c1 * (data_range ** 2))
    c2 = (c2 * (data_range ** 2))

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return ssim_map.mean()

def compute_linear_warmup(global_step, start_step, warmup_steps, max_factor):
    if global_step < start_step:
        return 0.0
    warm_progress = (global_step - start_step) / max(1, warmup_steps)
    warm_progress = min(max(warm_progress, 0.0), 1.0)
    return max_factor * warm_progress


def main(args):
    # 强制使用数学 SDPA 后端，支持梯度惩罚的二阶导数
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # ---------- 初始化生成器 ----------
    base_model_path = os.path.expanduser(
        args.pretrained_model_name_or_path
        or os.environ.get("SD_TURBO_PATH", "stabilityai/sd-turbo")
    )
    if base_model_path != "stabilityai/sd-turbo" and not os.path.isdir(base_model_path):
        raise ValueError(
            "Provide stabilityai/sd-turbo or an existing local SD-Turbo directory. "
            f"Got: {base_model_path}"
        )
    # Pix2Pix_Turbo and its scheduler read this variable for every base component.
    os.environ["SD_TURBO_PATH"] = base_model_path
    net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
    net_pix2pix.set_train()

    # ---------- 加载 TeVNet（冻结 + 归一化 wrapper） ----------
    device = accelerator.device
    if args.tev_ckpt_path is not None:
        tevnet_raw = load_tevnet_from_checkpoint(
            args.tev_ckpt_path,
            device=device,
            out_channels=2 + args.tev_vnums,
        )
        tevnet = TeVNetNormalized(tevnet_raw)
        tevloss_module = TeVloss(vnums=args.tev_vnums)
        if accelerator.is_main_process:
            print(f"TeVNet loaded from {args.tev_ckpt_path} and wrapped with normalization.")
    else:
        tevnet = None
        tevloss_module = None
        if accelerator.is_main_process:
            print("No TeVNet checkpoint provided, skipping physical losses.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---------- 判别器 ----------
    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type="clip", loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()
    net_lpips.requires_grad_(False)

    # ---------- 可训练参数收集 ----------
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + \
        list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    # ---------- 生成器优化器 ----------
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Generator can have two optimizer updates per global step after GAN warmup.
    # Use a longer scheduler horizon so LR will not decay too early.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * 2 * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ---------- 判别器优化器：学习率为生成器的 1/8 ----------
    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate / 8,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ---------- 数据集 ----------
    dataset_train = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train",
        tokenizer=net_pix2pix.tokenizer,
        prompt_dropout_prob=args.prompt_dropout_prob,
        generic_prompt=args.generic_prompt,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    dataset_val = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.test_image_prep,
        split="val",
        tokenizer=net_pix2pix.tokenizer,
        prompt_dropout_prob=0.0,
        generic_prompt=args.generic_prompt,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # ---------- Accelerator prepare ----------
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)

    # ---------- CLIP 重归一化 ----------
    t_clip_renorm = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # FP16 mixed precision is managed by Accelerate autocast.
    # Keep FP32 model parameters for safe gradient scaling.
    # Do not manually convert trainable weights to FP16.

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # 关闭判别器的融合注意力
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # ---------- 断点续训 ----------
    custom_step = 0
    if args.resume_from_checkpoint is not None:
        accelerator.load_state(args.resume_from_checkpoint)
        state_file = os.path.join(args.resume_from_checkpoint, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                custom_step = json.load(f)["global_step"]
        if accelerator.is_main_process:
            print(f"Resumed training from step {custom_step}")

    # ---------- 验证集 FID 参考统计 ----------
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("legacy_tensorflow", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            if args.fid_grayscale_ref:
                x_pil = x_pil.convert("L").convert("RGB")
            else:
                x_pil = x_pil.convert("RGB")
            # Match the validation data transform before computing reference FID features.
            x_pil = build_transform(args.test_image_prep)(x_pil)
            out_pil = transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(
            os.path.join(args.dataset_folder, "val_B"),
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=torch.device("cuda"),
            mode="legacy_tensorflow",
            custom_image_tranform=fn_transform,
            description="",
            verbose=True,
        )
    else:
        feat_model = None
        ref_stats = None
        fn_transform = None

    # ========================================================
    # 训练主循环
    # ========================================================
    global_step = custom_step
    progress_bar.update(global_step)

    scale_T_ema = None
    scale_V_ema = None
    best_fid = float("inf")
    best_step = -1
    no_improve_eval_count = 0
    should_stop = False

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape

                # ---- GAN warmup factor ----
                gan_factor = compute_linear_warmup(
                    global_step=global_step,
                    start_step=args.gan_start_step,
                    warmup_steps=args.gan_warmup_steps,
                    max_factor=1.0,
                )

                # ---- 生成器前向：raw output + straight-through clamp ----
                x_tgt_pred_raw = net_pix2pix(
                    x_src,
                    prompt_tokens=batch["input_ids"],
                    deterministic=True,
                    clamp_output=False,
                )
                x_tgt_pred = ste_clamp(x_tgt_pred_raw, -1.0, 1.0)

                # ---- 基础重建损失 ----
                loss_l2 = F.mse_loss(
                    x_tgt_pred.float(),
                    x_tgt.float(),
                    reduction="mean",
                ) * args.lambda_l2

                loss_lpips = net_lpips(
                    x_tgt_pred.float(),
                    x_tgt.float(),
                ).mean() * args.lambda_lpips

                loss_gray = (
                    F.l1_loss(x_tgt_pred[:, 0:1], x_tgt_pred[:, 1:2]) +
                    F.l1_loss(x_tgt_pred[:, 1:2], x_tgt_pred[:, 2:3]) +
                    F.l1_loss(x_tgt_pred[:, 0:1], x_tgt_pred[:, 2:3])
                ) * args.lambda_gray

                loss = loss_l2 + loss_lpips + loss_gray

                # ---- 物理损失：使用 raw output，避免 hard clamp 破坏梯度 ----
                if tevnet is not None:
                    tev_warmup_factor = compute_linear_warmup(
                        global_step=global_step,
                        start_step=args.phys_start_step,
                        warmup_steps=args.phys_warmup_steps,
                        max_factor=args.phys_max_factor,
                    )

                    # 阈值熔断：当 scale_V_ema 异常偏大时，物理损失权重减半
                    if scale_V_ema is not None and scale_V_ema.mean() > 30.0:
                        tev_warmup_factor *= 0.5

                    pred_fake = tevnet(x_tgt_pred_raw.float())
                    with torch.no_grad():
                        pred_real = tevnet(x_tgt.float())

                    # 提取分量
                    e_fake = pred_fake[:, 0:1, :, :]
                    T_fake = pred_fake[:, 1:2, :, :]
                    V_fake = pred_fake[:, 2:, :, :]

                    e_real = pred_real[:, 0:1, :, :]
                    T_real = pred_real[:, 1:2, :, :]
                    V_real = pred_real[:, 2:, :, :]

                    loss_e_raw = F.mse_loss(e_fake, e_real)
                    loss_T_raw = F.mse_loss(T_fake, T_real)
                    loss_V_raw = F.mse_loss(V_fake, V_real)

                    # 动态平衡 + EMA
                    with torch.no_grad():
                        base = loss_e_raw.detach() + 1e-6
                        scale_T_raw = base / (loss_T_raw.detach() + 1e-6)
                        scale_V_raw = base / (loss_V_raw.detach() + 1e-6)

                        scale_T_raw = torch.clamp(scale_T_raw, 0.2, 1.5)
                        scale_V_raw = torch.clamp(scale_V_raw, 0.001, 25.0)

                        # CUMULATIVE ABLATION: current clipped ratios, no EMA smoothing.
                        scale_T_ema = scale_T_raw
                        scale_V_ema = scale_V_raw

                    loss_e_balanced = loss_e_raw * args.lambda_tev_e
                    loss_T_balanced = loss_T_raw * scale_T_ema * args.lambda_tev_T
                    loss_V_balanced = loss_V_raw * scale_V_ema * args.lambda_tev_V
                    loss_tev_raw = loss_e_balanced + loss_T_balanced + loss_V_balanced

                    # Use the STE-clamped valid image as reconstruction target.
                    # This avoids comparing TeV reconstruction to out-of-range raw pixels.
                    loss_rec_raw = tevloss_module.loss_rec_log(
                        pred_fake,
                        x_tgt_pred.float(),
                    )

                    loss_rec = args.lambda_tev_rec * loss_rec_raw * tev_warmup_factor
                    loss_tev = args.lambda_tev_tev * loss_tev_raw * tev_warmup_factor

                    loss += loss_rec + loss_tev

                    # 记录各分量损失和缩放因子
                    loss_rec_val = loss_rec_raw.detach().item()
                    loss_tev_val = loss_tev_raw.detach().item()
                    loss_e_raw_val = loss_e_raw.detach().item()
                    loss_T_raw_val = loss_T_raw.detach().item()
                    loss_V_raw_val = loss_V_raw.detach().item()
                    loss_rec_actual = loss_rec.detach().item()
                    loss_tev_actual = loss_tev.detach().item()
                    scale_T_val = scale_T_ema.mean().item()
                    scale_V_val = scale_V_ema.mean().item()
                else:
                    tev_warmup_factor = 0.0
                    loss_rec_val = loss_tev_val = 0.0
                    loss_e_raw_val = loss_T_raw_val = loss_V_raw_val = 0.0
                    loss_rec_actual = loss_tev_actual = 0.0
                    scale_T_val = scale_V_val = 0.0

                # ---- CLIP 语义相似度损失 ----
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm,
                        (224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                else:
                    loss_clipsim = torch.tensor(0.0, device=x_src.device)

                # 生成器优化第一步：重建 + 物理 + CLIP
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # ---- 生成器对抗损失：GAN warmup 后逐步启用 ----
                # Important: do not run a zero-loss optimizer step before GAN starts.
                # Otherwise AdamW weight decay and LR scheduler are still applied.
                if gan_factor > 0:
                    x_tgt_pred_raw_gan = net_pix2pix(
                        x_src,
                        prompt_tokens=batch["input_ids"],
                        deterministic=True,
                        clamp_output=False,
                    )
                    x_tgt_pred_gan = ste_clamp(x_tgt_pred_raw_gan, -1.0, 1.0)

                    lossG = net_disc(
                        x_tgt_pred_gan,
                        for_G=True,
                    ).mean() * args.lambda_gan * gan_factor

                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    # 清理生成器对抗步可能留在判别器上的梯度，避免跳过 D 更新时累积旧梯度
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                else:
                    x_tgt_pred_gan = x_tgt_pred.detach()
                    lossG = torch.tensor(0.0, device=x_src.device)
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                # ---- 判别器更新（延后启动 + 降低更新频率 + 可配置梯度惩罚） ----
                if gan_factor > 0 and (global_step % args.d_update_every == 0):
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan * gan_factor
                    lossD_fake = net_disc(x_tgt_pred_gan.detach(), for_real=False).mean() * args.lambda_gan * gan_factor

                    alpha = torch.rand(x_tgt.shape[0], 1, 1, 1, device=x_tgt.device)
                    x_hat = alpha * x_tgt.detach() + (1 - alpha) * x_tgt_pred_gan.detach()
                    x_hat.requires_grad = True
                    d_hat = net_disc(x_hat)
                    grad = torch.autograd.grad(
                        outputs=d_hat.sum(),
                        inputs=x_hat,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * args.lambda_gp

                    lossD = lossD_real + lossD_fake + grad_penalty
                    accelerator.backward(lossD)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                else:
                    lossD = torch.tensor(0.0, device=x_src.device)

            # ---- 日志与评估 ----
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_gray"] = loss_gray.detach().item()
                    logs["gan_factor"] = gan_factor
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    if tevnet is not None:
                        logs["loss_e_raw"] = loss_e_raw_val
                        logs["loss_T_raw"] = loss_T_raw_val
                        logs["loss_V_raw"] = loss_V_raw_val
                        logs["loss_rec_raw"] = loss_rec_val
                        logs["loss_tev_raw"] = loss_tev_val
                        logs["loss_rec_actual"] = loss_rec_actual
                        logs["loss_tev_actual"] = loss_tev_actual
                        logs["loss_e_balanced"] = loss_e_balanced.detach().item()
                        logs["loss_T_balanced"] = loss_T_balanced.detach().item()
                        logs["loss_V_balanced"] = loss_V_balanced.detach().item()
                        logs["warmup_factor"] = tev_warmup_factor
                        logs["scale_T"] = scale_T_val
                        logs["scale_V"] = scale_V_val
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        to_pil = transforms.ToPILImage()
                        log_dict = {
                            "train/source": [
                                wandb.Image(
                                    to_pil(x_src[idx].float().detach().cpu()),
                                    caption=f"idx={idx}",
                                ) for idx in range(B)
                            ],
                            "train/target": [
                                wandb.Image(
                                    to_pil(x_tgt[idx].float().detach().cpu() * 0.5 + 0.5).convert("L"),
                                    caption=f"idx={idx}",
                                ) for idx in range(B)
                            ],
                            "train/model_output": [
                                wandb.Image(
                                    to_pil(x_tgt_pred_gan[idx].float().detach().cpu() * 0.5 + 0.5).convert("L"),
                                    caption=f"idx={idx}",
                                ) for idx in range(B)
                            ],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    if global_step % args.checkpointing_steps == 1:
                        state_dir = os.path.join(args.output_dir, "training_state")
                        accelerator.save_state(state_dir)
                        with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:
                            json.dump({"global_step": global_step}, f)

                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim, l_ssim, l_psnr = [], [], [], [], []
                        fid_dir = os.path.join(args.output_dir, "eval", f"fid_{global_step}")
                        if args.track_val_fid:
                            os.makedirs(fid_dir, exist_ok=True)

                        for step_val, batch_val in enumerate(dl_val):
                            if step_val >= args.num_samples_eval:
                                break
                            x_src_val = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt_val = batch_val["output_pixel_values"].cuda()
                            B_val = x_src_val.shape[0]
                            assert B_val == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                x_tgt_pred_val = accelerator.unwrap_model(net_pix2pix)(
                                    x_src_val,
                                    prompt_tokens=batch_val["input_ids"].cuda(),
                                    deterministic=True,
                                )
                                loss_l2_val = F.mse_loss(
                                    x_tgt_pred_val.float(),
                                    x_tgt_val.float(),
                                    reduction="mean",
                                )
                                loss_lpips_val = net_lpips(x_tgt_pred_val.float(), x_tgt_val.float()).mean()
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred_val * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(
                                    x_tgt_pred_renorm,
                                    (224, 224),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred_val.device)
                                clipsim_val, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim_val = clipsim_val.mean()

                                pred_gray_val = rgb_minus1_to_gray01(x_tgt_pred_val)
                                tgt_gray_val = rgb_minus1_to_gray01(x_tgt_val)
                                ssim_val = compute_batch_ssim(pred_gray_val, tgt_gray_val)
                                psnr_val = compute_batch_psnr(pred_gray_val, tgt_gray_val)

                                l_l2.append(loss_l2_val.item())
                                l_lpips.append(loss_lpips_val.item())
                                l_clipsim.append(clipsim_val.item())
                                l_ssim.append(ssim_val.item())
                                l_psnr.append(psnr_val.item())

                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred_val[0].cpu() * 0.5 + 0.5).convert("L").convert("RGB")
                                outf = os.path.join(fid_dir, f"val_{step_val}.png")
                                output_pil.save(outf)

                        if args.track_val_fid:
                            curr_stats = get_folder_features(
                                fid_dir,
                                model=feat_model,
                                num_workers=0,
                                num=None,
                                shuffle=False,
                                seed=0,
                                batch_size=8,
                                device=torch.device("cuda"),
                                mode="legacy_tensorflow",
                                custom_image_tranform=fn_transform,
                                description="",
                                verbose=True,
                            )
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/fid"] = fid_score
                            logs["val/clean_fid"] = fid_score

                            if fid_score < best_fid:
                                best_fid = fid_score
                                best_step = global_step
                                best_path = os.path.join(args.output_dir, "checkpoints", "model_best.pkl")
                                accelerator.unwrap_model(net_pix2pix).save_model(best_path)
                                with open(os.path.join(args.output_dir, "best_fid.json"), "w", encoding="utf-8") as f:
                                    json.dump(
                                        {"best_fid": float(best_fid), "best_step": int(best_step)},
                                        f,
                                        indent=2,
                                    )
                                no_improve_eval_count = 0
                            else:
                                no_improve_eval_count += 1

                            logs["val/best_fid"] = best_fid
                            logs["val/best_step"] = best_step
                            logs["val/no_improve_eval_count"] = no_improve_eval_count

                            if args.early_stop_patience_eval > 0 and no_improve_eval_count >= args.early_stop_patience_eval:
                                should_stop = True

                        logs["val/l2"] = float(np.mean(l_l2)) if len(l_l2) else 0.0
                        logs["val/lpips"] = float(np.mean(l_lpips)) if len(l_lpips) else 0.0
                        logs["val/clipsim"] = float(np.mean(l_clipsim)) if len(l_clipsim) else 0.0
                        logs["val/ssim"] = float(np.mean(l_ssim)) if len(l_ssim) else 0.0
                        logs["val/psnr"] = float(np.mean(l_psnr)) if len(l_psnr) else 0.0
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps or should_stop:
                break
        if global_step >= args.max_train_steps or should_stop:
            break

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", "model_final.pkl")
        accelerator.unwrap_model(net_pix2pix).save_model(final_path)
        if best_step >= 0:
            print(f"Best FID: {best_fid:.6f} at step {best_step}. Saved to checkpoints/model_best.pkl")
        print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    args = parse_args_paired_training()
    args.cumulative_ablation_mode = 'pid_delay_ramp_scale'
    print('[CUMULATIVE ABLATION] pid_delay_ramp_scale', flush=True)
    main(args)
