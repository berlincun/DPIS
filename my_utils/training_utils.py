import os
import argparse
import json
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring paired pix2pix-Turbo training.
    """
    parser = argparse.ArgumentParser()

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.25, type=float)
    parser.add_argument("--lambda_lpips", default=4.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=1.0, type=float)
    parser.add_argument("--lambda_gray", type=float, default=0.1,
                        help="Weight for RGB-channel consistency loss on generated infrared images.")
    parser.add_argument("--lambda_gp", type=float, default=2.0,
                        help="Gradient penalty weight for the discriminator.")

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt_dropout_prob", type=float, default=0.15,
                        help="Probability of replacing the per-image prompt with a generic infrared prompt during training.")
    parser.add_argument(
        "--generic_prompt",
        type=str,
        default="a long-wave infrared thermal image, grayscale heat map, realistic thermal signatures",
        help="Generic prompt used when prompt dropout is triggered."
    )

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--early_stop_patience_eval", type=int, default=0,
                        help="Stop if FID does not improve for this many evaluations. 0 disables early stopping.")
    parser.add_argument("--fid_grayscale_ref", action="store_true", default=True,
                        help="Convert FID reference images to grayscale RGB, matching saved generated infrared images.")
    parser.add_argument("--num_samples_eval", type=int, default=3000, help="Number of samples to use for all evaluation")
    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=9000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for optimizer.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", help=("The scheduler type to use."))
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs.")
    parser.add_argument("--report_to", type=str, default="wandb", help=("The integration to report the results."))
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a saved accelerator state directory to resume training.")

    # physical loss parameters
    parser.add_argument("--lambda_tev_rec", type=float, default=0.008,
                        help="Weight for physical reconstruction loss L_rec")
    parser.add_argument("--lambda_tev_tev", type=float, default=0.0005,
                        help="Weight for physical TeV space loss L_tev")
    parser.add_argument("--lambda_tev_e", type=float, default=1.0,
                        help="Relative weight of e component inside L_tev")
    parser.add_argument("--lambda_tev_T", type=float, default=1.0,
                        help="Relative weight of T component inside L_tev")
    parser.add_argument("--lambda_tev_V", type=float, default=0.05,
                        help="Relative weight of V component inside L_tev. Lower value reduces V-related instability.")
    parser.add_argument("--tev_ckpt_path", type=str, default=None,
                        help="Path to pre-trained TeVNet checkpoint, e.g. checkpoints/tevnet_best.pth")
    parser.add_argument("--tev_vnums", type=int, default=4,
                        help="Number of V channels in TeVNet. Must match checkpoint.")
    parser.add_argument("--phys_start_step", type=int, default=3000,
                        help="Step at which physical losses start.")
    parser.add_argument("--phys_warmup_steps", type=int, default=2500,
                        help="Number of steps used to warm up physical loss weights.")
    parser.add_argument("--phys_max_factor", type=float, default=0.25,
                        help="Maximum multiplier for physical loss warmup.")

    # GAN warmup / discriminator schedule
    parser.add_argument("--gan_start_step", type=int, default=2000,
                        help="Step at which GAN losses start.")
    parser.add_argument("--gan_warmup_steps", type=int, default=2000,
                        help="Number of steps used to warm up GAN losses.")
    parser.add_argument("--d_update_every", type=int, default=2,
                        help="Update discriminator once every N optimizer steps after GAN warmup starts.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def center_crop_square(img):
    """Crop the centered largest square; identical geometry for paired images."""
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def build_transform(image_prep):
    """
    Constructs a deterministic transformation pipeline based on the specified image preparation method.
    For paired RGB-IR training, prefer resize_512x512 or resized_crop_512.
    Do not use independent random crop/flip unless you implement synchronized transforms.
    """
    if image_prep == "center_crop_resize_512":
        T = transforms.Compose([
            transforms.Lambda(center_crop_square),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
        ])
    elif image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        # Kept for backward compatibility with the original repo.
        # For paired RGB-IR data, this applies random transforms independently to A/B.
        # Prefer resize_512x512 unless synchronized augmentation is implemented.
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Unknown image_prep: {image_prep}")
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder,
        split,
        image_prep,
        tokenizer,
        prompt_dropout_prob=0.0,
        generic_prompt=None,
    ):
        super().__init__()

        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "val":
            self.input_folder = os.path.join(dataset_folder, "val_A")
            self.output_folder = os.path.join(dataset_folder, "val_B")
            captions = os.path.join(dataset_folder, "val_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.isdir(self.input_folder):
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        if not os.path.isdir(self.output_folder):
            raise FileNotFoundError(f"Output folder not found: {self.output_folder}")
        if not os.path.isfile(captions):
            raise FileNotFoundError(f"Prompt json not found: {captions}")

        with open(captions, "r", encoding="utf-8") as f:
            captions_dict = json.load(f)

        input_images = {
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        }
        output_images = {
            f for f in os.listdir(self.output_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        }

        valid_images = input_images & output_images & set(captions_dict.keys())
        self.captions = {k: captions_dict[k] for k in sorted(valid_images)}
        self.img_names = list(self.captions.keys())

        if len(self.img_names) == 0:
            raise RuntimeError(
                f"No paired images with prompts found for split={split}. "
                f"Check A/B filenames and prompt json keys."
            )

        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer
        self.split = split
        self.prompt_dropout_prob = prompt_dropout_prob
        self.generic_prompt = generic_prompt

        print(f" {split} 集真实加载图片数量: {len(self.img_names)}")
        missing_output = len(input_images - output_images)
        missing_input = len(output_images - input_images)
        missing_prompt = len((input_images & output_images) - set(captions_dict.keys()))
        if missing_output or missing_input or missing_prompt:
            print(
                f" [{split}] skipped unmatched files: "
                f"missing_B={missing_output}, missing_A={missing_input}, missing_prompt={missing_prompt}"
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        input_img = Image.open(os.path.join(self.input_folder, img_name)).convert("RGB")

        # Infrared GT is forced to grayscale and then expanded to 3 channels.
        # This keeps LPIPS, VAE, CLIP, and FID processing consistent with 3-channel tensors.
        output_img = Image.open(os.path.join(self.output_folder, img_name)).convert("L").convert("RGB")

        caption = self.captions[img_name]
        if self.split == "train" and self.prompt_dropout_prob > 0:
            if random.random() < self.prompt_dropout_prob:
                caption = self.generic_prompt

        # input images scaled to [0, 1]
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        # output images scaled to [-1, 1]
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }
