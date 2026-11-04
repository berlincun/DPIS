import torch
from torch import nn
import torch.nn.functional as F
import math
import segmentation_models_pytorch as smp

class TeVNet(nn.Module):
    """Physical decomposition network for infrared images."""
    def __init__(self, in_channels=3, out_channels=6, smp_model='Unet', smp_encoder='resnet18', smp_encoder_weights=None):
        super().__init__()
        self.tevnet = getattr(smp, smp_model)(
            encoder_name=smp_encoder,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x):
        preds = self.tevnet(x)
        preds[:, 0, :, :] = torch.sigmoid(preds[:, 0, :, :])   # e in [0,1]
        preds[:, 1, :, :] = F.relu(preds[:, 1, :, :])          # T >= 0
        return preds


class TeVloss:
    """Physical reconstruction loss and helper functions."""
    def __init__(self, vnums=4, loss_type='MSE'):
        self.vnums = vnums
        if loss_type == 'L1':
            self.loss = nn.L1Loss()
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type {loss_type}")

    def rec(self, preds, x):
        x_mean = torch.mean(x, dim=1)
        e = preds[:, 0, :, :]
        T = preds[:, 1, :, :]

        b, _, h, w = preds.shape
        V = preds[:, 2:2 + self.vnums, :, :]
        h_split_nums = int(math.sqrt(self.vnums))
        w_split_nums = self.vnums // h_split_nums
        assert h_split_nums * w_split_nums == self.vnums

        x_beta = F.avg_pool2d(x_mean, (h // h_split_nums, w // w_split_nums)).reshape(b, 1, self.vnums)
        v_pred = V.reshape(b, self.vnums, h * w)
        env = torch.matmul(x_beta, v_pred)
        env = env.view(b, 1, h, w)

        rec_img = e.unsqueeze(1) * T.unsqueeze(1) + (1 - e.unsqueeze(1)) * env
        return rec_img

    def loss_rec(self, preds, x):
        x_mean = torch.mean(x, dim=1).unsqueeze(1)
        rec_img = self.rec(preds, x)
        loss = self.loss(rec_img, x_mean)
        return loss

    # ✅ 新增：对数空间重建损失（更稳定，避免梯度消失）
    def loss_rec_log(self, preds, x):
        rec_img = self.rec(preds, x)
        x_mean = torch.mean(x, dim=1).unsqueeze(1)
        return torch.log(1 + F.l1_loss(rec_img, x_mean))


class TeVNetNormalized(nn.Module):
    """
    Wrapper for TeVNet that accepts [-1,1] input and normalizes to [0,1].
    Avoids manual clamp in training loop, preserves gradient flow.
    """
    def __init__(self, tevnet):
        super().__init__()
        self.tevnet = tevnet
        # 冻结所有参数，仅作为特征提取器
        for param in self.tevnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: expected in [-1, 1].
        Use straight-through clamp:
        - forward value is clamped to [-1, 1]
        - backward gradient is not killed outside the range
        """
        x_clamped = x.clamp(-1, 1)
        x_ste = x + (x_clamped - x).detach()
        x_01 = (x_ste + 1.0) / 2.0
        return self.tevnet(x_01)

def load_tevnet_from_checkpoint(ckpt_path, device, smp_model='Unet', smp_encoder='resnet18', out_channels=6):
    net = TeVNet(in_channels=3, out_channels=out_channels,
                 smp_model=smp_model,
                 smp_encoder=smp_encoder,
                 smp_encoder_weights=None)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # 移除 DataParallel 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    net.load_state_dict(new_state_dict, strict=True)
    net.to(device)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    return net