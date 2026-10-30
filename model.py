"""
小型 DDPM UNet + DDIM 采样。64×64、base_ch=32、无 attention。

- eps-prediction，线性 beta 调度（DDPM 原始设定），与 lucidrains 基线同构但更小，
  峰值显存约 3 GB（batch 64 / 64px），8 GB 卡上留足余量。
- 时间步编码用 sinusoidal + 两层 MLP；每个 ResBlock 用 FiLM 注入。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


def _num_groups(ch: int) -> int:
    g = min(8, ch)
    while ch % g != 0:
        g //= 2
    return max(g, 1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        self.b1 = ResBlock(in_ch, out_ch, temb_dim)
        self.b2 = ResBlock(out_ch, out_ch, temb_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, temb):
        x = self.b2(self.b1(x, temb), temb)
        # 返回 (下采样后的特征, 下采样前的特征)：前者继续往下走，后者作为 skip
        return self.down(x), x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, temb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.b1 = ResBlock(in_ch + skip_ch, out_ch, temb_dim)
        self.b2 = ResBlock(out_ch, out_ch, temb_dim)

    def forward(self, x, skip, temb):
        x = self.up(x)
        # 28 这类不能被 16 整除的边长：卷积下采样 floor 之后，反过来时对不上，直接按 skip 的尺寸对齐
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.b2(self.b1(x, temb), temb)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = C.BASE_CH, ch_mults=(1, 2, 4, 8)):
        super().__init__()
        temb_dim = base * 4
        self.time = TimeEmbedding(base)
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        chs = [base * m for m in ch_mults]          # 32, 64, 128, 256
        self.downs = nn.ModuleList()
        prev = base
        for ch in chs:
            self.downs.append(Down(prev, ch, temb_dim))
            prev = ch

        self.mid1 = ResBlock(prev, prev, temb_dim)
        self.mid2 = ResBlock(prev, prev, temb_dim)

        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(Up(prev, ch, ch, temb_dim))
            prev = ch

        self.norm_out = nn.GroupNorm(_num_groups(prev), prev)
        self.conv_out = nn.Conv2d(prev, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(t)
        h = self.conv_in(x)
        skips = []
        for d in self.downs:
            h, skip = d(h, temb)
            skips.append(skip)
        h = self.mid2(self.mid1(h, temb), temb)
        for u, skip in zip(self.ups, reversed(skips)):
            h = u(h, skip, temb)
        return self.conv_out(F.silu(self.norm_out(h)))


# ------------------------------------------------------------------ 扩散过程
class Diffusion:
    """线性 beta 调度。t=0 最干净，t=T-1 最噪（与 DDPM 论文一致）。"""

    def __init__(self, timesteps: int = C.TIMESTEPS, device: str | torch.device = "cpu"):
        self.T = timesteps
        betas = torch.linspace(C.BETA_START, C.BETA_END, timesteps, dtype=torch.float64)
        self.betas = betas.float().to(device)
        self.alphas = (1.0 - betas).float().to(device)
        self.abar = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, n: int, shape, steps: int = C.DDIM_STEPS,
                    device: str | torch.device = "cpu", batch: int = 64) -> torch.Tensor:
        """确定性 DDIM（eta=0），返回 [-1,1] 的 NHWC 之外仍为 NCHW。"""
        model.eval()
        ts = torch.linspace(self.T - 1, 0, steps + 1).round().long().to(device)
        outs = []
        for start in range(0, n, batch):
            m = min(batch, n - start)
            x = torch.randn(m, *shape, device=device)
            for i in range(steps):
                t = ts[i].expand(m)
                t_next = ts[i + 1]
                eps = model(x, t)
                a_t = self.abar[t][:, None, None, None]
                a_n = self.abar[t_next]
                x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
                x0 = x0.clamp(-1, 1)
                x = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps
            outs.append(x)
        return torch.cat(outs, dim=0)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())