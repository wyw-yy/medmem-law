"""排查：为什么某些 checkpoint 采出来是"空白图"（M 掉到 0.09）。

对比两个 ckpt：e4800（M=0.656，疑似退化）与 e4960（M=0.947，正常）。
分别看不同 DDIM 步数、不同 seed 下，生成的图像是否塌成数据集均值。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import config as C
import data as D
import model as Mdl

RUN = "pathmnist_n250_s1"


def load(epoch: int, device="cuda"):
    p = C.CACHE / "ckpt" / RUN / f"e{epoch:04d}.pt"
    net = Mdl.UNet()
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("model", sd.get("net", sd))
    net.load_state_dict({k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()})
    return net.to(device).eval()


def probe(net, steps, n=16, seed=0, device="cuda"):
    diff = Mdl.Diffusion(device=device)
    torch.manual_seed(seed)
    g = diff.ddim_sample(net, n, (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE), steps=steps, device=device)
    u8 = D.to_uint8(g.float().cpu().numpy())
    sd = u8.reshape(len(u8), -1).std(1)
    return dict(flat=int((sd < 20).sum()), n=n, std_min=float(sd.min()), std_med=float(np.median(sd)))


def main():
    tr = D.train_subset("pathmnist", 250, 1)
    tr_std = float(tr.reshape(len(tr), -1).std())
    print(f"真实训练图的像素 std（参考）：{tr_std:.1f}")
    for ep in (4800, 4960):
        net = load(ep)
        for steps in (50, 100, 200):
            for seed in (0, 1):
                r = probe(net, steps, seed=seed)
                print(f"e{ep} steps={steps:3d} seed={seed}  flat={r['flat']}/{r['n']}  "
                      f"std_min={r['std_min']:.1f} std_med={r['std_med']:.1f}")


if __name__ == "__main__":
    main()