"""验证「EMA 权重能否稳定采样质量」—— 不重训，用现存的最后 4 个 ckpt 做权重平均。

背景：pathmnist N=250 的 M(E) 曲线不是单调的，e2240=0.88 而 e2400=0.09。
肉眼检查发现低 M 的那些 epoch 采出来是「低对比度接近数据集均值」的图，
说明是**从非 EMA 的 SGD 权重采样**不稳定，而不是记忆化真的时有时无。
标准做法是采样用权重的指数滑动平均（EMA）。这里用最后 4 个 ckpt 的等权平均
（≈ EMA 的粗近似）来检验：平均权重采出来的图，M 会不会更接近高值且更稳定。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import glob
import numpy as np
import torch

import config as C
import data as D
import metric as MT
import model as Mdl

RUN = "pathmnist_n250_s1"


def load_state(epoch: int):
    p = C.CACHE / "ckpt" / RUN / f"e{epoch:04d}.pt"
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("model", sd.get("net", sd))
    return {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}


def main():
    eps = [4800, 4960, 5120, 5125]
    states = [load_state(e) for e in eps]
    avg = {k: sum(s[k] for s in states) / len(states) for k in states[0]}

    enc = MT.load_encoder()
    tr = D.train_subset("pathmnist", 250, 1)
    ref = D.reference_set("pathmnist")
    f_tr = MT.encode(tr, enc)
    mu, W = MT.fit_whitening(f_tr)
    bank = MT.apply_whitening(f_tr, mu, W)
    refb = MT.apply_whitening(MT.encode(ref, enc), mu, W)
    tau = MT.calibrate_tau(MT.max_cos_sim(refb, bank))["tau"]
    print(f"tau={tau:.4f}")

    diff = Mdl.Diffusion(device="cuda")
    for tag, sd in [("e4800", states[0]), ("e4960", states[1]), ("e5125", states[3]), ("avg4", avg)]:
        net = Mdl.UNet()
        net.load_state_dict(sd)
        net = net.to("cuda").eval()
        torch.manual_seed(0)
        g = diff.ddim_sample(net, 128, (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE), steps=50, device="cuda")
        u8 = D.to_uint8(g.float().cpu().numpy())
        f = MT.apply_whitening(MT.encode(u8, enc), mu, W)
        r = MT.memorization_rate(f, bank, tau)
        std = float(u8.reshape(len(u8), -1).std(1).mean())
        print(f"{tag:6s} M={r['M']:.3f}  sim_mean={r['sim_mean']:.3f}  "
              f"sim_p95={r['sim_p95']:.3f}  像素std={std:.1f}")


if __name__ == "__main__":
    main()