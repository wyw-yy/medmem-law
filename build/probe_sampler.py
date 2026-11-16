"""采样步数稳健性：同一条曲线的检查点，用不同 DDIM 步数重采样，看 M 会不会变。

动机：审稿人一定会问"你的 0.2% 检出率是不是采样器（DDIM 50 步）的函数"。
如果 M 对步数不敏感，这一节就是一句话；如果敏感，那主网格的"检出限"要连带说明
——两者都必须有实测，不能靠猜。

设计：
- 取**平台上**的检查点（M 已经饱和，最容易被采样质量影响出假起伏）：
  pathmnist_n250_s1 @ e5120、pathmnist_n1000_s1 @ e2400；
- 每档步数 20 / 50 / 100，各采 512 张；
- 用与原测量完全相同的编码器 + 白化截断 + τ（τ 由参照集重算，逐配置一致）；
- 输出 results/sampler_robustness.csv，列：run/epoch/steps/M/sim_max/sim_p95/tau/n_gen/seconds。

为什么用 EMA 权重：非 EMA 权重会随机塌成均值图（见 train.py 的 WeightEMA 注释），
那会把"采样器差异"和"EMA 差异"混在一起。

用法：
  python -m build.probe_sampler                        # 默认两组检查点 × 三档步数
  python -m build.probe_sampler --steps 50,100 --n 256
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C
import data as D
import metric as MT
import model as M

OUT = C.RESULTS / "sampler_robustness.csv"


def sample_with(net, n: int, steps: int, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    diff = M.Diffusion(device=C.DEVICE)
    gen = diff.ddim_sample(net, n, (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE),
                           batch=C.SAMPLE_BATCH, steps=steps, device=C.DEVICE)
    return D.to_uint8(gen.float().cpu().numpy())


def one(run: str, dataset: str, n_train: int, seed: int, epoch: int, steps: int,
        n_sample: int, ctx) -> dict:
    enc, mu, W, bank, tau = ctx
    ck_path = C.CACHE / "ckpt" / run / f"e{epoch:04d}.pt"
    if not ck_path.exists():
        return {"run": run, "epoch": epoch, "steps": steps, "skip": "no ckpt"}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    key = "ema" if "ema" in ck else "model"
    net = M.UNet().to(C.DEVICE)
    net.load_state_dict({k: v.float() for k, v in ck[key].items()})
    net.eval()
    t0 = time.time()
    gen = sample_with(net, n_sample, steps)
    secs = time.time() - t0
    f = MT.apply_whitening(MT.encode(gen, enc), mu, W)
    res = MT.memorization_rate(f, bank, tau, thresholds=C.FIXED_THRESHOLDS)
    row = {"run": run, "dataset": dataset, "n": n_train, "seed": seed, "epoch": epoch,
           "steps": steps, "weights": key, "n_gen": int(res["n_gen"]),
           "M": float(res["M"]), "sim_max": float(res["sim_max"]),
           "sim_p95": float(res["sim_p95"]), "sim_mean": float(res["sim_mean"]),
           "tau": float(tau), "seconds": round(secs, 1)}
    print(f"  {run} e{epoch} steps={steps:3d} [{key}] M={row['M']:.4f} "
          f"sim_max={row['sim_max']:.3f} ({secs:.0f}s)", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="pathmnist_n250_s1:5120,pathmnist_n1000_s1:2400",
                    help="run:epoch 逗号分隔")
    ap.add_argument("--steps", default="20,50,100")
    ap.add_argument("--n", type=int, default=C.N_SAMPLE, help="每个配置采样张数")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cases = []
    for c in a.cases.split(","):
        run, ep = c.split(":")
        cases.append((run.strip(), int(ep)))
    steps_list = [int(s) for s in a.steps.split(",")]

    rows = []
    for run, ep in cases:
        import re
        m = re.search(r"(pathmnist|bloodmnist|pneumoniamnist|dermamnist|octmnist)_n(\d+)_s(\d+)", run)
        if not m:
            raise SystemExit(f"无法解析 {run}")
        dataset, n_train, seed = m.group(1), int(m.group(2)), int(m.group(3))
        print(f"[{run}] 构建测量上下文（dataset={dataset} N={n_train} seed={seed}）", flush=True)
        train_u8 = D.train_subset(dataset, n_train, seed)
        ref_u8 = D.reference_set(dataset)
        enc = MT.load_encoder(device=C.DEVICE)
        f_train = MT.encode(train_u8, enc)
        mu, W = MT.fit_whitening(f_train)
        bank = MT.apply_whitening(f_train, mu, W)
        ref_bank = MT.apply_whitening(MT.encode(ref_u8, enc), mu, W)
        cal = MT.calibrate_tau(MT.max_cos_sim(ref_bank, bank))
        ctx = (enc, mu, W, bank, cal["tau"])
        print(f"  τ={cal['tau']:.4f} k={W.shape[1]}", flush=True)
        for st in steps_list:
            rows.append(one(run, dataset, n_train, seed, ep, st, a.n, ctx))
            pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"写出 {a.out}（{len(df)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())