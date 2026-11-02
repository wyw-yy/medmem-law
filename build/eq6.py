"""Bonnaire et al. (NeurIPS 2025, arXiv 2505.17638) 的记忆化判据 Eq.(6) 的独立实现。

原文定义（§2 Evaluation metrics）：
    memorized  <=>  E[ ||x_tau - a_mu1|| / ||x_tau - a_mu2|| ] < k
其中 a_mu1 / a_mu2 是 x_tau 在**训练集里像素 L2 意义下**的最近 / 次近样本，k = 1/3
（原文注：k 取 1/2 或 1/4 不改变关于 scaling 的结论）。

为什么要单独实现它：
1. 我们的主指标 M 是「白化 DINOv2 特征余弦 >= tau(FPR=0.5%)」，tau 依赖 N 与编码器，
   与文献不可直接比；Eq.(6) 是无参数的相对判据，可跨论文比。
2. 它同时能判别 "M=0 到底是没记住，还是我们的度量在该档失效"：
   - 若 Eq.(6) 也 0 → 真的没记住；
   - 若 Eq.(6) 明显非 0 而 M=0 → 是我们的 tau 在该档偏保守（度量失效，须写进限制）。

用法：
    python -m build.eq6 --runs pathmnist_n250_s1:e5125 pneumoniamnist_n64_s31:e2000 ...
    python -m build.eq6 --runs ... --k 0.5           # 敏感性
输出：results/eq6.csv（每个 (run, epoch, k) 一行）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as C
import data as D


def _load_gen(run: str, epoch: int) -> np.ndarray:
    p = C.CACHE / "samples" / run / f"e{epoch:04d}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p)


def _flat(x: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> float32 [N, D] on device, 保留像素 L2 尺度（不归一化）。"""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(C.DEVICE)
    t = t.reshape(t.shape[0], -1).float()
    return t


def _nnd2(gen: torch.Tensor, ref: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """返回每个生成样本到 ref 的最近与次近距离（平方），形状 [M, 2]。"""
    rn = (ref * ref).sum(1)
    out = []
    for i in range(0, gen.shape[0], chunk):
        g = gen[i:i + chunk]
        d2 = (g * g).sum(1, keepdim=True) + rn[None, :] - 2.0 * (g @ ref.T)
        d2 = d2.clamp_min_(0.0)
        v, _ = torch.topk(d2, k=2, largest=False)
        out.append(v)
    return torch.cat(out, 0)


def eq6_for_run(run: str, epoch: int, dataset: str, n: int, seed: int,
                k: float, n_gen_cap: int | None) -> dict:
    gen_u8 = _load_gen(run, epoch)
    if n_gen_cap is not None and gen_u8.shape[0] > n_gen_cap:
        gen_u8 = gen_u8[:n_gen_cap]
    tr_u8, _ = D.load_split(dataset, "train", size=C.IMAGE_SIZE)
    idx = D.train_subset_indices(dataset, n, seed, size=C.IMAGE_SIZE)
    tr_u8 = tr_u8[idx]

    gen = _flat(gen_u8)
    ref = _flat(tr_u8)
    nn = _nnd2(gen, ref)
    d1 = nn[:, 0].sqrt()
    d2 = nn[:, 1].clamp_min(1e-12).sqrt()
    ratio = (d1 / d2).cpu().numpy()
    mem = ratio < k
    return dict(
        run=run, dataset=dataset, n=int(n), seed=int(seed), epoch=int(epoch),
        k=float(k), n_gen=int(ratio.size), f_mem=float(mem.mean()),
        ratio_mean=float(ratio.mean()), ratio_p05=float(np.percentile(ratio, 5)),
        ratio_p50=float(np.percentile(ratio, 50)),
        n_exact=int((ratio < 0.05).sum()),
        d1_p50=float(np.percentile(d1.cpu().numpy(), 50)),
        d2_p50=float(np.percentile(d2.cpu().numpy(), 50)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run:epoch 形式，如 pathmnist_n250_s1:e5125")
    ap.add_argument("--k", type=float, default=1.0 / 3.0, help="原文 1/3")
    ap.add_argument("--n-gen", type=int, default=None, help="限制生成样本数（提速）")
    ap.add_argument("--out", default="results/eq6.csv")
    a = ap.parse_args()

    meta = {}
    for p in (C.RESULTS / "runs").glob("*.json"):
        try:
            meta[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    rows = []
    for spec in a.runs:
        run, ep = spec.rsplit(":", 1)
        ep = int(str(ep).lstrip("eE"))
        m = meta.get(run, {})
        ds = m.get("dataset") or run.split("_")[0]
        n = int(m.get("n") or run.split("_n")[1].split("_")[0])
        sd = int(m.get("seed") or run.split("_s")[1])
        r = eq6_for_run(run, ep, ds, n, sd, a.k, a.n_gen)
        rows.append(r)
        print(f"{run}@{ep}: f_mem={r['f_mem']:.4f} ratio_med={r['ratio_p50']:.3f} "
              f"exact={r['n_exact']} d1_p50={r['d1_p50']:.1f} d2_p50={r['d2_p50']:.1f}",
              flush=True)

    out = C.RESULTS / Path(a.out).name
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        keep = old[~old.set_index(["run", "epoch", "k"]).index.isin(
            new.set_index(["run", "epoch", "k"]).index)]
        new = pd.concat([keep, new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"→ {out}（{len(rows)} 行）", flush=True)


if __name__ == "__main__":
    main()