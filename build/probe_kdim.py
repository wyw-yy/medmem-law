"""诊断：白化截断维 k 是否在 N 之间造成度量口径不一致（这会是定律拟合偏差的元凶之一）。

方法：取每个 N 的**最后一个**采样点（M 最大、信号最强），在同一个 N 的训练 bank 上
分别用 k=125 / 256 / 384 建立白化，各自按 FPR=0.5% 校准 τ，比较 M。

如果 M 随 k 大幅变化，说明"每个 N 用 k=N/2 截断"（当前做法）引入了与 N 同向的偏差，
必须统一 k 之后重评分。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
import data as D
import metric as MT

KS = [125, 256, 384]
EPOCHS = {250: 5125, 500: 2550, 1000: 2400, 2000: 2900, 4000: 1400}


def main() -> None:
    enc = MT.load_encoder()
    print(f"编码器 {C.ENCODER} dim={enc[1]}")
    rows = []
    for N in sorted(EPOCHS):
        run = f"pathmnist_n{N}_s1"
        ep = EPOCHS[N]
        f = C.CACHE / "samples" / run / f"e{ep:04d}.npy"
        if not f.exists():
            print(f"[skip] {f} 不存在")
            continue
        gen = np.load(f)
        tr = D.load_split("pathmnist", "train", C.IMAGE_SIZE)
        idx = D.train_subset_indices("pathmnist", N, 1, C.IMAGE_SIZE)
        bank_imgs = tr[0][idx]
        gen_feat = MT.encode(gen, encoder=enc)
        bank_feat = MT.encode(bank_imgs, encoder=enc)
        te = D.load_split("pathmnist", "test", C.IMAGE_SIZE)
        ref_idx = np.random.default_rng(7).choice(len(te[0]), min(2000, len(te[0])), replace=False)
        ref_feat = MT.encode(te[0][ref_idx], encoder=enc)
        line = f"N={N:>5} ep={ep:>5} n_gen={len(gen)}"
        for k in KS:
            if k > enc[1]:
                continue
            mu, W = MT.fit_whitening(bank_feat, max_dim=k)
            B = MT.apply_whitening(bank_feat, mu, W)
            R = MT.apply_whitening(ref_feat, mu, W)
            G = MT.apply_whitening(gen_feat, mu, W)
            ref_sims = MT.max_cos_sim(R, B)
            cal = MT.calibrate_tau(ref_sims)
            res = MT.memorization_rate(G, B, cal["tau"])
            line += f" | k={k}: τ={cal['tau']:.3f} M={res['M']:.3f} sim_max={res['sim_max']:.3f}"
            rows.append(dict(n=N, epoch=ep, k=k, tau=cal["tau"], M=res["M"],
                             sim_max=res["sim_max"], sim_mean=res["sim_mean"]))
        print(line)

    if rows:
        out = C.RESULTS / "probe_kdim.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n写入 {out}")


if __name__ == "__main__":
    main()