"""用另一个编码器重算记忆率曲线（B10：度量稳健性，不是模型改动）。

动机：整篇的 M 依赖一个自由选择——特征编码器。主结果用 DINOv2 ViT-S/14（dim 384，
tau≈0.67）。若结论（记忆率随 E 上升、平台随 N 下降、闭环能控住）在 ResNet50（dim 2048，
tau≈0.82）上也成立，那"记忆化"不是编码器的产物。

做法：复用已有的生成样本（`cache/samples/{run}/e*.npy`），只换特征与阈值：
  bank = 训练子集（同 data.train_subset_indices）
  ref  = 非成员参照集 2000 张
  tau  = ref 对 bank 的最大相似度的 99.5 分位（同主口径，逐 N 校准）
输出另一个文件 `results/encoder_<name>.csv`，绝不覆盖 metrics_long.csv
（后者是主结果的唯一真相来源，混入第二编码器会让审计失效）。

用法：
  python -m build.rescore_encoder --encoder resnet50 --runs pathmnist_n250_s1 pathmnist_n500_s1 pathmnist_n1000_s1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import config as C
import data as D
import metric as MT

OUT_COLS = ["run", "dataset", "n", "seed", "epoch", "encoder", "tau", "M",
            "sim_mean", "sim_p95", "sim_max", "n_gen"]


def run_meta(run: str):
    parts = run.split("_")
    dataset = parts[0]
    n = int(parts[1][1:])
    seed = int(parts[2][1:])
    return dataset, n, seed


def score_one(run: str, encoder, enc_name: str, rows: list[dict], epochs: list[int]):
    dataset, n, seed = run_meta(run)
    train, _ = D.load_split(dataset, "train")
    idx = D.train_subset_indices(dataset, n, seed)
    bank_u8 = train[idx]
    ref_u8 = D.reference_set(dataset, C.N_REF)

    f_gen_bank = MT.encode(bank_u8, encoder=encoder)
    mu, W = MT.fit_whitening(f_gen_bank)
    bank = MT.apply_whitening(f_gen_bank, mu, W)
    ref = MT.apply_whitening(MT.encode(ref_u8, encoder=encoder), mu, W)
    ref_max = MT.max_cos_sim(ref, bank)
    cal = MT.calibrate_tau(ref_max, fpr=C.FID_FPR, leak_cos=C.LEAK_COS)
    tau = cal["tau"]
    print(f"  {run}: bank={len(bank)} ref={len(ref)} tau={tau:.4f} "
          f"(去泄漏 {cal['n_leak_dropped']} 张)", flush=True)

    for ep in epochs:
        p = C.CACHE / "samples" / run / f"e{ep:04d}.npy"
        if not p.exists():
            continue
        gen_u8 = np.load(p)
        gen = MT.apply_whitening(MT.encode(gen_u8, encoder=encoder), mu, W)
        res = MT.memorization_rate(gen, bank, tau)
        rows.append(dict(run=run, dataset=dataset, n=n, seed=seed, epoch=int(ep),
                         encoder=f"{enc_name}:{encoder[1]}", tau=float(tau),
                         M=float(res["M"]), sim_mean=float(res["sim_mean"]),
                         sim_p95=float(res["sim_p95"]), sim_max=float(res["sim_max"]),
                         n_gen=int(len(gen_u8))))
        print(f"   e{ep:04d} M={res['M']:.4f} (hits={int(round(res['M']*len(gen_u8)))}/{len(gen_u8)})",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--runs", nargs="+", default=["pathmnist_n250_s1", "pathmnist_n500_s1",
                                                  "pathmnist_n1000_s1"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS / (a.out or f"encoder_{a.encoder}.csv")
    # 允许 -> 用显式允许回退，避免误以为在跑 DINOv2
    enc = MT.load_encoder(a.encoder, allow_fallback=True)
    print(f"编码器 {a.encoder} dim={enc[1]} 输出 {out}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for run in a.runs:
        ep_dir = C.CACHE / "samples" / run
        if not ep_dir.exists():
            print(f"跳过 {run}（没有样本）", flush=True)
            continue
        eps = sorted(int(p.stem[1:]) for p in ep_dir.glob("e*.npy"))
        score_one(run, enc, a.encoder, rows, eps)

    df = pd.DataFrame(rows, columns=OUT_COLS)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"写出 {out}（{len(df)} 行，{time.time() - t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())