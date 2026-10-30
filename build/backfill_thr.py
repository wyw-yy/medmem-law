"""给已有的 metrics_long.csv 行补 M_050 / M_060 / M_070 / sim_p50（固定阈值稳健性列）。

为什么需要：主结论用 τ（按 FPR=0.5% 逐 N 校准）算 M。审稿人一定会问
"换个阈值结论还成立吗"。所以每个测量点还要报固定阈值 0.5/0.6/0.7 下的记忆率。
这些列是后加的，老行没有；本脚本只补缺失的那些行，不重算已有值。

用法：
    python -m build.backfill_thr            # 只补缺失
    python -m build.backfill_thr --dry-run  # 只看要补多少行
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C          # noqa: E402
import data as D            # noqa: E402
import metric as MT         # noqa: E402
from tools import score_run as S   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=C.DEVICE)
    a = ap.parse_args()

    if not S.CSV.exists():
        raise SystemExit(f"没有 {S.CSV}")
    df = S.load_csv()
    for c in ("M_050", "M_060", "M_070", "sim_p50"):
        if c not in df.columns:
            df[c] = np.nan

    todo = df[df["M_050"].isna()]
    print(f"共 {len(df)} 行，需补 {len(todo)} 行（{todo['run'].nunique()} 个 run）")
    if a.dry_run or not len(todo):
        print(todo.groupby("run")["epoch"].agg(["size", "min", "max"]).to_string())
        return 0

    enc = MT.load_encoder(device=a.device)
    enc_tag = f"{C.ENCODER}:{enc[1]}"

    for run, g in todo.groupby("run"):
        dataset, n, seed = S.parse_tag(run)
        sdir = C.CACHE / "samples" / run
        files = {int(p.stem[1:]): p for p in sdir.glob("e*.npy")}
        if not files:
            print(f"[{run}] 没有样本文件，跳过")
            continue

        train_u8 = D.train_subset(dataset, n, seed)
        f_train = MT.encode(train_u8, enc)
        mu, W = MT.fit_whitening(f_train)
        bank = MT.apply_whitening(f_train, mu, W)
        old_enc = set(g["encoder"].dropna().astype(str).unique())
        if old_enc and old_enc != {enc_tag}:
            print(f"[{run}] 编码器不一致 {old_enc} vs {enc_tag}，跳过")
            continue

        idx = 0
        for ep in sorted(g["epoch"].astype(int)):
            if ep not in files:
                continue
            gen = np.load(files[ep])
            f_gen = MT.apply_whitening(MT.encode(gen, enc), mu, W)
            res = MT.memorization_rate(f_gen, bank, float(g.loc[g["epoch"] == ep, "tau"].iloc[0]),
                                       thresholds=C.FIXED_THRESHOLDS)
            m = (df["run"] == run) & (df["epoch"].astype(int) == ep)
            for k in ("sim_p50", "M_050", "M_060", "M_070"):
                df.loc[m, k] = res[k]
            idx += 1
            if idx % 8 == 0:
                df.to_csv(S.CSV, index=False)
                print(f"  [{run}] 已补 {idx} 点", flush=True)
        df.to_csv(S.CSV, index=False)
        print(f"[{run}] 补完 {idx} 点")

    df.to_csv(S.CSV, index=False)
    print(f"写出 {S.CSV}（{len(df)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())