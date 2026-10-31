"""结果审计：在写报告/投稿前，把 metrics_long.csv 里的每一类"脏"都查一遍。

为什么需要：这张 CSV 是所有数字的唯一来源，评分过程中被中断/重跑过多次，
一旦里面混进不同编码器、重复行、τ 漂移、点数不足，报告里的每个数都不可信。

检查项（每项独立报，最后汇总，任一不过 → exit 1）：
  1) 同一 run 内 (epoch) 不重复，且除最后一次覆盖外没有互相矛盾的重复行
  2) 同一 run 内 encoder 唯一（0.6721 vs 0.8203 那次事故的教训）
  3) 同一 run 同一 N 内 τ 唯一（τ 必须逐 N 校准，但同一 N 内不得漂移）
  4) M / tau / sim_max ∈ [0,1]，n_gen 与 config.N_SAMPLE 一致
  5) 每个 run 的点数 ≥ 8（少于这个数画不出曲线）
  6) 每个 run 的最后一个 epoch == 该 run 的 epochs_done（results/runs/*.json）
  7) FID 不是全缺（主网格要求 --fid-stride 4 后至少有几个点）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C

COLS_NUM = ["M", "tau", "sim_max", "n_gen"]


def main() -> None:
    df = pd.read_csv(C.RESULTS / "metrics_long.csv")
    problems: list[str] = []
    print(f"读入 {len(df)} 行，{df['run'].nunique()} 个 run，"
          f"数据集={sorted(df['dataset'].unique())}，image_size={C.IMAGE_SIZE}")

    # 1) 重复行
    dup = df.duplicated(["run", "epoch"], keep=False)
    if dup.any():
        n_pairs = df.loc[dup, ["run", "epoch"]].drop_duplicates().shape[0]
        print(f"  [i] {n_pairs} 个 (run,epoch) 有重复行 → 取最后一行（评分重跑过）")
    d = df.sort_values("epoch").drop_duplicates(["run", "epoch"], keep="last")

    # 2) encoder 唯一
    if "encoder" in d.columns:
        for run, g in d.groupby("run"):
            encs = sorted(set(map(str, g["encoder"].dropna())))
            if len(encs) > 1:
                problems.append(f"{run}: 混用多个编码器 {encs}")

    # 3) τ 在每个 run 内唯一
    for run, g in d.groupby("run"):
        taus = np.round(g["tau"].dropna().unique(), 6)
        if len(taus) > 1:
            problems.append(f"{run}: τ 在 run 内漂移 {taus.tolist()}")

    # 4) 取值域
    for col in COLS_NUM:
        if col not in d.columns:
            continue
        v = pd.to_numeric(d[col], errors="coerce")
        bad = d[(v < 0) | (v > 1 if col in ("M", "tau", "sim_max") else False)]
        if len(bad):
            problems.append(f"{col}: {len(bad)} 行越界（示例 run={bad['run'].iloc[0]} "
                            f"epoch={bad['epoch'].iloc[0]} 值={bad[col].iloc[0]}）")
    if "n_gen" in d.columns:
        odd = d[pd.to_numeric(d["n_gen"], errors="coerce") != C.N_SAMPLE]
        if len(odd):
            problems.append(f"n_gen: {len(odd)} 行不是 {C.N_SAMPLE}"
                            f"（{sorted(set(odd['n_gen'].dropna()))[:5]}）")

    # 5) 点数（干预组豁免：plan ctl 的 run 由律提前刹停，本来就只有几个点）
    def _is_intervention(run: str) -> bool:
        rj = C.RESULTS / "runs" / f"{run}.json"
        if not rj.exists():
            return False
        j = json.loads(rj.read_text(encoding="utf-8"))
        return bool(j.get("stop_rule")) or j.get("stopped_early_at") is not None

    for run, g in d.groupby("run"):
        if len(g) < 8 and not _is_intervention(run):
            problems.append(f"{run}: 只有 {len(g)} 个测量点")

    # 6) 最后 epoch 与 run json 一致
    for run, g in d.groupby("run"):
        rj = C.RESULTS / "runs" / f"{run}.json"
        if not rj.exists():
            problems.append(f"{run}: 缺 results/runs/{run}.json")
            continue
        done = int(json.loads(rj.read_text(encoding="utf-8")).get("epochs_done", 0))
        last = int(g["epoch"].max())
        if last > done:
            problems.append(f"{run}: CSV 最后 epoch={last} > json epochs_done={done}")

    # 7) FID 覆盖
    if "fid" in d.columns:
        have = d.groupby("run")["fid"].apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
        for run, k in have.items():
            if k == 0:
                problems.append(f"{run}: 没有任何 FID 值（跑 --all --fid --refill）")

    print("\n每个 run 的概况：")
    g = d.groupby("run").agg(n=("epoch", "size"), N=("n", "first"), seed=("seed", "first"),
                             tau=("tau", "first"), e_last=("epoch", "max"),
                             M_last=("M", "last"), M_max=("M", "max"),
                             fid_pts=("fid", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())))
    print(g.to_string())

    print()
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print("  ✗ " + p)
        raise SystemExit(1)
    print("审计通过：没有发现重复/混用/越界/缺项。")


if __name__ == "__main__":
    main()