"""跨数据集概览：每个 (dataset, N, seed) 的起跳点、末点、平台与 FID。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C


def first_cross(g: pd.DataFrame, eps: float) -> float:
    """第一次 M ≥ eps 的 epoch；没到就 nan。要求连续两个点都 ≥ eps 以免量化毛刺。"""
    g = g.sort_values("epoch")
    m = g["M"].to_numpy(float)
    e = g["epoch"].to_numpy(int)
    for i in range(len(m) - 1):
        if m[i] >= eps and m[i + 1] >= eps:
            return float(e[i])
    return float("nan")


def main() -> None:
    d = pd.read_csv(C.RESULTS / "metrics_long.csv")
    rows = []
    for (ds, n, s), g in d.groupby(["dataset", "n", "seed"]):
        g = g.sort_values("epoch")
        last = g.iloc[-1]
        rows.append(dict(
            dataset=ds, n=int(n), seed=int(s), points=len(g),
            tau=round(float(g["tau"].iloc[0]), 4),
            epochs=int(last["epoch"]),
            steps=int(last["epoch"] * n / C.BATCH),
            M_last=round(float(last["M"]), 4),
            M_max=round(float(g["M"].max()), 4),
            onset_1pct=first_cross(g, 0.01),
            onset_5pct=first_cross(g, 0.05),
            steps_1pct=(first_cross(g, 0.01) * n / C.BATCH
                        if np.isfinite(first_cross(g, 0.01)) else np.nan),
            fid_last=(round(float(last["fid"]), 1) if pd.notna(last.get("fid")) else np.nan),
        ))
    t = pd.DataFrame(rows).sort_values(["dataset", "n", "seed"])
    print(t.to_string(index=False))
    out = C.RESULTS / "grid_summary.csv"
    t.to_csv(out, index=False, encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()