"""临时探针：逐 run 的拟合优度（parity 残差），定位律在哪类 run 上系统性失败。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
import law as L  # noqa: E402


def main() -> int:
    d = pd.read_csv(C.RESULTS / "metrics_long.csv")
    fit = json.load(open(C.RESULTS / "law_fit.json", encoding="utf-8"))
    best = fit["best"]
    p = fit["fits"][best]["params"]
    print("best =", best, [round(float(x), 4) for x in p])
    print("rows =", len(d), " runs =", d["run"].nunique())

    stopped = L._stopped_runs() if hasattr(L, "_stopped_runs") else set()
    print("排除的干预组 run =", sorted(stopped))

    rows = []
    for run, g in d.groupby("run"):
        gg = g.groupby("epoch", as_index=False).agg(
            M=("M", "mean"), n=("n", "first"), seed=("seed", "first"),
            dataset=("dataset", "first"))
        pred = np.asarray(L.model_L4(p, gg["n"].to_numpy(float),
                                     gg["epoch"].to_numpy(float)), dtype=float).ravel()
        pred = np.clip(pred, 1e-5, 1.0)
        obs = gg["M"].to_numpy(float)
        mask = obs > 0
        if mask.sum() >= 3:
            lp, lo = np.log10(pred[mask]), np.log10(obs[mask])
            r2 = 1 - ((lp - lo) ** 2).sum() / ((lo - lo.mean()) ** 2).sum()
            bias = float(np.median(pred[mask] / obs[mask]))
        else:
            r2, bias = np.nan, np.nan
        rows.append(dict(
            run=run, dataset=gg["dataset"].iloc[0], n=int(gg["n"].iloc[0]),
            seed=int(gg["seed"].iloc[0]), epochs_done=int(gg["epoch"].max()),
            n_pts=len(gg), n_pos=int(mask.sum()), M_max=round(float(obs.max()), 4),
            r2_log=round(float(r2), 3) if r2 == r2 else np.nan,
            bias=round(bias, 2) if bias == bias else np.nan))

    t = pd.DataFrame(rows).sort_values(["dataset", "n", "seed"])
    print()
    print(t.to_string(index=False))
    t.to_csv(C.RESULTS / "parity_by_run.csv", index=False, encoding="utf-8")
    print("\n写出 results/parity_by_run.csv")

    # 同一 N 下 seed 之间的落差（解释"为什么单 run 拟合不好"）
    print("\n同一 N 的 seed 间 M_max 落差：")
    for (ds, n), g in t.groupby(["dataset", "n"]):
        if len(g) > 1:
            print(f"  {ds} N={n:<5d} M_max=" + " / ".join(f"{x:.3f}" for x in g["M_max"])
                  + f"   倍差={g['M_max'].max() / max(g['M_max'].min(), 1e-4):.1f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())