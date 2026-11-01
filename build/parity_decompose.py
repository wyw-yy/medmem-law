"""诊断：逐 run 的拟合误差里，有多少是"时间错位"（onset 早晚），有多少是"高度错"（幅度）。

做法：对每个 run，只允许一个自由度 δ（把律的时间轴拉伸 E→E·δ），在 δ∈[1/8, 8] 上网格搜索，
使 log 空间的平方误差最小。比较三种残差：
  原样（δ=1） / 只调 δ / 只调幅度 A（乘性，1 个自由度）
如果"只调 δ"就能把误差压掉大半，说明律错在**什么时候发作**，不是**发作多少**——
这正是种子方差（同一 N 下 onset 差 8–60 倍）的定量表达。
"""

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

FLOOR = 0.02      # 可测下限：M ≥ 2% 才算"看见了记忆化"（≈10/512）


def se(lp, lo):
    return float(((lp - lo) ** 2).sum())


def main() -> int:
    d = pd.read_csv(C.RESULTS / "metrics_long.csv")
    fit = json.load(open(C.RESULTS / "law_fit.json", encoding="utf-8"))
    best = fit["best"]
    p = np.array(fit["fits"][best]["params"], dtype=float)
    f = L.CANDIDATES[best]["f"]
    stopped = L._stopped_runs()

    rows = []
    for run, g in d.groupby("run"):
        if run in stopped:
            continue
        gg = g.groupby("epoch", as_index=False).agg(M=("M", "mean"), n=("n", "first"),
                                                   seed=("seed", "first"),
                                                   dataset=("dataset", "first"))
        obs = gg["M"].to_numpy(float)
        n_arr = gg["n"].to_numpy(float)
        e_arr = gg["epoch"].to_numpy(float)
        mask = obs >= FLOOR
        if mask.sum() < 3:
            continue
        lo = np.log10(obs[mask])

        def pred(scale_e=1.0, scale_m=1.0):
            v = np.asarray(f(p, n_arr, e_arr * scale_e), dtype=float).ravel()
            return np.log10(np.clip(v * scale_m, 1e-5, 1.0))[mask]

        base = se(pred(), lo)
        # 只调时间轴
        grid = np.exp(np.linspace(np.log(1 / 8), np.log(8), 241))
        e_d = min(((se(pred(s), lo), s) for s in grid), key=lambda t: t[0])
        # 只调幅度
        mgrid = np.exp(np.linspace(np.log(0.05), np.log(20), 241))
        e_m = min(((se(pred(1.0, m), lo), m) for m in mgrid), key=lambda t: t[0])

        rows.append(dict(run=run, dataset=gg["dataset"].iloc[0], n=int(gg["n"].iloc[0]),
                         seed=int(gg["seed"].iloc[0]), n_pts=int(mask.sum()),
                         M_max=round(float(obs.max()), 4),
                         sse0=round(base, 2),
                         sse_shift=round(e_d[0], 2), delta_shift=round(e_d[1], 3),
                         sse_amp=round(e_m[0], 2), amp=round(e_m[1], 3),
                         gain_shift=round(1 - e_d[0] / base, 3),
                         gain_amp=round(1 - e_m[0] / base, 3)))

    t = pd.DataFrame(rows).sort_values("sse0", ascending=False)
    print(t.to_string(index=False))
    t.to_csv(C.RESULTS / "parity_decompose.csv", index=False, encoding="utf-8")

    print(f"\n可比 run：{len(t)} 个（只用 M≥{FLOOR} 的点）")
    print(f"原样 SSE 总和 = {t['sse0'].sum():.1f}")
    print(f"只调时间轴后 = {t['sse_shift'].sum():.1f}"
          f"（消掉 {1 - t['sse_shift'].sum() / t['sse0'].sum():.1%}）")
    print(f"只调幅度后   = {t['sse_amp'].sum():.1f}"
          f"（消掉 {1 - t['sse_amp'].sum() / t['sse0'].sum():.1%}）")
    print("\n时间轴偏移 δ 中位数 =", round(float(t['delta_shift'].median()), 3),
          " 幅度倍数中位数 =", round(float(t['amp'].median()), 3))

    main_ds = t[t["dataset"] == "pathmnist"]
    print(f"\n只看 pathmnist（律的拟合域）：{len(main_ds)} 个 run，"
          f"原样 {main_ds['sse0'].sum():.1f} → 调时间轴 {main_ds['sse_shift'].sum():.1f}"
          f"（消掉 {1 - main_ds['sse_shift'].sum() / main_ds['sse0'].sum():.1%}），"
          f"→ 调幅度 {main_ds['sse_amp'].sum():.1f}"
          f"（消掉 {1 - main_ds['sse_amp'].sum() / main_ds['sse0'].sum():.1%}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())