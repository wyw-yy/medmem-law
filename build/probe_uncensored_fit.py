"""实验：只用「已检出记忆化」(M>0) 的点拟合，看律在可测区间是否真的成立。

动机：229 个点里有 144 个 M=0。它们被删失惩罚（单侧 softplus）处理后，优化目标
被"预测接近 0"主导 —— 结果律在可测区间（M=0.05–0.8）系统低估 2–10 倍。
这里对比三种目标函数：
  A 全部点（当前 law.py 的做法，删失惩罚）
  B 只用 M>0 的点（纯 log 最小二乘）
  C 只用 M>0，且按 sqrt(n_gen) 加权（大样本点更可信）
输出：results/uncensored_fit.json + 屏幕表
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import config as C
import law as L


def fit_subset(name: str, df: pd.DataFrame, weights: np.ndarray | None = None) -> dict:
    spec = L.CANDIDATES[name]
    f, k = spec["f"], spec["k"]
    N = df["n"].to_numpy(float)
    E = df["epoch"].to_numpy(float)
    M = df["M"].to_numpy(float)
    w = np.ones_like(M) if weights is None else weights

    def resid(p):
        pred = np.clip(f(p, N, E), 1e-6, 1.0)
        return w * (np.log10(pred) - np.log10(np.clip(M, 1e-6, 1.0)))

    best = None
    for p0 in L._p0s(spec):
        try:
            r = least_squares(resid, np.asarray(p0, float), bounds=(spec["lo"], spec["hi"]),
                              max_nfev=4000, method="trf")
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    pred = np.clip(f(p, N, E), 1e-6, 1.0)
    ss_res = np.sum((np.log10(pred) - np.log10(np.clip(M, 1e-6, 1.0))) ** 2)
    ss_tot = np.sum((np.log10(M) - np.log10(M).mean()) ** 2)
    ss_res_lin = np.sum((pred - M) ** 2)
    ss_tot_lin = np.sum((M - M.mean()) ** 2)
    return dict(params=[float(x) for x in p],
                pnames=spec["pnames"],
                r2_log=float(1 - ss_res / max(ss_tot, 1e-12)),
                r2_lin=float(1 - ss_res_lin / max(ss_tot_lin, 1e-12)),
                cost=float(best.cost))


def main() -> None:
    df = L.load_metrics()
    pos = df[df["M"] > 0].copy()
    print(f"全部 {len(df)} 点，其中已检出 {len(pos)} 点\n")
    res = {}
    hdr = f"{'law':22s}{'R²_log':>9}{'R²_lin':>9}   逐档 bias（中位 pred/obs）"
    print(hdr)
    for name in L.CANDIDATES:
        try:
            r = fit_subset(name, pos)
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s} 失败 {type(e).__name__}: {e}")
            continue
        f = L.CANDIDATES[name]["f"]
        bias = []
        for N in sorted(pos["n"].unique()):
            sub = pos[pos["n"] == N]
            pred = np.clip(f(r["params"], sub["n"].to_numpy(float), sub["epoch"].to_numpy(float)), 1e-6, 1.0)
            bias.append(f"N{int(N)}={np.median(pred / sub['M'].to_numpy(float)):.2f}")
        r["bias_by_n"] = bias
        res[name] = r
        print(f"{name:22s}{r['r2_log']:>9.3f}{r['r2_lin']:>9.3f}   " + "  ".join(bias))
        print(f"{'':22s}params=" + ", ".join(f"{k}={v:.4g}" for k, v in zip(r["pnames"], r["params"])))
    out = C.RESULTS / "uncensored_fit.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()