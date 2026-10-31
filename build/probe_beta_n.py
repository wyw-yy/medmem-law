"""试探：让曲线形状参数 β 也随 N 变化（β = β0·(N/250)^δ），看能否同时提高 R² 与留一。

只做诊断，不改 CANDIDATES；结论若支持，再按 PROTOCOL 的修订流程加进候选。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import law as L  # noqa: E402


def _logf(x):
    return np.log(np.clip(x, 1e-12, None))


def f_L7(p, N, E):
    """M = A0·(N/250)^-a · [1 - exp(-(E / (E50_0·(N/250)^g))^(b0·(N/250)^d))]"""
    A0, a, b0, d, E50_0, g = p
    A = A0 * np.power(N / 250.0, -a)
    B = b0 * np.power(N / 250.0, d)
    E50 = E50_0 * np.power(N / 250.0, g)
    return A * (1.0 - np.exp(-np.power(E / E50, B)))


L.CANDIDATES["L7_amp_shape"] = {
    "f": f_L7,
    "pnames": ["A0_at250", "alpha", "beta0_at250", "delta_beta", "E50_at250", "gamma"],
    "p0": [0.85, 2.0, 1.5, 0.0, 700.0, 0.2],
    "lo": [1e-3, -2.0, 0.2, -1.5, 1e-2, -1.0],
    "hi": [1.0, 6.0, 8.0, 1.5, 1e5, 3.0],
    "k": 6,
}

df = L.load_metrics()
df = df[df["dataset"] == "pathmnist"].copy()
df = L.collapse_seeds(df)
print(f"{len(df)} 个种子均值点")

for name in ("L4_amp_decay", "L7_amp_shape"):
    fit = L.fit_one(name, df)
    loo = L.loo_by_n(name, df)
    params = dict(zip(L.CANDIDATES[name]["pnames"], [round(v, 4) for v in fit["params"]]))
    print(f"\n{name}")
    print(f"  R²_log {fit['r2_log']:.4f}  R²_lin {fit['r2_lin']:.4f}  "
          f"R²log_meas {fit['r2_log_meas']:.4f}  R²lin_meas {fit['r2_lin_meas']:.4f}")
    print(f"  AICc {fit['aicc']:.1f}  LOO折内 {loo['r2_mean']:.3f}  LOO池化 {loo['r2_pooled']:.3f}  "
          f"删失合规 {loo.get('censored_compliance_mean'):.2f}")
    print(f"  params {params}")