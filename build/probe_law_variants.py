"""探路：主网格跑完后，先看清"哪个律形式真的能外推"，再定最终模型。

背景（重要）：等步数预算下 N=2000/4000 全是 M=0（低于检出限），N=1000 只升到 0.21
就没了预算。所以能提供"曲线形状"的只有 N=250/500/1000 三档，而这三档的**起始 epoch
几乎一样（≈600）**、**平台高度差很多（0.93 / 0.80 / ≥0.21）**。
这暗示机理上：起背时刻在 epoch 坐标里大致与 N 无关（γ≈0），差别在幅度。

于是这里并排试 6 个形式，都用 law.py 的删失残差，看 R²_log 和留一 N 的 R²：
  V1 L1（基线，共享 Minf，γ 自由）
  V2 L1 + 幅度按幂律衰减   Minf·(N/N0)^-δ
  V3 L1 + γ 钉死为 0 + 幅度衰减
  V4 L2 Hill + 幅度衰减
  V5 饱和幅度           Minf/(1+(N/N0)^δ)
  V6 起点与幅度都可变 + 起点用 step 坐标（S=E·N/B，理论上 onset_steps ∝ N^α）

用法：python build\probe_law_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import config as C
import law as L

N0 = 1000.0
df = L.load_metrics()
N = df["n"].to_numpy(float)
E = df["epoch"].to_numpy(float)
M = df["M"].to_numpy(float)
mdet = L._mdet(df)
obs = M > 0
print(f"{len(df)} 点，其中有信号 {int(obs.sum())} 点；N={sorted(set(N.astype(int)))}")


def f_v1(p, N, E):
    Minf, beta, c, gamma = p
    return Minf * (1 - np.exp(-np.power(E / (c * np.power(N, gamma)), beta)))


def f_v2(p, N, E):
    Minf, beta, c, gamma, delta = p
    amp = Minf * np.power(N / N0, -delta)
    return amp * (1 - np.exp(-np.power(E / (c * np.power(N, gamma)), beta)))


def f_v3(p, N, E):
    Minf, beta, c, delta = p
    amp = Minf * np.power(N / N0, -delta)
    return amp * (1 - np.exp(-np.power(E / c, beta)))


def f_v4(p, N, E):
    Minf, h, c, gamma, delta = p
    amp = Minf * np.power(N / N0, -delta)
    r = np.power(np.maximum(E, 1e-9), h)
    return amp * r / (r + np.power(c * np.power(N, gamma), h))


def f_v5(p, N, E):
    Minf, beta, c, gamma, delta = p
    amp = Minf / (1.0 + np.power(N / N0, delta))
    return amp * (1 - np.exp(-np.power(E / (c * np.power(N, gamma)), beta)))


def f_v6(p, N, E):
    # 起点用"步数"坐标写成 epoch：S = E*N/B，S_onset = a*N^alpha → E_onset = a*B*N^(alpha-1)
    Minf, beta, a, alpha, delta = p
    alpha = max(alpha, 0.0)
    E_onset = a * C.BATCH * np.power(N, alpha - 1.0)
    amp = Minf * np.power(N / N0, -delta)
    return amp * (1 - np.exp(-np.power(E / np.maximum(E_onset, 1e-6), beta)))


VARIANTS = {
    "V1_L1": dict(f=f_v1, p0=[0.9, 1.6, 200.0, 0.2], lo=[1e-3, 0.2, 1e-2, -1.0], hi=[1.0, 8.0, 1e5, 3.0]),
    "V2_L1_amp": dict(f=f_v2, p0=[0.9, 1.6, 200.0, 0.2, 0.2], lo=[1e-3, 0.2, 1e-2, -1.0, -1.0],
                      hi=[1.0, 8.0, 1e5, 3.0, 3.0]),
    "V3_L1_amp_g0": dict(f=f_v3, p0=[0.9, 1.6, 600.0, 0.2], lo=[1e-3, 0.2, 1e-2, -1.0], hi=[1.0, 8.0, 1e5, 3.0]),
    "V4_L2_amp": dict(f=f_v4, p0=[0.9, 1.6, 200.0, 0.2, 0.2], lo=[1e-3, 0.2, 1e-2, -1.0, -1.0],
                      hi=[1.0, 8.0, 1e5, 3.0, 3.0]),
    "V5_sat_amp": dict(f=f_v5, p0=[0.9, 1.6, 200.0, 0.2, 1.0], lo=[1e-3, 0.2, 1e-2, -1.0, -3.0],
                       hi=[1.0, 8.0, 1e5, 3.0, 5.0]),
    "V6_step_onset": dict(f=f_v6, p0=[0.9, 1.6, 3.0, 1.0, 0.2], lo=[1e-3, 0.2, 1e-3, 0.0, -1.0],
                          hi=[1.0, 8.0, 1e5, 3.0, 3.0]),
}


def fit(spec, N, E, M, mdetTP):
    best = None
    for s0 in (0.5, 1.0, 2.0):
        p0 = list(spec["p0"])
        p0[0] = min(0.98, p0[0] * s0 / 1.5)
        try:
            r = least_squares(lambda p: L._resid(spec["f"], p, N, E, M, mdetTP), p0,
                              bounds=(spec["lo"], spec["hi"]), max_nfev=20000, x_scale="jac")
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


rows = []
for name, spec in VARIANTS.items():
    r = fit(spec, N, E, M, mdet)
    if r is None:
        print(f"{name}: 拟合失败")
        continue
    p = r.x
    pred = np.clip(L._pred(spec["f"], p, N, E), 1e-9, 1.0)
    yo = np.log10(M[obs] + L.EPS_LOG)
    yp = np.log10(pred[obs] + L.EPS_LOG)
    r2 = 1 - np.sum((yo - yp) ** 2) / np.sum((yo - yo.mean()) ** 2)
    sse = float(np.sum(L._resid(spec["f"], p, N, E, M, mdet) ** 2))
    k = len(p)
    n = len(M)
    aicc = n * np.log(max(sse / n, 1e-12)) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)

    # 留一 N：被留出的一档若全删失，只记合规率
    per = {}
    for held in sorted(set(N.astype(int))):
        tr = (N != held)
        te = (N == held)
        rr = fit(spec, N[tr], E[tr], M[tr], mdet[tr])
        if rr is None:
            continue
        pr = np.clip(L._pred(spec["f"], rr.x, N[te], E[te]), 0, 1)
        o = M[te] > 0
        if o.sum() > 1:
            a = np.log10(M[te][o] + L.EPS_LOG)
            b = np.log10(pr[o] + L.EPS_LOG)
            per[held] = float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
        else:
            per[held] = None
    r2s = [v for v in per.values() if v is not None]
    rows.append(dict(model=name, k=k, r2_log=round(float(r2), 4), aicc=round(float(aicc), 1),
                     loo_mean=round(float(np.mean(r2s)), 3) if r2s else None,
                     loo_min=round(float(np.min(r2s)), 3) if r2s else None,
                     params=[round(float(v), 3) for v in p],
                     per_n={str(a): (None if b is None else round(b, 3)) for a, b in per.items()}))

out = pd.DataFrame(rows).sort_values("aicc")
pd.set_option("display.width", 200)
print(out[["model", "k", "r2_log", "aicc", "loo_mean", "loo_min"]].to_string(index=False))
print()
for r in rows:
    print(f"{r['model']:<16} params={r['params']}  留一明细={r['per_n']}")
out.to_csv(C.BUILD / "law_variants.csv", index=False)
print(f"\n写出 {C.BUILD / 'law_variants.csv'}")