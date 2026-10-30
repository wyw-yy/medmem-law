"""
控制律拟合：M(N,E) → R²、AICc、留一 N 交叉验证 → 反解安全早停 E*(N,ε)。

读 results/metrics_long.csv（列：run,n,seed,epoch,M,...），输出 results/law_fit.json。

三个候选律（先写死选型判据，避免事后挑最好看的）：
  L1 拉伸指数  M = Minf * (1 - exp(-(E/E50)^beta)),  E50 = c * N^gamma
  L2 Hill      M = Minf * E^h / (E^h + K^h),         K   = c * N^gamma
  L3 双时间尺度 M = Minf * Phi((E - tau)/w),          tau = a * N^gamma

选型判据（预先声明）：
  1) AICc 最小；2) 留一 N 的平均 R² 最大；3) 参数能讲出临床含义。
拟合在对数空间做（log10(M + 1e-4)），因为小 M 区间才是安全边界所在，
用线性空间拟合会被 M≈Minf 的饱和段主导。同时报告两种 R²。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.special import erf
from scipy.stats import norm

import config as C

EPS_LOG = 1e-4


# ------------------------------------------------------------------ 候选律
def model_L1(p, N, E):
    Minf, beta, c, gamma = p
    E50 = c * np.power(N, gamma)
    return Minf * (1.0 - np.exp(-np.power(E / E50, beta)))


def model_L2(p, N, E):
    Minf, h, c, gamma = p
    K = c * np.power(N, gamma)
    r = np.power(np.maximum(E, 1e-9), h)
    return Minf * r / (r + np.power(K, h))


def model_L3(p, N, E):
    Minf, a, gamma, w = p
    tau = a * np.power(N, gamma)
    return Minf * 0.5 * (1.0 + erf((E - tau) / max(w, 1e-6)))


CANDIDATES = {
    "L1_stretched_exp": dict(f=model_L1, k=4,
                             pnames=["Minf", "beta_h", "E50_per_N", "gamma"],
                             p0=[0.5, 1.5, 200.0, 1.0],
                             lo=[1e-3, 0.2, 1e-2, -1.0], hi=[1.0, 8.0, 1e5, 3.0]),
    "L2_hill": dict(f=model_L2, k=4,
                    pnames=["Minf", "beta_h", "E50_per_N", "gamma"],
                    p0=[0.5, 1.5, 200.0, 1.0],
                    lo=[1e-3, 0.2, 1e-2, -1.0], hi=[1.0, 8.0, 1e5, 3.0]),
    "L3_two_timescale": dict(f=model_L3, k=4,
                             pnames=["Minf", "a_tau", "gamma", "w"],
                             p0=[0.9, 1.0, 1.0, 40.0],
                             lo=[1e-3, 1e-2, -1.0, 1.0], hi=[1.0, 1e5, 3.0, 1e4]),
}


def _pred(f, p, N, E):
    with np.errstate(all="ignore"):
        v = f(p, N, E)
    return np.nan_to_num(v, nan=1.0, posinf=1.0, neginf=0.0)


# ---------------------------------------------------------------- 删失（censoring）
# 为什么必须处理：等步数预算下，N=2000/4000 的每个测量点都是 M=0（512 张样本里
# 一张都没超阈值）。这不是"记忆率恰好等于 1e-4"，而是"**低于检出限**"。
# 直接按 log(M+1e-4)=-4 当观测点去拟合，等于硬塞 33 个假观测：
# 留一 N 时留出这两档，模型的外推残差会炸到 1e6 量级（实测 LOO R²=-1.2e6）。
# 正确做法是**单侧惩罚**：预测高于检出限才罚，低于就几乎不贡献残差。
CENSOR_K = 4.0      # softplus 的陡度，越大越接近硬不等式约束


def _mdet(df: pd.DataFrame) -> np.ndarray:
    """每个测量点的检出限：0 次命中时的 95% 二项上界 ≈ 3/n_gen。"""
    if "n_gen" in df.columns:
        ng = df["n_gen"].astype(float).to_numpy()
    else:
        ng = np.full(len(df), float(C.N_SAMPLE))
    return 3.0 / np.maximum(ng, 1.0)


def _resid(f, p, N, E, M, mdet):
    """残差向量：有信号的点和删失点用两套规则。

    - M>0：log10 空间的普通残差（相对误差，符合乘性噪声）。
    - M=0：`softplus(k*(log10 pred - log10 mdet))`，预测低于检出限时≈0，
      高于时≈(log10 pred - log10 mdet)。既平滑（可做最小二乘）又是单侧的。
    """
    pred = np.clip(_pred(f, p, N, E), 1e-9, 1.0)
    lp = np.log10(pred + EPS_LOG)
    obs = M > 0
    r = np.zeros(len(M))
    r[obs] = lp[obs] - np.log10(M[obs] + EPS_LOG)
    if (~obs).any():
        z = CENSOR_K * (lp[~obs] - np.log10(mdet[~obs] + EPS_LOG))
        r[~obs] = np.log1p(np.exp(np.clip(z, -30, 30))) / CENSOR_K
    return r


def fit_one(name: str, df: pd.DataFrame) -> dict:
    spec = CANDIDATES[name]
    f = spec["f"]
    N = df["n"].to_numpy(float)
    E = df["epoch"].to_numpy(float)
    M = df["M"].to_numpy(float)
    mdet = _mdet(df)
    obs = M > 0

    def resid(p):
        return _resid(f, p, N, E, M, mdet)

    best = None
    # 多起点：非凸问题，单起点容易卡在 gamma 的假解上
    for g0 in (0.3, 0.6, 1.0, 1.5):
        for s0 in (0.5, 1.0, 2.0):
            p0 = list(spec["p0"])
            p0[3] = g0
            p0[0] = p0[0] * min(1.0, s0)
            try:
                r = least_squares(resid, p0, bounds=(spec["lo"], spec["hi"]),
                                  max_nfev=20000, x_scale="jac")
            except Exception:  # noqa: BLE001
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        raise RuntimeError(f"{name} 拟合失败")
    p = best.x
    pred = _pred(f, p, N, E)

    # R² 只在"有信号"的点上算 —— 删失点没有可比的 Y 值，塞进去会把 R² 虚抬
    sse = float(np.sum(resid(p) ** 2))
    if obs.sum() > 1:
        yo = np.log10(M[obs] + EPS_LOG)
        yp = np.log10(np.clip(pred[obs], 1e-9, 1.0) + EPS_LOG)
        r2_log = float(1 - np.sum((yo - yp) ** 2) / max(np.sum((yo - yo.mean()) ** 2), 1e-12))
        ss = float(np.sum((M[obs] - pred[obs]) ** 2))
        sst_lin = float(np.sum((M[obs] - M[obs].mean()) ** 2))
        r2_lin = float(1 - ss / max(sst_lin, 1e-12))
    else:
        r2_log = r2_lin = float("nan")

    # 删失点的合规率：预测值是否落在检出限以下
    cens_ok = None
    if (~obs).any():
        cens_ok = float(np.mean(pred[~obs] <= mdet[~obs]))

    nobs, k = len(df), spec["k"]
    aic = nobs * np.log(max(sse / nobs, 1e-12)) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(nobs - k - 1, 1)
    return dict(name=name, params=[float(v) for v in p], pred=pred,
                r2_log=float(r2_log), r2_lin=float(r2_lin),
                rmse_log=float(np.sqrt(sse / nobs)), aicc=float(aicc), k=k,
                n_obs=int(obs.sum()), n_censored=int((~obs).sum()),
                censored_compliance=cens_ok)


def loo_by_n(name: str, df: pd.DataFrame) -> dict:
    """留一 N：用其余 N 拟合，预测被留出的 N。

    被留出的这一档如果全是 M=0，就没有 R² 可言 —— 只报告"预测是否落在检出限以下"。
    R² 只 pooled 在真正有信号的档上，这才是诚实的外推检验。
    """
    out = {}
    for held in sorted(df["n"].unique()):
        tr = df[df["n"] != held]
        te = df[df["n"] == held]
        if len(te) == 0 or len(tr) < 8:
            continue
        spec = CANDIDATES[name]
        f = spec["f"]
        Mtr = tr["M"].to_numpy(float)
        mtr = _mdet(tr)

        def resid(p):
            return _resid(f, p, tr["n"].to_numpy(float), tr["epoch"].to_numpy(float), Mtr, mtr)

        best = None
        for g0 in (0.3, 0.6, 1.0, 1.5):
            p0 = list(spec["p0"])
            p0[3] = g0
            try:
                r = least_squares(resid, p0, bounds=(spec["lo"], spec["hi"]), max_nfev=20000)
            except Exception:  # noqa: BLE001
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            continue
        pred = np.clip(_pred(f, best.x, te["n"].to_numpy(float), te["epoch"].to_numpy(float)), 0.0, 1.0)
        yte = te["M"].to_numpy(float)
        mt = _mdet(te)
        o = yte > 0
        r2 = None
        if o.sum() > 1:
            yo = np.log10(yte[o] + EPS_LOG)
            yp = np.log10(pred[o] + EPS_LOG)
            r2 = float(1 - np.sum((yo - yp) ** 2) / max(np.sum((yo - yo.mean()) ** 2), 1e-12))
        out[str(int(held))] = {
            "r2_log": r2,
            "mae": float(np.mean(np.abs(yte[o] - pred[o]))) if o.any() else None,
            "censored_compliance": (float(np.mean(pred[~o] <= mt[~o])) if (~o).any() else None),
            "n_obs": int(o.sum()),
        }
    r2s = [v["r2_log"] for v in out.values() if v["r2_log"] is not None]
    cc = [v["censored_compliance"] for v in out.values() if v["censored_compliance"] is not None]
    return {"per_n": out, "r2_mean": float(np.mean(r2s)) if r2s else None,
            "r2_min": float(np.min(r2s)) if r2s else None,
            "censored_compliance_mean": float(np.mean(cc)) if cc else None}


def solve_E_star(name: str, params, N: float, eps: float) -> float:
    """反解 M(E)=eps 的 E。无解（该 N 下永远达不到 eps）返回 nan。"""
    f = CANDIDATES[name]["f"]
    g = lambda e: float(_pred(f, params, np.array([N]), np.array([e]))[0]) - eps  # noqa: E731
    lo, hi = 1.0, 1e5
    try:
        if g(lo) >= 0:
            return 0.0
        if g(hi) < 0:
            return float("nan")
        return float(brentq(g, lo, hi, maxiter=500))
    except Exception:  # noqa: BLE001
        return float("nan")


def load_metrics(path: Path | None = None) -> pd.DataFrame:
    path = path or (C.RESULTS / "metrics_long.csv")
    df = pd.read_csv(path)
    need = {"run", "n", "epoch", "M"}
    if not need.issubset(df.columns):
        raise ValueError(f"metrics_long.csv 缺列：{need - set(df.columns)}")
    df = df[df["M"].notna()].copy()
    df["n"] = df["n"].astype(int)
    df["epoch"] = df["epoch"].astype(int)
    # 同一 (run,epoch) 多次评分取最后一次
    df = df.sort_values("epoch").drop_duplicates(["run", "epoch"], keep="last")
    return df


def main(metrics_csv: str | None = None, out: str | None = None):
    df = load_metrics(Path(metrics_csv) if metrics_csv else None)
    print(f"读入 {len(df)} 个测量点，N={sorted(df['n'].unique())}，"
          f"run={df['run'].nunique()}，E∈[{df['epoch'].min()},{df['epoch'].max()}]")

    fits, rows = {}, []
    for name in CANDIDATES:
        try:
            fit = fit_one(name, df)
            loo = loo_by_n(name, df)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: 失败 {type(e).__name__}: {e}")
            continue
        fits[name] = {k: v for k, v in fit.items() if k != "pred"} | {"loo": loo}
        rows.append((name, fit["r2_log"], fit["r2_lin"], fit["aicc"], loo["r2_mean"],
                     loo["r2_min"], loo.get("censored_compliance_mean")))

    print("\n候选律对比（选型按 AICc 最小 + 留一 R² 最大；删失点只进单侧惩罚，不进 R²）")
    print(f"{'律':<20}{'R²_log':>9}{'R²_lin':>9}{'AICc':>10}{'LOO_R²':>9}{'LOO_min':>9}{'删失合规':>10}")
    for r in rows:
        print(f"{r[0]:<20}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>10.1f}"
              f"{(r[4] or float('nan')):>9.4f}{(r[5] or float('nan')):>9.4f}"
              f"{(r[6] if r[6] is not None else float('nan')):>10.2f}")

    if not rows:
        raise SystemExit("所有候选律都拟合失败，检查 metrics_long.csv")
    # 选型：AICc 最小者（并列时看 LOO）
    best_name = min(rows, key=lambda r: (r[3], -(r[4] or -9)))[0]
    best = dict(fits[best_name])
    gam = best["params"][2] if best_name == "L3_two_timescale" else best["params"][3]
    print(f"\n选定：{best_name}  R²_log={best['r2_log']:.4f}  LOO_R²={best['loo']['r2_mean']}")
    print(f"γ = {gam:.3f}  （E50 = c·N^γ；γ≈1 = 数据翻倍，开始背的 epoch 也翻倍）")

    # 敏感性检验：只用"有信号"的 N 重拟合。若 γ 明显变化，说明大 N 档全靠删失约束撑着，
    # 结论必须写成"在可测范围内"。
    sig_n = [int(n) for n, v in df.groupby("n")["M"].max().items() if v > 0]
    gamma_unc, r2_unc = None, None
    if len(sig_n) >= 3:
        try:
            fu = fit_one(best_name, df[df["n"].isin(sig_n)])
            gamma_unc = float(fu["params"][2] if best_name == "L3_two_timescale" else fu["params"][3])
            r2_unc = float(fu["r2_log"])
            print(f"敏感性（只用有信号的 N={sig_n} 拟合）：γ={gamma_unc:.3f}，R²_log={r2_unc:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"敏感性拟合失败：{type(e).__name__}: {e}")
    else:
        print(f"（只有 {len(sig_n)} 档 N 有信号，做不了敏感性检验）")

    # 安全早停表
    e_star = {}
    for eps in (C.EPS_MEM, 0.05, 0.10):
        e_star[str(eps)] = {str(int(N)): solve_E_star(best_name, best["params"], N, eps)
                            for N in sorted(df["n"].unique())}
    print(f"\n安全早停 E*(N, ε)（ε=可接受记忆率）")
    print("  N  " + "".join(f"{e:>12}" for e in e_star))
    for N in sorted(df["n"].unique()):
        print(f"{N:>5}" + "".join(f"{e_star[e][str(int(N))]:>12.0f}" for e in e_star))

    result = {
        "best": best_name,
        "fits": fits,
        "E_star": e_star,
        "n_points": int(len(df)),
        "n_grid": sorted(int(x) for x in df["n"].unique()),
        "epochs": sorted(int(x) for x in df["epoch"].unique()),
        "gamma_hat": float(best["params"][2] if best_name == "L3_two_timescale" else best["params"][3]),
        "gamma_uncensored_fit": gamma_unc,
        "r2_log_uncensored_fit": r2_unc,
        "n_signal_only": sig_n,
        "censoring": {
            "rule": "M=0 视为低于检出限（0 命中时 95% 二项上界 ≈ 3/n_gen），单侧 softplus 惩罚，不进 R²",
            "n_censored_points": int((df["M"] == 0).sum()),
            "n_observed_points": int((df["M"] > 0).sum()),
        },
    }
    out_path = Path(out) if out else (C.RESULTS / "law_fit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {out_path}")
    return result


if __name__ == "__main__":
    import sys

    main(*(sys.argv[1:3]))