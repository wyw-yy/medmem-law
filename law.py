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
MEAS_FLOOR = 0.02  # 高于 4 倍假阳性地板（FPR=0.5%）才算"可测"


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


# L4/L5：除了「什么时候开始背」，还允许「最多能背多少」随 N 衰减。
# 动机（deep 档实测）：N=250 的平台 0.80，N=500 只有 0.40，N=2000/4000 在
# 9 万步内都低于检出限（M_max 0.006/0.016）。如果强行让 Minf 与 N 无关
# （L1/L2/L3），模型必然在大 N 处系统性高估 —— 这正是留一 N 崩的原因。
# 理论上这也对应 Bonnaire et al.：数据越多，越不存在可长期记住的样本。
def model_L4(p, N, E):
    """幅度衰减 + 时间尺度平移，两者都锚定在 N0=250。

    M(N,E) = A0·(N/N0)^(-α) · [1 - exp(-(E/E50)^β)],  E50 = E50_0·(N/N0)^γ

    参考点锚定的原因：用 c·N^γ 时 c 和 γ 强相关（N 很大时 E50 不可见，
    优化器会把 c 顶到上界 1e5、γ 掉到 -0.67 这种不可解释的角落）。
    """
    A0, alpha, beta, E50_0, gamma = p
    r = np.maximum(N, 1.0) / 250.0
    amp = A0 * np.power(r, -alpha)
    return amp * (1.0 - np.exp(-np.power(E / (E50_0 * np.power(r, gamma)), beta)))


def model_L6(p, N, E):
    """幅度随 N **指数**衰减（物理动机：单个样本的平均曝光量 ∝ 步数/N）。

    M(N,E) = A0·exp(-(N-250)/Nc) · [1 - exp(-(E/E50)^β)],  E50 = E50_0·(N/250)^γ

    为什么加这条：实测平台高度 0.79/0.39/0.075/≈0/≈0.007 的逐次减半倍数是
    2.0 / 5.2 / >10 / ≈1，**不是常数**，因此幂律（L4/L5）在 N=500 处系统性低估
    2–3 倍；而指数衰减 e^{-250/Nc} 给出 2.0 / 4.0 / 16，与实测吻合得多。
    """
    A0, Nc, beta, E50_0, gamma = p
    r = np.maximum(N, 1.0) / 250.0
    amp = A0 * np.exp(-(np.maximum(N, 1.0) - 250.0) / max(Nc, 1.0))
    return amp * (1.0 - np.exp(-np.power(E / (E50_0 * np.power(r, gamma)), beta)))


def model_L5(p, N, E):
    """只有幅度衰减，时间尺度与 N 无关（γ=0）。"""
    A0, alpha, beta, E50_0 = p
    r = np.maximum(N, 1.0) / 250.0
    amp = A0 * np.power(r, -alpha)
    return amp * (1.0 - np.exp(-np.power(E / max(E50_0, 1e-6), beta)))


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
    "L4_amp_decay": dict(f=model_L4, k=5,
                         pnames=["A0_at250", "alpha", "beta_h", "E50_at250", "gamma"],
                         p0=[0.85, 2.0, 1.5, 700.0, 0.2],
                         lo=[1e-3, -2.0, 0.2, 1e-2, -1.0], hi=[1.0, 6.0, 8.0, 1e5, 3.0]),
    "L5_amp_decay_only": dict(f=model_L5, k=4,
                              pnames=["A0_at250", "alpha", "beta_h", "E50_at250"],
                              p0=[0.85, 2.0, 1.5, 700.0],
                              lo=[1e-3, -2.0, 0.2, 1e-2], hi=[1.0, 6.0, 8.0, 1e5]),
    "L6_amp_exp": dict(f=model_L6, k=5,
                       pnames=["A0_at250", "Nc", "beta_h", "E50_at250", "gamma"],
                       p0=[0.85, 400.0, 1.5, 700.0, 0.2],
                       lo=[1e-3, 50.0, 0.2, 1e-2, -1.0], hi=[1.0, 5000.0, 8.0, 1e5, 3.0]),
}


def _gamma_of(name: str, params) -> float:
    """取 gamma 参数；没有 gamma 的律（L5）视为 0（时间尺度与 N 无关）。"""
    pn = CANDIDATES[name]["pnames"]
    return float(params[pn.index("gamma")]) if "gamma" in pn else 0.0


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


def _p0s(spec: dict) -> list[list[float]]:
    """多起点初值：按**参数名**生成，不能硬编码下标。

    踩过的坑：早期用 `p0[3] = g0` 把第 4 个参数当 gamma 扫，但 L4/L5 的第 4 个
    参数是 E50（不是 gamma）→ 等于把时间尺度钉死在 0.3–1.5，优化器只能靠把
    E50 顶到上界 1e5 来逃，得到一个退化解（γ=-0.67、E50=1e5）。
    """
    base = list(spec["p0"])
    pn = spec["pnames"]
    gammas = [0.0, 0.5, 1.0, 1.5] if "gamma" in pn else [None]
    outs = []
    for g in gammas:
        for s in (0.5, 1.0, 3.0):
            p = list(base)
            if g is not None:
                p[pn.index("gamma")] = g
            for nm in ("E50_per_N", "E50_at250", "E50", "a_tau"):
                if nm in pn:
                    i = pn.index(nm)
                    p[i] = float(np.clip(base[i] * s, spec["lo"][i], spec["hi"][i]))
            outs.append(p)
    return outs


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
    for p0 in _p0s(spec):
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

    # 「可测区间」的 R²：只在 M 明显高于度量假阳性地板（0.5%）的点上算。
    # 检出限附近（M≈0.004，512 张里 2 张）的点相对误差天然巨大，
    # 把它们算进 R² 等于用"噪声除以噪声"，会低估律在真正可测区域的质量。
    meas = M >= MEAS_FLOOR
    if int(meas.sum()) > 2:
        ym = np.log10(M[meas] + EPS_LOG)
        pm = np.log10(np.clip(pred[meas], 1e-9, 1.0) + EPS_LOG)
        r2_log_meas = float(1 - np.sum((ym - pm) ** 2) / max(np.sum((ym - ym.mean()) ** 2), 1e-12))
        r2_lin_meas = float(1 - np.sum((M[meas] - pred[meas]) ** 2)
                            / max(np.sum((M[meas] - M[meas].mean()) ** 2), 1e-12))
    else:
        r2_log_meas = r2_lin_meas = float("nan")

    # 删失点的合规率：预测值是否落在检出限以下
    cens_ok = None
    if (~obs).any():
        cens_ok = float(np.mean(pred[~obs] <= mdet[~obs]))

    nobs, k = len(df), spec["k"]
    aic = nobs * np.log(max(sse / nobs, 1e-12)) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(nobs - k - 1, 1)
    return dict(name=name, params=[float(v) for v in p], pred=pred,
                r2_log=float(r2_log), r2_lin=float(r2_lin),
                r2_log_meas=float(r2_log_meas), r2_lin_meas=float(r2_lin_meas),
                n_meas=int(meas.sum()),
                rmse_log=float(np.sqrt(sse / nobs)), aicc=float(aicc), k=k,
                n_obs=int(obs.sum()), n_censored=int((~obs).sum()),
                censored_compliance=cens_ok)


def loo_by_n(name: str, df: pd.DataFrame) -> dict:
    """留一 N：用其余 N 拟合，预测被留出的 N。

    被留出的这一档如果全是 M=0，就没有 R² 可言 —— 只报告"预测是否落在检出限以下"。
    R² 只 pooled 在真正有信号的档上，这才是诚实的外推检验。
    """
    out = {}
    pooled_y, pooled_p = [], []
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
        for p0 in _p0s(spec):
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
        if o.any():
            pooled_y.append(np.log10(yte[o] + EPS_LOG))
            pooled_p.append(np.log10(pred[o] + EPS_LOG))
    r2s = [v["r2_log"] for v in out.values() if v["r2_log"] is not None]
    cc = [v["censored_compliance"] for v in out.values() if v["censored_compliance"] is not None]
    # 池化 R²：把所有留出折的预测拼在一起算一个 R²。
    # 为什么必须有它：某一折如果几乎全是 M≈0（大 N 档），折内方差≈0，折内 R²
    # 会被一个极小的分母放大成 −20 这种毫无意义的数（实测 LOO 出现 −1.2e6）。
    r2_pooled = None
    if pooled_y:
        ay, ap = np.concatenate(pooled_y), np.concatenate(pooled_p)
        r2_pooled = float(1 - np.sum((ay - ap) ** 2) / max(np.sum((ay - ay.mean()) ** 2), 1e-12))
    return {"per_n": out, "r2_mean": float(np.mean(r2s)) if r2s else None,
            "r2_min": float(np.min(r2s)) if r2s else None,
            "r2_pooled": r2_pooled,
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


def _stopped_runs() -> set[str]:
    """被干预停掉的 run（开环律 stop_rule，或在线闭环 stop_eps）。

    为什么必须排除：这些曲线在停点处被截断，停点之后的数据缺失。若混进拟合，
    (a) 长 E 区间的样本被系统性挖空，(b) 更糟——停点本身就是"按 M≥ε 停"选出来的，
    把它们的样本点算进去会让律在 ε 附近显得特别准，属于自我实现。
    """
    out: set[str] = set()
    for p in sorted((C.RESULTS / "runs").glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("stop_rule") or j.get("stop_eps") is not None:
            out.add(j.get("run", p.stem))
    return out


def load_metrics(path: Path | None = None, drop_stopped: bool = True) -> pd.DataFrame:
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
    # 被律提前停掉的 run 是"干预组"，不是观察组：曲线在 E* 处截断，
    # 拿来拟合会把中后期的高 M 点人为删掉、把曲线压平。
    if drop_stopped:
        bad = _stopped_runs()
        if bad:
            n0 = len(df)
            df = df[~df["run"].isin(bad)].copy()
            print(f"[law] 排除 {len(bad)} 个被早停干预的 run（{n0} → {len(df)} 点）")
    return df


def collapse_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """把同一 (dataset, N, epoch) 的多个 seed 折成一条平均曲线。

    为什么要这一步：M(N, E) 描述的是**期望记忆率**，seed 之间的差异是噪声，
    不该算进模型误差里（否则 R² 变成"律 vs 单条曲线"的比较，反而惩罚了做重复实验这件事）。
    折完后 R² 是"律 vs 种子均值"的拟合优度；单条曲线上的偏差另外用 LOO 池化看。
    """
    keys = ["dataset", "n", "epoch"]
    g = df.groupby(keys, as_index=False).agg(
        M=("M", "mean"),
        tau=("tau", "mean"),
        n_gen=("n_gen", "mean"),
        n_seeds=("M", "size"),
    )
    g["run"] = g["dataset"] + "_n" + g["n"].astype(str) + "_mean"
    g["seed"] = 0
    return g


def main(metrics_csv: str | None = None, out: str | None = None,
         datasets: list[str] | None = None, seeds: list[int] | None = None,
         per_seed: bool = False):
    df = load_metrics(Path(metrics_csv) if metrics_csv else None)
    if datasets:
        df = df[df["dataset"].isin(datasets)].copy()
    if seeds:
        df = df[df["seed"].isin(seeds)].copy()
    if not len(df):
        raise SystemExit("筛选后没有测量点 —— 检查 --dataset / --seed")
    n_runs = df["run"].nunique()
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
    # 默认按 (dataset, N, epoch) 折种子（律描述期望记忆率）；--per-seed 保留每条曲线
    if not per_seed and n_seeds > 1:
        df = collapse_seeds(df)
        print(f"[law] 已把 {n_seeds} 个 seed 折成种子均值曲线（R² 是「律 vs 均值」的拟合优度）")
    print(f"读入 {len(df)} 个测量点，N={sorted(df['n'].unique())}，"
          f"run={n_runs}，E∈[{df['epoch'].min()},{df['epoch'].max()}]"
          + (f"，数据集={datasets}" if datasets else "")
          + (f"，seed={seeds}" if seeds else ""))

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
                     loo["r2_min"], loo.get("censored_compliance_mean"),
                     loo.get("r2_pooled"), fit.get("r2_log_meas"),
                     fit.get("r2_lin_meas"), fit.get("n_meas")))

    print("\n候选律对比（选型按 AICc 最小 + 留一 R² 最大；删失点只进单侧惩罚，不进 R²）")
    print("  R²(meas) = 只在 M ≥ 0.02 的可测点上算 —— 检出限附近的点相对误差天然巨大")
    print(f"{'律':<20}{'R²_log':>9}{'R²_lin':>9}{'R²log_m':>9}{'R²lin_m':>9}{'n_m':>5}"
          f"{'LOO池化':>9}{'删失合规':>10}")
    for r in rows:
        print(f"{r[0]:<20}{r[1]:>9.4f}{r[2]:>9.4f}"
              f"{(r[8] if r[8] is not None else float('nan')):>9.4f}"
              f"{(r[9] if r[9] is not None else float('nan')):>9.4f}"
              f"{(r[10] if r[10] is not None else 0):>5d}"
              f"{(r[7] if r[7] is not None else float('nan')):>9.4f}"
              f"{(r[6] if r[6] is not None else float('nan')):>10.2f}")
    print(f"{'':<20}{'':>9}{'':>9}{'':>9}{'':>9}   AICc / LOO折内 / LOO_min 见 results/law_fit.json")

    if not rows:
        raise SystemExit("所有候选律都拟合失败，检查 metrics_long.csv")
    # 选型：AICc 最小者（并列时看 LOO）
    best_name = min(rows, key=lambda r: (r[3], -(r[4] or -9)))[0]
    best = dict(fits[best_name])
    gam = _gamma_of(best_name, best["params"])
    print(f"\n选定：{best_name}  R²_log={best['r2_log']:.4f}  "
          f"LOO池化={best['loo'].get('r2_pooled')}")
    print(f"γ = {gam:+.3f}  （epoch 坐标：E50 = E50_0·(N/250)^γ；"
          f"换算到优化步坐标 t50 ∝ N^(γ+1) = N^{gam + 1:+.2f}）")
    if "alpha" in CANDIDATES[best_name]["pnames"]:
        i = CANDIDATES[best_name]["pnames"].index("alpha")
        print(f"α = {best['params'][i]:.3f}  （平台高度 ∝ N^-α：α>0 = 数据越多越背不住）")

    # 敏感性检验：只用"有信号"的 N 重拟合。若 γ 明显变化，说明大 N 档全靠删失约束撑着，
    # 结论必须写成"在可测范围内"。
    sig_n = [int(n) for n, v in df.groupby("n")["M"].max().items() if v > 0]
    gamma_unc, r2_unc = None, None
    if len(sig_n) >= 3:
        try:
            fu = fit_one(best_name, df[df["n"].isin(sig_n)])
            gamma_unc = float(_gamma_of(best_name, fu["params"]))
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
        "gamma_hat": _gamma_of(best_name, best["params"]),
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
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="拟合记忆化控制律 M(N,E)")
    ap.add_argument("--metrics", default=None, help="测量 CSV（默认 results/metrics_long.csv）")
    ap.add_argument("--out", default=None, help="输出 JSON（默认 results/law_fit.json）")
    ap.add_argument("--dataset", default="", help="只用某个数据集，逗号分隔；跨数据集验证时用")
    ap.add_argument("--seed", default="", help="只用某些 seed，逗号分隔")
    ap.add_argument("--per-seed", action="store_true",
                    help="不折种子，把每个 seed 当独立观测拟合（口径敏感性检验用）")
    a = ap.parse_args(sys.argv[1:])
    dst = [x for x in a.dataset.split(",") if x] or None
    sds = [int(x) for x in a.seed.split(",") if x] or None
    out = a.out
    if out is None and dst:
        out = str(C.RESULTS / f"law_fit_{'_'.join(dst)}.json")
    main(a.metrics, out, datasets=dst, seeds=sds, per_seed=a.per_seed)