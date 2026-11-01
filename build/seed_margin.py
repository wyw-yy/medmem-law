"""B11 交付分析：seed 方差 → 带余量的开环早停建议 `E*_safe`。

问题：律给的是期望时刻表；7 个 seed 的 onset 相差最多 8–20 倍，所以"按律停"会踩雷。
这里把 seed 方差算成可交付的量：
    E*_safe(N, ε) = E*_law(N, ε) · 10^(−k·σ_log(N))，  σ_log = std(log10 onset_epoch)
并把"该口径下有几个 seed 合规、省了多少轮"报出来（预声明见 PROTOCOL §20 B11）。

输出 `results/seed_margin.csv`（每个 dataset/N/ε 一行）。
用法：python -m build.seed_margin [--k 2] [--dataset pathmnist]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as C
import law as L

EPS_SET = (0.01, 0.02, 0.05)


def onset_epoch(g: pd.DataFrame, eps: float) -> float:
    """首次连续两个测量点 M ≥ eps 的轮次；没有则 NaN。"""
    g = g.sort_values("epoch")
    e = g["epoch"].to_numpy(dtype=float)
    m = g["M"].to_numpy(dtype=float)
    for i in range(len(m) - 1):
        if m[i] >= eps and m[i + 1] >= eps:
            return float(e[i])
    return float("nan")


def fid_at(sub: pd.DataFrame, epoch: float) -> float:
    """停点处的 FID：用该 N 下 seed 1 的 FID 采样点线性插值（没测过 FID 就返回 nan）。"""
    s1 = sub[sub["seed"] == 1].sort_values("epoch")
    pts = s1[s1["fid"].notna()]
    if len(pts) == 0:
        return float("nan")
    x = pts["epoch"].to_numpy(dtype=float)
    y = pts["fid"].to_numpy(dtype=float)
    return float(np.interp(epoch, x, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=2.0, help="余量倍数（默认 2σ）")
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--eps-set", default=",".join(str(x) for x in EPS_SET))
    a = ap.parse_args()
    eps_set = [float(x) for x in a.eps_set.split(",")]

    df = pd.read_csv(C.RESULTS / "metrics_long.csv")
    df = df[df["dataset"] == a.dataset]
    law_fit = __import__("json").loads((C.RESULTS / "law_fit.json").read_text(encoding="utf-8"))
    best = law_fit["best"]
    params = law_fit["fits"][best]["params"]

    rows = []
    for eps in eps_set:
        for n in sorted(df["n"].unique()):
            sub = df[df["n"] == n]
            onsets, lasts = [], []
            for seed, g in sub.groupby("seed"):
                o = onset_epoch(g, eps)
                if o == o:
                    onsets.append(o)
                lasts.append(float(g.sort_values("epoch")["M"].iloc[-1]))
            if not onsets:
                rows.append(dict(dataset=a.dataset, n=int(n), eps=eps, n_seeds=len(sub["seed"].unique()),
                                 n_onset=0, onset_med=np.nan, sigma_log=np.nan,
                                 E_law=np.nan, E_safe=np.nan, saving_safe=np.nan,
                                 M_law=np.nan, M_safe=np.nan, viol_law=np.nan, viol_safe=np.nan))
                continue

            sigma_log = float(np.std(np.log10(onsets), ddof=1)) if len(onsets) > 1 else 0.0
            e_law = float(L.solve_E_star(best, params, int(n), eps))
            # 大 N 档律可能永远达不到 ε（solve_E_star 返回 nan）——那就没有"按律停"这一说
            e_safe = e_law * 10 ** (-a.k * sigma_log) if e_law == e_law else float("nan")
            budget = float(C.epochs_for(int(n)))

            def m_at(epoch: float) -> float:
                """停点处的 M：取不超过该轮次的最后一个实测点（保守），没有就取第一个点。"""
                best_m = np.nan
                for seed, g in sub.groupby("seed"):
                    gg = g.sort_values("epoch")
                    cand = gg[gg["epoch"] <= epoch]["M"]
                    v = float(cand.iloc[-1]) if len(cand) else float(gg["M"].iloc[0])
                    best_m = v if best_m != best_m else max(best_m, v)
                return best_m

            m_law, m_safe = m_at(e_law), m_at(e_safe)
            f_law = fid_at(sub, e_law) if e_law == e_law else float("nan")
            f_safe = fid_at(sub, e_safe) if e_safe == e_safe else float("nan")
            f_full = fid_at(sub, float(sub["epoch"].max()))

            def r(x):
                return round(float(x)) if x == x else float("nan")

            rows.append(dict(
                dataset=a.dataset, n=int(n), eps=eps, n_seeds=len(sub["seed"].unique()),
                n_onset=len(onsets), onset_med=float(np.median(onsets)), sigma_log=sigma_log,
                E_law=r(e_law), E_safe=r(e_safe),
                saving_safe=round(1 - e_safe / budget, 3) if e_safe == e_safe else float("nan"),
                M_law=m_law, M_safe=m_safe,
                fid_law=f_law, fid_safe=f_safe, fid_full=f_full,
                fid_rel_safe=(round(f_safe / f_full, 2) if f_safe == f_safe and f_full == f_full and f_full else float("nan")),
                viol_law=int(m_law > eps) if m_law == m_law else -1,
                viol_safe=int(m_safe > eps) if m_safe == m_safe else -1,
            ))

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "seed_margin.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(f"律 {best}  params={np.round(params, 4).tolist()}  k={a.k}σ")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    ok = out[(out["viol_safe"] == 0) & (out["saving_safe"] >= 0.30)]
    print(f"\n既安全（最差 seed 合规）又省 ≥30% 的行：{len(ok)}/{len(out)}")
    if len(ok):
        print(ok[["n", "eps", "E_safe", "saving_safe", "M_safe"]].to_string(index=False))
    print(f"\n写出 {C.RESULTS / 'seed_margin.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())