"""
安全早停建议表（课题的最终交付物之一）。

输入：results/metrics_long.csv（实测曲线）+ results/law_fit.json（拟合好的律）
输出：results/safe_early_stop.csv 与终端表格

对每个 (dataset, N, ε)：
  E*        律给出的安全早停轮次（M(E*) = ε 的反解）
  M_stop    在 E* 附近实测到的记忆率（应该是 ≤ ε 的最后一个测量点）
  fid_stop  在 E* 停止时的生成质量
  M_full / fid_full   跑满 E_max 时的记忆率与质量（= PPT 的基线）
  save%     相比跑满省下的轮次
  d_fid     早停带来的质量代价（fid_stop - fid_full，正数=变差）
  ok        M_stop ≤ ε 是否成立（**所有 seed 里最差的那条**都满足才算成立）
  n_seeds   该 (数据集, N) 实际参与判定的 seed 数
  M_stop_mean / E_onset_max   seed 间的均值与最晚发作点，用来看种子方差
  informative  ε 是否高于度量假阳性地板（FPR=0.005 → 地板 0.005，ε 要 > 0.01 才有信息量）
  E_onset   实测「首次连续 2 点 M ≥ ε」的轮次（不依赖律的经验发作时间，取**最早**的 seed）
  E_star_emp / M_stop_emp / ok_emp   经验保守停点：每个 seed 发作前的最后一个测量点取最小
  onset_ratio  E* / E_onset：>1 说明律的停点已经越过实测发作点（危险），
               ≪1 说明律在该数据域上过于保守（换数据域没重新标定时典型表现）
  ok_quality  早停点的 FID 相对代价 ≤ Q_TOL（默认 1.5 倍，即质量最多差 50%）
  recommended  ok and ok_emp and ok_quality and informative —— 这才是能交给医院的那一行

⚠ 本表**不把被律提前停掉的 run 当成观察点**（`law.load_metrics` 会剔除干预组），
  否则 ctl run 会在自己的停点上被当成"最差 seed"，结论自我循环。
⚠ fid 只在部分检查点测过（--fid-stride），所以 fid_stop 取「≤ E* 的最近一个有点」。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

EPS_GRID = (0.005, 0.01, 0.02, 0.05)
Q_TOL = 1.5  # 质量容忍：早停点的 FID 最多是跑满时的 1.5 倍
# τ 是按「非成员图的 FPR = FID_FPR」校准的，所以**即使一个样本都没被记住**，
# 也会有约 FID_FPR 比例的生成样本越过 τ —— 这是度量的假阳性地板。
# eps 落在地板附近（≤2 倍地板）时，M ≤ eps 这件事没有信息量，不能当成"安全"。
FP_FLOOR = float(getattr(C, "FID_FPR", 0.005))
EPS_INFORMATIVE = 2.0 * FP_FLOOR


def _onset(g: pd.DataFrame, eps: float, need: int = 2) -> float:
    """实测发作轮次：首次连续 need 个测量点 M ≥ eps。避免单点量化噪声当成发作。"""
    e = g["epoch"].to_numpy().astype(float)
    m = g["M"].to_numpy().astype(float)
    flags = m >= eps
    for i in range(len(flags) - need + 1):
        if flags[i:i + need].all():
            return float(e[i])
    return float("nan")


def _law_params(law: dict) -> tuple[str, list]:
    """law_fit.json 里 "best" 是候选律名字，参数在 fits[名字]["params"]。"""
    name = law["best"]
    return name, law["fits"][name]["params"]


def main():
    mp = C.RESULTS / "metrics_long.csv"
    lp = C.RESULTS / "law_fit.json"
    if not mp.exists() or not lp.exists():
        print("缺 metrics_long.csv 或 law_fit.json，先跑 law")
        return
    import law as L

    # 用 law 模块自己的预测函数与反解，避免公式重复实现。
    # 注意用 L.load_metrics()：它会剔除"被律提前停掉"的干预组 run，
    # 否则 ctl run 会在自己的停点上被当成"最差 seed"，结论自我循环。
    df = L.load_metrics(mp)
    law = json.loads(lp.read_text(encoding="utf-8"))
    name, params = _law_params(law)

    rows = []
    for ds, gd in df.groupby("dataset"):
        for N, gn in gd.groupby("n"):
            seeds = sorted(gn["seed"].unique()) if "seed" in gn else [0]
            for eps in EPS_GRID:
                e_star = L.solve_E_star(name, params, float(N), eps)
                if e_star != e_star:  # nan：这个 N 永远达不到 eps
                    continue
                per_seed = []
                for s in seeds:
                    g = gn[gn["seed"] == s].sort_values("epoch") if "seed" in gn else gn.sort_values("epoch")
                    at = g[g["epoch"] <= e_star]
                    if len(at) == 0:
                        continue
                    stop = at.iloc[-1]
                    fids = at.dropna(subset=["fid"])
                    ff = g["fid"].dropna()
                    per_seed.append({
                        "seed": int(s),
                        "epoch": int(stop["epoch"]),
                        "M_stop": float(stop["M"]),
                        "M_full": float(g["M"].iloc[-1]),
                        "e_full": int(g["epoch"].max()),
                        "fid_stop": float(fids["fid"].iloc[-1]) if len(fids) else np.nan,
                        "fid_full": float(ff.iloc[-1]) if len(ff) else np.nan,
                        "onset": _onset(g, eps),
                    })
                if not per_seed:
                    continue
                # 关键口径：律给的 E* 要在**所有 seed 上**都满足 M ≤ ε 才算合规。
                # 只按 seed 1 报告会掩盖种子间发作时间的差异（实测 N=250 在
                # e640 三个 seed 分别是 0.002 / 0.033 / 0.057，差 20 倍）。
                worst = max(per_seed, key=lambda r: r["M_stop"])
                fid_stop = worst["fid_stop"]
                fid_full = worst["fid_full"]
                onsets = [r["onset"] for r in per_seed if r["onset"] == r["onset"]]
                fid_rel = (fid_stop / fid_full) if (pd.notna(fid_stop) and pd.notna(fid_full) and fid_full > 0) else np.nan
                ok = bool(worst["M_stop"] <= eps)
                ok_quality = bool(pd.notna(fid_rel) and fid_rel <= Q_TOL)

                # 经验保守停点：用「每个 seed 发作前的最后一个测量点」，再取所有 seed 的**最小值**。
                # 为什么不能直接用发作轮次：发作点本身按定义就 ≥ ε，拿它当停点必然"违约"；
                # 也不能只看均值：N=250 在 e640 三个 seed 的 M 是 0.002 / 0.033 / 0.057。
                safes = []
                for r in per_seed:
                    g = gn[gn["seed"] == r["seed"]].sort_values("epoch") if "seed" in gn else gn.sort_values("epoch")
                    o = r["onset"]
                    sub = g[g["epoch"] < o] if o == o else g
                    if len(sub):
                        safes.append(int(sub["epoch"].iloc[-1]))
                e_emp = float(min(safes)) if safes else float("nan")
                m_emp, ok_emp = float("nan"), False
                if e_emp == e_emp:
                    cands = []
                    for s in seeds:
                        g = gn[gn["seed"] == s].sort_values("epoch") if "seed" in gn else gn.sort_values("epoch")
                        at = g[g["epoch"] <= e_emp]
                        if len(at):
                            cands.append(float(at.iloc[-1]["M"]))
                    if cands:
                        m_emp = max(cands)
                        ok_emp = bool(m_emp <= eps)

                rows.append({
                    "dataset": ds, "n": int(N), "eps": eps,
                    "E_star": round(float(e_star), 1),
                    "E_measured": int(worst["epoch"]),
                    "n_seeds": len(per_seed),
                    "M_stop": float(worst["M_stop"]),
                    "M_stop_mean": float(np.mean([r["M_stop"] for r in per_seed])),
                    "M_full": float(worst["M_full"]),
                    "epoch_saving": 1 - worst["epoch"] / max(worst["e_full"], 1),
                    "fid_stop": fid_stop, "fid_full": fid_full,
                    "d_fid": (fid_stop - fid_full) if pd.notna(fid_stop) and pd.notna(fid_full) else np.nan,
                    "ok": ok,
                    "E_star_emp": e_emp,
                    "M_stop_emp": m_emp,
                    "ok_emp": ok_emp,
                    "E_onset": e_emp,
                    "E_onset_max": float(max(onsets)) if onsets else float("nan"),
                    "onset_ratio": (float(e_star) / e_emp) if e_emp == e_emp and e_emp > 0 else float("nan"),
                    "fid_rel": fid_rel,
                    "ok_quality": ok_quality,
                    "informative": bool(eps > EPS_INFORMATIVE),
                    # 交付口径用**三者都要过**：律的 E* 不能晚于经验最坏发作点，
                    # 质量代价要在容忍范围内，且 eps 本身要高于度量的假阳性地板。
                    "recommended": bool(ok and ok_emp and ok_quality and eps > EPS_INFORMATIVE),
                })

    t = pd.DataFrame(rows).sort_values(["dataset", "eps", "n"])
    out = C.RESULTS / "safe_early_stop.csv"
    t.to_csv(out, index=False)
    print(f"写入 {out}（{len(t)} 行），律 = {name}\n")

    for ds, g in t.groupby("dataset"):
        print(f"== {ds} ==（M_停 取**所有 seed 里最差的那条**，不是平均）")
        print(f"{'ε':>7}{'N':>5}{'seed':>5}{'E*':>7}{'停轮':>7}{'M_停':>8}{'M_均':>8}{'M_满':>8}"
              f"{'省%':>6}{'FID停':>7}{'FID满':>7}{'ΔFID':>7}{'发作':>6}{'E*/发作':>8}{'OK':>4}{'质量':>5}{'可用':>5}")
        for _, r in g.iterrows():
            fs = f"{r.fid_stop:.1f}" if pd.notna(r.fid_stop) else "  -"
            ff = f"{r.fid_full:.1f}" if pd.notna(r.fid_full) else "  -"
            df_ = f"{r.d_fid:+.1f}" if pd.notna(r.d_fid) else "  -"
            on = f"{r.E_onset:.0f}" if pd.notna(r.E_onset) else "  -"
            ra = f"{r.onset_ratio:.2f}" if pd.notna(r.onset_ratio) else "  -"
            print(f"{r.eps:>7.3f}{r.n:>5}{int(r.n_seeds):>5}{r.E_star:>7.0f}{r.E_measured:>7}"
                  f"{r.M_stop:>8.4f}{r.M_stop_mean:>8.4f}{r.M_full:>8.4f}{100*r.epoch_saving:>6.0f}"
                  f"{fs:>7}{ff:>7}{df_:>7}{on:>6}{ra:>8}"
                  f"{'Y' if r.ok else 'N':>4}{'Y' if r.ok_quality else 'N':>5}"
                  f"{('Y' if r.recommended else 'N'):>5}{('' if r.informative else '  <-ε太低'):>10}")
        print()


if __name__ == "__main__":
    main()