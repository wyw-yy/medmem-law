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
  ok        M_stop ≤ ε 是否成立（律是否自洽）
  E_onset   实测「首次连续 2 点 M ≥ ε」的轮次（不依赖律的经验发作时间）
  onset_ratio  E* / E_onset：>1 说明律的停点已经越过实测发作点（危险），
               ≪1 说明律在该数据域上过于保守（换数据域没重新标定时的典型表现）
  ok_quality  早停点的 FID 相对代价 ≤ Q_TOL（默认 1.5 倍，即质量最多差 50%）
  recommended  ok and ok_quality —— 这才是可以直接交给医院用的那一行

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
    df = pd.read_csv(mp)
    df = df[df["M"].notna()].copy()
    law = json.loads(lp.read_text(encoding="utf-8"))
    name, params = _law_params(law)

    # 用 law 模块自己的预测函数与反解，避免公式重复实现
    import law as L

    rows = []
    for ds, gd in df.groupby("dataset"):
        for N, g in gd.groupby("n"):
            g = g.sort_values("epoch")
            e_full = int(g["epoch"].max())
            last = g.iloc[-1]
            for eps in EPS_GRID:
                e_star = L.solve_E_star(name, params, float(N), eps)
                if e_star != e_star:  # nan：这个 N 永远达不到 eps
                    continue
                at = g[g["epoch"] <= e_star]
                if len(at) == 0:
                    continue
                stop = at.iloc[-1]
                fids = at.dropna(subset=["fid"])
                fid_stop = float(fids["fid"].iloc[-1]) if len(fids) else np.nan
                fid_full = float(g["fid"].dropna().iloc[-1]) if g["fid"].notna().any() else np.nan
                onset = _onset(g, eps)
                ok = bool(stop["M"] <= eps)
                fid_rel = (fid_stop / fid_full) if (pd.notna(fid_stop) and pd.notna(fid_full) and fid_full > 0) else np.nan
                ok_quality = bool(pd.notna(fid_rel) and fid_rel <= Q_TOL)
                rows.append({
                    "dataset": ds, "n": int(N), "eps": eps,
                    "E_star": round(float(e_star), 1),
                    "E_measured": int(stop["epoch"]),
                    "M_stop": float(stop["M"]),
                    "M_full": float(last["M"]),
                    "epoch_saving": 1 - stop["epoch"] / max(e_full, 1),
                    "fid_stop": fid_stop, "fid_full": fid_full,
                    "d_fid": (fid_stop - fid_full) if pd.notna(fid_stop) and pd.notna(fid_full) else np.nan,
                    "ok": ok,
                    "E_onset": onset,
                    "onset_ratio": (float(e_star) / onset) if onset == onset and onset > 0 else np.nan,
                    "fid_rel": fid_rel,
                    "ok_quality": ok_quality,
                    "recommended": bool(ok and ok_quality),
                })

    t = pd.DataFrame(rows).sort_values(["dataset", "eps", "n"])
    out = C.RESULTS / "safe_early_stop.csv"
    t.to_csv(out, index=False)
    print(f"写入 {out}（{len(t)} 行），律 = {name}\n")

    for ds, g in t.groupby("dataset"):
        print(f"== {ds} ==")
        print(f"{'ε':>7}{'N':>6}{'E*':>8}{'实测点':>8}{'M_停':>8}{'M_满':>8}"
              f"{'省%':>7}{'FID停':>8}{'FID满':>8}{'ΔFID':>7}{'发作点':>8}{'E*/发作':>8}{'OK':>4}{'质量':>5}{'可用':>5}")
        for _, r in g.iterrows():
            fs = f"{r.fid_stop:.1f}" if pd.notna(r.fid_stop) else "  -"
            ff = f"{r.fid_full:.1f}" if pd.notna(r.fid_full) else "  -"
            df_ = f"{r.d_fid:+.1f}" if pd.notna(r.d_fid) else "  -"
            on = f"{r.E_onset:.0f}" if pd.notna(r.E_onset) else "  -"
            ra = f"{r.onset_ratio:.2f}" if pd.notna(r.onset_ratio) else "  -"
            print(f"{r.eps:>7.3f}{r.n:>6}{r.E_star:>8.0f}{r.E_measured:>8}"
                  f"{r.M_stop:>8.4f}{r.M_full:>8.4f}{100*r.epoch_saving:>7.0f}"
                  f"{fs:>8}{ff:>8}{df_:>7}{on:>8}{ra:>8}"
                  f"{'Y' if r.ok else 'N':>4}{'Y' if r.ok_quality else 'N':>5}"
                  f"{'Y' if r.recommended else 'N':>5}")
        print()


if __name__ == "__main__":
    main()