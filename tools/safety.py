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

⚠ fid 只在部分检查点测过（--fid-stride），所以 fid_stop 取「≤ E* 的最近一个有点」。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

EPS_GRID = (0.005, 0.01, 0.02, 0.05)


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
                rows.append({
                    "dataset": ds, "n": int(N), "eps": eps,
                    "E_star": round(float(e_star), 1),
                    "E_measured": int(stop["epoch"]),
                    "M_stop": float(stop["M"]),
                    "M_full": float(last["M"]),
                    "epoch_saving": 1 - stop["epoch"] / max(e_full, 1),
                    "fid_stop": fid_stop, "fid_full": fid_full,
                    "d_fid": (fid_stop - fid_full) if pd.notna(fid_stop) and pd.notna(fid_full) else np.nan,
                    "ok": bool(stop["M"] <= eps),
                })

    t = pd.DataFrame(rows).sort_values(["dataset", "eps", "n"])
    out = C.RESULTS / "safe_early_stop.csv"
    t.to_csv(out, index=False)
    print(f"写入 {out}（{len(t)} 行），律 = {name}\n")

    for ds, g in t.groupby("dataset"):
        print(f"== {ds} ==")
        print(f"{'ε':>7}{'N':>6}{'E*':>8}{'实测点':>8}{'M_停':>8}{'M_满':>8}"
              f"{'省%':>7}{'FID停':>8}{'FID满':>8}{'ΔFID':>7}{'OK':>4}")
        for _, r in g.iterrows():
            fs = f"{r.fid_stop:.1f}" if pd.notna(r.fid_stop) else "  -"
            ff = f"{r.fid_full:.1f}" if pd.notna(r.fid_full) else "  -"
            df_ = f"{r.d_fid:+.1f}" if pd.notna(r.d_fid) else "  -"
            print(f"{r.eps:>7.3f}{r.n:>6}{r.E_star:>8.0f}{r.E_measured:>8}"
                  f"{r.M_stop:>8.4f}{r.M_full:>8.4f}{100*r.epoch_saving:>7.0f}"
                  f"{fs:>8}{ff:>8}{df_:>7}{'Y' if r.ok else 'N':>4}")
        print()


if __name__ == "__main__":
    main()