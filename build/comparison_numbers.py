"""为 COMPARISON.md 取数：把"生成质量起点与 N 无关"这类提法变成可核对的数字。

取三样东西：
1. 每档 N 的**第一个** FID（训练最早期的生成质量）——检验 τ_gen（质量起点）是否与 N 无关；
2. 每档 N 达成 ε 所需步数（onset）及其对数斜率——对应文献的 τ_mem ∝ n；
3. epoch 坐标的 γ 与步数坐标 γ+1 的换算，用于把我们的律和"步数线性"对齐。
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


def main() -> int:
    d = pd.read_csv(C.RESULTS / "metrics_long.csv")
    d = d[d["dataset"] == "pathmnist"]
    fit = json.load(open(C.RESULTS / "law_fit.json", encoding="utf-8"))
    boot = json.load(open(C.RESULTS / "law_bootstrap.json", encoding="utf-8"))

    print("== 每档 N 的首个 FID（检验质量起点是否与 N 无关）==")
    rows = []
    for n, g in d.groupby("n"):
        f = g.dropna(subset=["fid"]).sort_values("epoch")
        if f.empty:
            continue
        f0, fl = f.iloc[0], f.iloc[-1]
        rows.append(dict(n=int(n), e_first=int(f0["epoch"]), fid_first=round(float(f0["fid"]), 1),
                         steps_first=int(f0["epoch"] * n / C.BATCH),
                         e_last=int(fl["epoch"]), fid_last=round(float(fl["fid"]), 1),
                         drop=round(float(f0["fid"] - fl["fid"]), 1)))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print(f"\n首个 FID 的极差 = {t['fid_first'].max() - t['fid_first'].min():.1f}"
          f"（相对 {t['fid_first'].mean():.1f} 的 {100 * (t['fid_first'].max() - t['fid_first'].min()) / t['fid_first'].mean():.1f}%）")
    t.to_csv(C.RESULTS / "fid_onset_vs_n.csv", index=False, encoding="utf-8")

    print("\n== onset 步数 vs N 的斜率（对应 τ_mem ∝ n）==")
    for eps in (0.01, 0.02, 0.05, 0.10, 0.20):
        pts = []
        for n, g in d.groupby("n"):
            gg = g.groupby("epoch")["M"].mean().sort_index()
            ok = gg[gg >= eps]
            if len(ok) == 0:
                continue
            e = int(ok.index[0])
            pts.append((int(n), e * n / C.BATCH))
        if len(pts) >= 3:
            lx = np.log10([p[0] for p in pts])
            ly = np.log10([p[1] for p in pts])
            slope = float(np.polyfit(lx, ly, 1)[0])
            print(f"  ε={eps:<5} 档数={len(pts)}  onset 步数斜率(步数坐标) = {slope:+.2f}"
                  f"   → epoch 坐标 γ = {slope - 1:+.2f}")
    print("\n律给出的 γ（epoch 坐标）= "
          f"{boot['ci']['gamma']['point']:+.3f} "
          f"[{boot['ci']['gamma']['lo']:+.3f}, {boot['ci']['gamma']['hi']:+.3f}]"
          f" → 步数坐标 = {boot['ci']['gamma']['point'] + 1:+.3f}")
    print("平台幅度指数 α =", round(float(fit["fits"][fit["best"]]["params"][1]), 3),
          "（A(N) ∝ N^-α，文献没有这一项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())