"""28px pilot 与 64px 主线的对照表（附录 A）。

为什么要这张表：主线是 64px（从 224px 源 LANCZOS 下采样），但第一轮实验是在 28px 上做的
（HF 上的 `albertvillanova/medmnist-v2` 只有 28px）。审稿人会问"换分辨率会不会改结论"，
这张表就是回答。

同一个 run 形状（N、seed 一致）下比较：
  τ（阈值，随分辨率变是正常的）、末点 M、跑满轮次、达成 ε 所需步数（onset）

输出：`results/pilot_vs_main.csv` 与打印表。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402

EPS = 0.01


def onset_steps(g: pd.DataFrame, eps: float = EPS) -> float:
    gg = g.groupby("epoch")["M"].mean().sort_index()
    ok = gg[gg >= eps]
    if len(ok) == 0:
        return np.nan
    return float(ok.index[0]) * float(g["n"].iloc[0]) / C.BATCH


def summarize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for (ds, n), g in df.groupby(["dataset", "n"]):
        last = g.sort_values("epoch").iloc[-1]
        rows.append(dict(
            res=tag, dataset=ds, n=int(n), runs=g["run"].nunique(),
            tau=round(float(g["tau"].mean()), 4),
            epochs=int(last["epoch"]),
            steps=int(last["epoch"] * n / C.BATCH),
            M_last=round(float(g.sort_values("epoch").groupby("run")["M"].last().mean()), 4),
            M_max=round(float(g.groupby("run")["M"].max().mean()), 4),
            onset_steps=onset_steps(g),
            fid_last=round(float(last["fid"]), 1) if pd.notna(last["fid"]) else np.nan))
    return pd.DataFrame(rows)


def main() -> int:
    p28 = pd.read_csv(C.RESULTS / "pilot28" / "metrics_long.csv")
    d64 = pd.read_csv(C.RESULTS / "metrics_long.csv")
    a = summarize(p28, "28px")
    b = summarize(d64[d64["dataset"] == "pathmnist"], "64px")
    t = pd.concat([a, b], ignore_index=True).sort_values(["n", "res"])
    print(t.to_string(index=False))

    wide = t.pivot_table(index="n", columns="res",
                         values=["tau", "M_last", "M_max", "epochs", "onset_steps", "fid_last"])
    print("\n== 同一 N 的 28px vs 64px（epochs 不同是因为预算不同，比较看 onset_steps 与 M）==")
    print(wide.to_string())
    t.to_csv(C.RESULTS / "pilot_vs_main.csv", index=False, encoding="utf-8")
    print("\n写出 results/pilot_vs_main.csv")

    # 结论性数字：分辨率对 τ 与 M 的影响
    j = wide
    if ("tau", "28px") in j and ("tau", "64px") in j:
        print("\nτ 变化（64px − 28px）：",
              {int(n): round(float(j.loc[n, ("tau", "64px")] - j.loc[n, ("tau", "28px")]), 3)
               for n in j.index})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())