"""R22 结果汇总：DP-SGD（σ=1/8）、容量（base=64）、224px 平台。

读三组独立 CSV，与主网格的同 (dataset, N, seed) 基线在**相同 epoch** 上对照，
输出 `results/r22_summary.csv` 与一张终端表。

对照规则：
  * DP σ=1/8  vs  `pathmnist_n250_s1`（同子集、同 seed、base=32、64px、非隐私）
  * base=64    vs  `pathmnist_n250_s1` / `pathmnist_n1000_s1`
  * 224px      vs  `bloodmnist_n250_s21_64`（同子集同 seed 的 64px 版）
比较点选在基线有测量点的 epoch 上（基线是等步数预算，点比较密）。

用法：
    python -m build.summary_r22
    python -m build.summary_r22 --latex   # 同时输出 paper 用的表体
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "metrics_long.csv"
OUT = ROOT / "results" / "r22_summary.csv"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


MCOL = "M"   # score_run 的列名是 M（曾叫 rate，兼容两者）


def _mcol(df: pd.DataFrame) -> str:
    return "M" if "M" in df.columns else "rate"


def at_epoch(df: pd.DataFrame, e: int, col: str = "") -> float:
    col = col or _mcol(df)
    """取最接近 e 的测量点（只往回找，避免用到未来信息）。"""
    cand = df[df["epoch"] <= e]
    if cand.empty:
        return float("nan")
    return float(cand.iloc[-1][col])


def firsts(df: pd.DataFrame) -> tuple[float, float, float]:
    """(首个 M>0 的 epoch, 首个 M>=1% 的 epoch, M_max)"""
    mc = _mcol(df)
    pos = df[df[mc] > 0]
    e1 = float(pos.iloc[0]["epoch"]) if len(pos) else float("nan")
    m1 = df[df[mc] >= 0.01]
    e2 = float(m1.iloc[0]["epoch"]) if len(m1) else float("nan")
    return e1, e2, float(df[mc].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    main_df = pd.read_csv(MAIN)
    rows = []

    def base(tag: str) -> pd.DataFrame:
        d = main_df[main_df["run"] == tag].sort_values("epoch")
        return d.reset_index(drop=True)

    specs = [
        ("A_dp_sigma1.0", "results/dp_s1_n250_s1.csv", "pathmnist_n250_s1",
         {"sigma": 1.0, "eps_ub_steps4800": 85.1}),
        ("B_dp_sigma8.0", "results/dp_s8_n250_s1.csv", "pathmnist_n250_s1",
         {"sigma": 8.0, "eps_ub_steps4800": 10.6}),
        ("C_base64_N250", "results/b64_pathmnist_n250_s1.csv", "pathmnist_n250_s1", {}),
        ("D_base64_N1000", "results/b64_pathmnist_n1000_s1.csv", "pathmnist_n1000_s1", {}),
        ("E_224px_N250", "results/pilot224_accum_blood_n250.csv", "bloodmnist_n250_s21_64", {}),
    ]

    for name, rel, ref, meta in specs:
        p = ROOT / rel
        if not p.exists():
            print(f"· {name}: 缺 {rel}（还没跑完？）")
            continue
        d = load(p)
        b = base(ref) if ref in set(main_df["run"]) else pd.DataFrame()
        e1, e2, mmax = firsts(d)
        # 共同的比较点：参照 run 有测量点、且 ≤ 本 run 最大 epoch 的那些
        pts = []
        if len(b):
            for e in b["epoch"].tolist():
                if e > d["epoch"].max():
                    continue
                pts.append((int(e), at_epoch(d, int(e)), at_epoch(b, int(e))))
        ratio = [(e, x, y) for e, x, y in pts if y > 0 and np.isfinite(x)]
        rec = {
            "run": name, "csv": rel, "ref": ref,
            "points": len(d), "epochs_max": int(d["epoch"].max()),
            "tau": round(float(d["tau"].iloc[0]), 4),
            "M_max": round(mmax, 4),
            "sim_max_max": round(float(d["sim_max"].max()), 4),
            "first_nonzero_epoch": e1, "first_1pct_epoch": e2,
            "fid_first": round(float(d["fid"].dropna().iloc[0]), 1) if d["fid"].notna().any() else np.nan,
            "fid_last": round(float(d["fid"].dropna().iloc[-1]), 1) if d["fid"].notna().any() else np.nan,
        }
        rec.update(meta)
        for e, x, y in pts:
            rec[f"M@{e}"] = round(x, 4)
            rec[f"ref@{e}"] = round(y, 4)
        rows.append(rec)

        print(f"\n=== {name} ===")
        print(f"  points={rec['points']}  epochs<= {rec['epochs_max']}  tau={rec['tau']}  "
              f"M_max={rec['M_max']}  sim_max={rec['sim_max_max']}")
        print(f"  首个非零 epoch={e1}  首个>=1% epoch={e2}  "
              f"FID {rec['fid_first']} → {rec['fid_last']}")
        if pts:
            print("  epoch   本 run   参照     比")
            for e, x, y in pts:
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                r = (f"{x / y:.2f}x" if y > 0 else "—")
                print(f"  {e:6d}  {x:7.4f}  {y:7.4f}   {r}")
        if ratio:
            worst = max(ratio, key=lambda t: t[1])
            print(f"  最大比值 {worst[1]/worst[2]:.2f}x @ epoch {worst[0]}")

    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
        print(f"\n写成 {OUT.relative_to(ROOT)}（{len(rows)} 行）")
    if a.latex:
        print("\n% ---- LaTeX 表体 ----")
        for r in rows:
            print(f"{r['run'].replace('_', r'\_')} & {r['tau']:.3f} & {r['M_max']:.3f} & "
                  f"{r['sim_max_max']:.3f} & {r['first_1pct_epoch']} & "
                  f"{r['fid_first']}/{r['fid_last']} \\\\")


if __name__ == "__main__":
    main()