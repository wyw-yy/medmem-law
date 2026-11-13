"""把 R22/R23 的所有对照实验汇总成一张表（只读 CSV，不占 GPU）。

这些实验**不进主律拟合**（训练规则与主网格不同：DP-SGD 加噪、改容量、改分辨率），
所以单独出一张对照表，供 NOTES §20 / PROTOCOL §24.10 / 论文附录引用。

用法：
    python -m build.summary_r23 [--out results/r23_summary.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# (组名, 标签, 说明)。标签对应 results/<tag>.csv；文件不存在就跳过。
GROUPS: list[tuple[str, str, str]] = [
    ("DP-噪声剂量", "dp_s1_n250_s1", "σ=1.0，1200 轮（ε_ub≈85）"),
    ("DP-噪声剂量", "dp_s2_n250_s1", "σ=2.0，1200 轮"),
    ("DP-噪声剂量", "dp_s4_n250_s1", "σ=4.0，1200 轮"),
    ("DP-噪声剂量", "dp_s8_n250_s1", "σ=8.0，1200 轮（ε_ub≈10.6）"),
    ("DP-噪声剂量", "dp_s8_n250_s1_long", "σ=8.0，5125 轮"),
    ("DP-噪声剂量", "dp_s2_n250_s1_long", "σ=2.0，5125 轮"),
    ("DP-噪声剂量", "dp_s4_n250_s1_long", "σ=4.0，5125 轮"),
    ("DP-噪声剂量", "dp_s1_n250_s1_long", "σ=1.0，5125 轮"),
    ("DP-裁剪范数", "dp_s8_c01_n250_s1", "σ=8，C=0.1"),
    ("DP-裁剪范数", "dp_s8_c05_n250_s1", "σ=8，C=0.5"),
    ("DP-裁剪范数", "dp_s8_c20_n250_s1", "σ=8，C=2.0"),
    ("容量", "b48_pathmnist_n250_s1", "base=48，N=250"),
    ("容量", "b64_pathmnist_n250_s1", "base=64，N=250"),
    ("容量", "b48_pathmnist_n1000_s1", "base=48，N=1000"),
    ("容量", "b64_pathmnist_n1000_s1", "base=64，N=1000"),
    ("分辨率", "pilot64_blood_n250_s21", "64px，N=250，seed21"),
    ("分辨率", "pilot64_blood_n250_s22", "64px，N=250，seed22"),
    ("分辨率", "pilot128_blood_n250", "128px，N=250，seed21"),
    ("分辨率", "pilot128_blood_n250_s22", "128px，N=250，seed22"),
    ("分辨率", "pilot128_blood_n1000", "128px，N=1000，seed21"),
    ("分辨率", "pilot224_blood_n250", "224px，N=250，batch16"),
    ("分辨率", "pilot224_accum_blood_n250", "224px，N=250，batch16×accum4"),
    ("分辨率", "pilot224_blood_n1000_a4", "224px，N=1000，batch16×accum4"),
    ("分辨率", "pilot224_blood_n1000", "224px，N=1000，batch16（旧）"),
]


def _first_at(df: pd.DataFrame, thr: float) -> float | None:
    hit = df[df["M"] >= thr]
    return float(hit["epoch"].iloc[0]) if len(hit) else None


def _tau(tag: str) -> float | None:
    """该 run 自己的 τ（逐 run 校准），从 runs/<tag>.json 里取。"""
    p = RES / "runs" / f"{tag}.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for key in ("tau", "thr", "tau_fpr"):
        if key in j:
            return float(j[key])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/r23_summary.csv")
    a = ap.parse_args()

    rows = []
    for group, tag, note in GROUPS:
        p = RES / f"{tag}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "M" not in df.columns or not len(df):
            continue
        df = df.sort_values("epoch")
        fid = df["fid"].dropna() if "fid" in df.columns else pd.Series(dtype=float)
        sim = df["sim_max"] if "sim_max" in df.columns else pd.Series(dtype=float)
        rows.append({
            "group": group,
            "run": tag,
            "note": note,
            "n_points": len(df),
            "epoch_last": int(df["epoch"].iloc[-1]),
            "M_max": round(float(df["M"].max()), 4),
            "M_last": round(float(df["M"].iloc[-1]), 4),
            "epoch_first_1pct": _first_at(df, 0.01),
            "epoch_first_5pct": _first_at(df, 0.05),
            "sim_max_max": round(float(sim.max()), 4) if len(sim) else None,
            "sim_max_last": round(float(sim.iloc[-1]), 4) if len(sim) else None,
            "fid_first": round(float(fid.iloc[0]), 1) if len(fid) else None,
            "fid_last": round(float(fid.iloc[-1]), 1) if len(fid) else None,
            "tau": _tau(tag),
        })

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"写成 {a.out}：{len(rows)} 组")
    for r in rows:
        print(f"  {r['group']:8s} {r['run']:32s} M_max={r['M_max']:.4f} "
              f"sim_max={r['sim_max_max']} fid={r['fid_first']}->{r['fid_last']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())