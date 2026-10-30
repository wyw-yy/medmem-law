"""阈值稳健性：用不同固定阈值下的记忆率重拟合 L4，看结论（γ、α、R²）是否被阈值选择绑架。

用法：python build\probe_threshold_law.py
输出：results\threshold_law.csv + 屏幕表格
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config as C
import law as L

COLS = ["M", "M_uniq", "M_050", "M_060", "M_070", "M_080", "M_090", "M_095"]
NAME = "L4_amp_decay"


def main() -> None:
    raw = L.load_metrics()
    print(f"读入 {len(raw)} 行，N={sorted(int(x) for x in raw['n'].unique())}")
    rows = []
    for col in COLS:
        if col not in raw.columns:
            continue
        df = raw.copy()
        df["M"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        sig = df.groupby("n")["M"].max()
        n_sig = int((sig > 0).sum())
        if n_sig < 2:
            print(f"{col:10s} 只有 {n_sig} 档有信号，跳过")
            continue
        try:
            fit = L.fit_one(NAME, df)
            loo = L.loo_by_n(NAME, df)
        except Exception as e:  # noqa: BLE001
            print(f"{col:10s} 拟合失败：{type(e).__name__}: {e}")
            continue
        p = dict(zip(L.CANDIDATES[NAME]["pnames"], fit["params"]))
        gamma_steps = p["gamma"] + 1.0          # E50_epoch ∝ N^γ  =>  t50_steps ∝ N^(γ+1)
        row = dict(metric=col, n_sig=n_sig, r2_log=fit["r2_log"], r2_lin=fit["r2_lin"],
                   aicc=fit["aicc"], loo_pooled=loo["r2_pooled"],
                   cens_ok=fit["censored_compliance"],
                   A0=p["A0_at250"], alpha=p["alpha"], beta=p["beta_h"],
                   E50_250=p["E50_at250"], gamma_epoch=p["gamma"], gamma_steps=gamma_steps)
        rows.append(row)
        print(f"{col:10s} σ档={n_sig} R²_log={fit['r2_log']:.4f} R²_lin={fit['r2_lin']:.4f} "
              f"LOO池化={loo['r2_pooled']:.3f} α={p['alpha']:.2f} γ_epoch={p['gamma']:+.2f} "
              f"γ_steps={gamma_steps:+.2f} 删失合规={fit['censored_compliance']:.2f}")

    if rows:
        out = C.RESULTS / "threshold_law.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
        print(f"\n写入 {out}")


if __name__ == "__main__":
    main()