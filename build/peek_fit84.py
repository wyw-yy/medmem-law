import json

import numpy as np
import pandas as pd

import config as C

fit = json.loads((C.RESULTS / "law_fit.json").read_text(encoding="utf-8"))
boot = json.loads((C.RESULTS / "law_bootstrap.json").read_text(encoding="utf-8"))
d = pd.read_csv(C.RESULTS / "metrics_long.csv")
best = fit["best"]
f = fit["fits"][best]
print("rows", len(d), "runs", d["run"].nunique(), "points", fit.get("n_obs"))
print("best", best, "params", [round(p, 4) for p in f["params"]])
for k in ("r2_log", "r2_lin", "r2_log_meas", "r2_lin_meas", "n_meas", "aicc",
          "loo_r2_pooled", "censored_compliance"):
    if k in f:
        print(f"  {k} = {f[k]}")
print("--- bootstrap ---")
for k, v in (boot.get("params") or {}).items():
    print(f"  {k}: {v}")
print("--- plateau by N (deep vs equal-step) ---")
d = d[d["encoder"].str.startswith("dinov2")]
d = d[d["dataset"] == "pathmnist"]
for N, g in d.groupby("n"):
    tail = g.groupby("run").apply(lambda x: x.sort_values("epoch").tail(max(3, len(x) // 5))["M"].max())
    print(f"  N={N:5d}  runs={len(tail):2d}  plateau(median)={np.median(tail):.4f}  max={tail.max():.4f}  "
          f"last_epoch={int(g['epoch'].max())}")
print("--- E_star ---")
for k, v in (f.get("E_star") or {}).items():
    print(f"  {k}: {v}")