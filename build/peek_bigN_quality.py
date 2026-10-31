"""看大 N 档早期高分点的图像质量，判断是"真记忆化"还是"采样塌陷造成的伪高分"。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

d = pd.read_csv("results/metrics_long.csv")
d = d[d.dataset == "pathmnist"]
rows = []
for (n, s), g in d.groupby(["n", "seed"]):
    g = g.sort_values("epoch")
    if n < 1000:
        continue
    imax = g.M.idxmax()
    top = g.loc[imax]
    lo, hi = g.head(2), g.tail(2)
    for tag, r in [("Mmax", top), ("first", lo.iloc[0]), ("last", hi.iloc[-1])]:
        rows.append({
            "n": int(n), "seed": int(s), "which": tag, "epoch": int(r["epoch"]),
            "M": round(float(r["M"]), 4), "tau": round(float(r["tau"]), 4),
            "sim_mean": round(float(r["sim_mean"]), 4), "sim_p95": round(float(r["sim_p95"]), 4),
            "sim_max": round(float(r["sim_max"]), 4),
            "fid": round(float(r["fid"]), 1) if pd.notna(r.get("fid")) else None,
        })
print(pd.DataFrame(rows).to_string(index=False))