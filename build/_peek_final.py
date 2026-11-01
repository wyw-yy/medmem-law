import json, pandas as pd
d = json.load(open("results/law_fit.json", encoding="utf-8"))
f = d["fits"][d["best"]]
print("best:", d["best"])
print("params:", {k: round(float(v), 4) for k, v in zip(f.get("pnames", d.get("pnames", [])), f["params"])})
for k in ("r2_log", "r2_lin", "r2_log_meas", "r2_lin_meas", "r2_pooled", "censored_compliance", "aicc"):
    if k in f:
        print(k, round(float(f[k]), 4))
s = pd.read_csv("results/stop_policy_summary.csv")
print("cols:", list(s.columns))
print(s.head(10).to_string(index=False))
