import json

import config as C

fit = json.loads((C.RESULTS / "law_fit.json").read_text(encoding="utf-8"))
best = fit["best"]
f = fit["fits"][best]
print("BEST", best)
for k, v in f.items():
    if isinstance(v, (int, float, str)):
        print(f"  {k} = {v}")
    elif isinstance(v, (list, dict)) and k in ("params", "pnames"):
        print(f"  {k} = {v}")
print("loo keys:", [k for k in f if "loo" in k.lower()])
lo = f.get("loo") or {}
for k, v in (lo.items() if isinstance(lo, dict) else []):
    print("  loo", k, v)
print("censoring:", fit.get("censoring"))
p = C.RESULTS / "law_fit_perseed.json"
if p.exists():
    pf = json.loads(p.read_text(encoding="utf-8"))
    b2 = pf["best"]
    g = pf["fits"][b2]
    print("PERSEED", b2, [round(x, 4) for x in g["params"]])
    for k in ("r2_log", "r2_lin", "loo_r2_pooled", "censored_compliance"):
        if k in g:
            print(f"   {k} = {g[k]}")