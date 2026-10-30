import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("TOP:", list(d.keys()))
print("gamma_hat:", d.get("gamma_hat"))
f = d["fits"]
print("fits type:", type(f).__name__)
if isinstance(f, dict):
    for k, v in f.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in v.items() if kk in ("params", "pnames", "r2_log", "r2_lin", "aicc")})
else:
    print(len(f))
    print(json.dumps(f[0], ensure_ascii=False)[:500])