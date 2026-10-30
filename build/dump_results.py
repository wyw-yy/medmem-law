"""把关键结果一次性打印出来，方便写论文/汇报时核对数字。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
R = ROOT / "results"


def main() -> None:
    j = json.loads((R / "law_fit.json").read_text(encoding="utf-8"))
    print(f"BEST: {j['best']}")
    for name, v in j["fits"].items():
        print(f"  {name:20s} R2log={v['r2_log']:.4f} R2lin={v['r2_lin']:.4f} "
              f"AICc={v['aicc']:8.1f} LOO={v['loo'].get('r2_pooled') or float('nan'):.3f} "
              f"params={[round(x, 4) for x in v['params']]}")
    print("E_star:", json.dumps(j["E_star"], ensure_ascii=False))
    print(f"gamma_hat={j['gamma_hat']:.3f}  uncensored={j.get('gamma_uncensored_fit'):.3f}")

    for fn in ("stop_policy_summary.csv", "transfer.csv", "threshold_law.csv"):
        p = R / fn
        if p.exists():
            print(f"\n===== {fn} =====")
            print(pd.read_csv(p).to_string(index=False))

    p = R / "safe_early_stop.csv"
    if p.exists():
        d = pd.read_csv(p)
        print("\n===== safe_early_stop (pathmnist) =====")
        print(d[d["dataset"] == "pathmnist"].to_string(index=False))


if __name__ == "__main__":
    main()