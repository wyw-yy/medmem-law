"""打印每个 run 的非零点，用来判断大 N 档的 M 是噪声还是真实记忆化。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

d = pd.read_csv("results/metrics_long.csv")
want = sys.argv[1] if len(sys.argv) > 1 else "pathmnist"
d = d[d.dataset == want]
for (n, s), g in d.groupby(["n", "seed"]):
    g = g.sort_values("epoch")
    nz = g[g.M > 0]
    imax = g.M.idxmax()
    print(f"N={n} seed={s}: {len(g)} pts, nonzero={len(nz)}, "
          f"Mmax={g.M.max():.4f}@e{int(g.loc[imax, 'epoch'])}, Mlast={g.M.iloc[-1]:.4f}")
    if len(nz):
        print("   " + " ".join(f"{int(r.epoch)}:{r.M:.3f}" for r in nz.itertuples()))