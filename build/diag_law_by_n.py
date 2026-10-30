"""逐档 N 的拟合诊断：哪条律在哪一档 N 上系统偏高/偏低。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
import law as L


def main() -> None:
    df = L.load_metrics()
    lj = json.loads((C.RESULTS / "law_fit.json").read_text(encoding="utf-8"))
    print(f"{len(df)} 点；逐档诊断（只用 M>0 的点算偏差，bias=median(pred/obs)）\n")
    hdr = f"{'N':>5} {'n_obs':>6}"
    names = list(lj["fits"])
    for nm in names:
        hdr += f" | {nm[:12]:>12} {'bias':>6}"
    print(hdr)
    for N in sorted(df["n"].unique()):
        sub = df[(df["n"] == N) & (df["M"] > 0)]
        line = f"{int(N):>5} {len(sub):>6}"
        for nm in names:
            p = lj["fits"][nm]["params"]
            f = L.CANDIDATES[nm]["f"]
            pred = f(p, sub["n"].to_numpy(float), sub["epoch"].to_numpy(float))
            obs = sub["M"].to_numpy(float)
            r2 = 1 - np.sum((np.log10(pred + 1e-4) - np.log10(obs + 1e-4)) ** 2) / \
                np.sum((np.log10(obs + 1e-4) - np.log10(obs + 1e-4).mean()) ** 2)
            bias = float(np.median(pred / obs))
            line += f" | {r2:>12.3f} {bias:>6.2f}"
        print(line)
    print("\n（bias>1 = 模型高估；<1 = 低估）")


if __name__ == "__main__":
    main()