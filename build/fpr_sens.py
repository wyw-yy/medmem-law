"""FPR 敏感性：把 τ 的校准目标从 0.5% 换成 0.1% / 1%，看 M 会不会翻结论。

为什么要做：τ 是这套度量的唯一自由参数，按"非成员参照图的最大余弦"的 99.5 分位
校准（FPR=0.5%）。审稿人一定会问"换个 FPR 结论还成立吗"。本脚本不重训、不重采样，
只复用最后检查点的采样结果，对同一批特征重新校准 τ 再算 M → 结论是否稳健一目了然。

输出：results/fpr_sensitivity.csv（run,n,epoch,fpr,tau,M,n_gen,M_floor）

用法：
    python -m build.fpr_sens
    python -m build.fpr_sens --runs pathmnist_n250_s1 pathmnist_n1000_s1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
import data as D  # noqa: E402
import metric as MT  # noqa: E402

FPR_GRID = (0.001, 0.005, 0.01)


def latest_sample(run: str) -> Path | None:
    d = C.CACHE / "samples" / run
    files = sorted(d.glob("e*.npy")) if d.exists() else []
    return files[-1] if files else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="默认取主网格 seed 1 的 N=250/500/1000")
    a = ap.parse_args()

    runs = a.runs or [f"{a.dataset}_n{n}_s1" for n in (250, 500, 1000)]
    enc = MT.load_encoder()
    ref_u8 = D.reference_set(a.dataset)
    rows: list[dict] = []

    for run in runs:
        p = latest_sample(run)
        if p is None:
            print(f"[{run}] 没有采样文件，跳过")
            continue
        parts = run.split("_")
        n = int([x for x in parts if x.startswith("n") and x[1:].isdigit()][0][1:])
        seed = int([x for x in parts if x.startswith("s") and x[1:].isdigit()][0][1:])

        gen_u8 = np.load(p)
        train_u8 = D.train_subset(a.dataset, n, seed)
        f_train = MT.encode(train_u8, enc)
        mu, W = MT.fit_whitening(f_train)
        bank = MT.apply_whitening(f_train, mu, W)
        f_gen = MT.apply_whitening(MT.encode(gen_u8, enc), mu, W)
        f_ref = MT.apply_whitening(MT.encode(ref_u8, enc), mu, W)

        sims_ref = MT.max_cos_sim(f_ref, bank)
        for fpr in FPR_GRID:
            cal = MT.calibrate_tau(sims_ref, fpr=fpr, leak_cos=C.LEAK_COS)
            res = MT.memorization_rate(f_gen, bank, cal["tau"])
            rows.append(dict(run=run, n=n, epoch=int(p.stem[1:]), fpr=fpr,
                             tau=cal["tau"], M=res["M"], n_gen=res["n_gen"],
                             M_floor=3.0 / res["n_gen"]))
            print(f"[{run}] e{int(p.stem[1:]):04d} FPR={fpr:.3f} τ={cal['tau']:.4f} "
                  f"M={res['M']:.4f} ({int(round(res['M'] * res['n_gen']))}/{res['n_gen']})")

    if not rows:
        raise SystemExit("没有可算的 run")
    out = C.RESULTS / "fpr_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n→ {out}（{len(rows)} 行）")


if __name__ == "__main__":
    main()