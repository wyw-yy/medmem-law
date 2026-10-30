"""跨数据集迁移：把 pathmnist 拟合出的律形状搬到别的数据集上，只重估两个尺度参数。

为什么这样设计：`night_queue --plan cross` 每个数据集只有 N∈{250,1000} 两档，
5 个参数不可能同时辨识（α、γ 需要至少 3 档 N）。所以诚实的迁移检验是：

  固定 α（幅度衰减）、β（上升陡度）、γ（时间尺度平移）来自 pathmnist，
  只在该数据集上重估 A0 与 E50_0（两个尺度），看 R² 是多少。

- R² 高 → 该数据集的记忆化形状与 pathmnist 一致，只是"整体更容易/更难背"。
- R² 低 → 形状本身不可迁移（这才是需要写进 limitation 的话）。

对照：把 α/β/γ 也放开（自由拟合，5 参），作为"该数据集自身上限"。

用法：python -m tools.transfer
输出：results/transfer.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import config as C
import law as L


def _fit_fixed(df: pd.DataFrame, base: list[float], free_idx: list[int],
               bounds: list[tuple[float, float]]) -> tuple[np.ndarray, dict]:
    """只优化 free_idx 指定的参数，其余固定为 base。"""
    N = df["n"].to_numpy(float)
    E = df["epoch"].to_numpy(float)
    M = df["M"].to_numpy(float)
    mdet = L._mdet(df)
    f = L.model_L4
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    def unpack(q):
        p = np.array(base, float)
        p[free_idx] = q
        return p

    def resid(q):
        return L._resid(f, unpack(q), N, E, M, mdet)

    q0 = np.array([base[i] for i in free_idx], float)
    q0 = np.clip(q0, lo * 1.001 + 1e-9, hi * 0.999)
    r = least_squares(resid, q0, bounds=(lo, hi), max_nfev=4000, method="trf")
    p = unpack(r.x)
    pred = L._pred(f, p, N, E)
    obs = M > 0
    if obs.sum() >= 3:
        r2_log = 1 - np.sum((np.log10(pred[obs] + 1e-4) - np.log10(M[obs] + 1e-4)) ** 2) / \
            np.sum((np.log10(M[obs] + 1e-4) - np.log10(M[obs] + 1e-4).mean()) ** 2)
        r2_lin = 1 - np.sum((pred[obs] - M[obs]) ** 2) / np.sum((M[obs] - M[obs].mean()) ** 2)
    else:
        r2_log = r2_lin = float("nan")
    cens_ok = float(np.mean(pred[~obs] <= mdet[~obs])) if (~obs).any() else float("nan")
    return p, dict(r2_log=float(r2_log), r2_lin=float(r2_lin),
                   censored_compliance=cens_ok, n_obs=int(obs.sum()), n=int(len(df)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(C.RESULTS / "law_fit.json"))
    args = ap.parse_args()

    lj = json.loads(Path(args.base).read_text(encoding="utf-8"))
    if lj["best"] != "L4_amp_decay":
        raise SystemExit(f"迁移检验是为 L4 写的，当前选定 {lj['best']}")
    base = list(lj["fits"]["L4_amp_decay"]["params"])
    names = L.CANDIDATES["L4_amp_decay"]["pnames"]
    print(f"基准（pathmnist）：" + ", ".join(f"{k}={v:.4g}" for k, v in zip(names, base)))

    df = L.load_metrics()
    rows = []
    for ds, sub in df.groupby("dataset"):
        if len(sub["n"].unique()) < 2:
            print(f"[skip] {ds}: 只有 {len(sub['n'].unique())} 档 N，做不了迁移检验")
            continue
        # A 迁移：只放开 A0 与 E50_0
        free = [names.index("A0_at250"), names.index("E50_at250")]
        bounds = [(1e-3, 1.0), (1e-2, 1e5)]
        p_t, s_t = _fit_fixed(sub, base, free, bounds)
        # B 上限：五个参数全放开
        free_b = list(range(5))
        spec = L.CANDIDATES["L4_amp_decay"]
        bounds_b = list(zip(spec["lo"], spec["hi"]))
        p_b, s_b = _fit_fixed(sub, base, free_b, bounds_b)
        for mode, p, s in (("transfer", p_t, s_t), ("free_fit", p_b, s_b)):
            rows.append(dict(dataset=ds, mode=mode,
                             **{k: float(v) for k, v in zip(names, p)},
                             **s))
            print(f"  {ds:14s} {mode:9s} R²_log={s['r2_log']:.3f} R²_lin={s['r2_lin']:.3f} "
                  f"删失合规={s['censored_compliance']:.2f} "
                  f"A0={p[0]:.3f} E50={p[3]:.0f} α={p[1]:.2f} γ={p[4]:+.2f} N档={sorted(sub['n'].unique())}")

    if rows:
        out = C.RESULTS / "transfer.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
        print(f"\n写入 {out}")
    else:
        print("没有可用的跨数据集数据（需要至少 2 档 N 的数据集）")


if __name__ == "__main__":
    main()