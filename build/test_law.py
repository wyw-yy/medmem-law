"""用已知真值的合成数据检验拟合管线：能不能把 gamma 找回来。

这是「先证伪自己的分析代码」的一步：在真数据上下结论之前，先确认
law.fit_one / loo_by_n / solve_E_star 在有噪声、采样点不均匀的情况下
不会给出假的高 R² 或错到离谱的 gamma。

用法：python build/test_law.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C
import law as L

rng = np.random.default_rng(7)

# 真值：L1，gamma=1.0（"按 epoch 计时，起点随 N 线性推迟"），beta=1.6
TRUE = dict(Minf=0.55, beta=1.6, c=4.0, gamma=1.0)   # E50 = 4.0 * N^1.0 → N=250:1000, 4000:16000


def synth(noise: float = 0.15) -> pd.DataFrame:
    rows = []
    for N in C.N_GRID:
        ep_max = C.epochs_for(N)
        every = C.ckpt_every_for(ep_max)
        for ep in range(every, ep_max + 1, every):
            m = L.model_L1([TRUE["Minf"], TRUE["beta"], TRUE["c"], TRUE["gamma"]],
                           np.array([N], float), np.array([ep], float))[0]
            # 观测噪声：乘性对数正态（M 的测量本来就是对 512 张图数比例，小 M 时相对误差大）
            m_obs = float(np.clip(m * np.exp(rng.normal(0, noise)), 0.0, 1.0))
            rows.append(dict(run=f"synth_n{N}_s1", n=N, seed=1, epoch=ep, M=m_obs))
    return pd.DataFrame(rows)


def main():
    df = synth()
    print(f"合成 {len(df)} 个点，N={sorted(df.n.unique())}，"
          f"E 范围 {df.epoch.min()}–{df.epoch.max()}")
    print(f"真值 gamma={TRUE['gamma']}  E50(N)= {TRUE['c']}·N^{TRUE['gamma']}")

    print(f"\n{'律':<20}{'R²_log':>9}{'R²_lin':>9}{'AICc':>10}{'LOO_R²':>9}  params")
    fits = {}
    for name in L.CANDIDATES:
        f = L.fit_one(name, df)
        loo = L.loo_by_n(name, df)
        fits[name] = (f, loo)
        p = np.round(f["params"], 3)
        print(f"{name:<20}{f['r2_log']:>9.4f}{f['r2_lin']:>9.4f}{f['aicc']:>10.1f}"
              f"{(loo['r2_mean'] or float('nan')):>9.4f}  {p}")

    # 关键检查：L1 应该把 gamma 找回到 1.0 附近
    p1 = fits["L1_stretched_exp"][0]["params"]
    got = p1[3]
    print(f"\nL1 复原 gamma = {got:.3f}（真值 1.0，误差 {abs(got-1):.3f}）")
    print(f"L1 复原 Minf  = {p1[0]:.3f}（真值 {TRUE['Minf']}）")
    ok = abs(got - 1.0) < 0.25 and fits["L1_stretched_exp"][0]["r2_log"] > 0.9
    print("判据：|gamma-1|<0.25 且 R²_log>0.9 →", "通过" if ok else "不通过 ⚠️")

    # E* 反解检查：eps=0.01 时，解析解满足 Minf*(1-exp(-(E/E50)^beta))=eps
    eps = 0.05
    e50 = TRUE["c"] * 250.0 ** TRUE["gamma"]
    analytic = e50 * (-np.log(1 - eps / TRUE["Minf"])) ** (1 / TRUE["beta"])
    numeric = L.solve_E_star("L1_stretched_exp", p1, 250.0, eps)
    print(f"E*(N=250, eps={eps})：解析 {analytic:.0f} vs 拟合反解 {numeric:.0f}")


if __name__ == "__main__":
    main()