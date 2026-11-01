"""B10 结果对照：同一批生成样本，两个编码器各算一遍 M(E)。

读 `results/encoder_resnet50.csv`（B10）与主 CSV（DINOv2），按 run 对齐 epoch，
输出：
  results/encoder_compare.csv  逐点对照（run, epoch, M_dino, M_rn50, 差）
  results/encoder_summary.csv  逐 run 汇总（tau、末点、平台、曲线相关、符号一致性）

判据（写进报告）：
  1. 两条曲线在同一 run 上的 Spearman 相关 ≥ 0.9 → 上升趋势一致；
  2. 末点 M 的相对差 ≤ 20% 或绝对差 ≤ 0.05 → 量级一致；
  3. 平台随 N 的排序完全一致（250 > 500 > 1000 > 1500）→ 定性结论不改。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

MAIN = "dinov2_s:384"


def main() -> int:
    main_df = pd.read_csv(C.RESULTS / "metrics_long.csv")
    main_df = main_df[main_df["encoder"] == MAIN]
    alt = pd.read_csv(C.RESULTS / "encoder_resnet50.csv")

    rows, summ = [], []
    for run in sorted(alt["run"].unique()):
        a = alt[alt["run"] == run].set_index("epoch")
        b = main_df[main_df["run"] == run].set_index("epoch")
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 3:
            print(f"跳过 {run}（共同点只有 {len(common)} 个）")
            continue
        for ep in common:
            rows.append(dict(run=run, epoch=int(ep), n=int(a.loc[ep, "n"]),
                             M_dino=float(b.loc[ep, "M"]), M_rn50=float(a.loc[ep, "M"]),
                             diff=float(a.loc[ep, "M"] - b.loc[ep, "M"])))

        m_d = b.loc[common, "M"].to_numpy()
        m_r = a.loc[common, "M"].to_numpy()
        # 平台 = 尾部 20% 的最大值
        k = max(1, len(m_d) // 5)
        rho = stats.spearmanr(m_d, m_r).statistic if len(set(m_d)) > 1 else np.nan
        summ.append(dict(
            run=run, n=int(a["n"].iloc[0]), points=len(common),
            tau_dino=float(b["tau"].iloc[0]), tau_rn50=float(a["tau"].iloc[0]),
            M_last_dino=float(m_d[-1]), M_last_rn50=float(m_r[-1]),
            plat_dino=float(m_d[-k:].max()), plat_rn50=float(m_r[-k:].max()),
            spearman=float(rho) if rho == rho else np.nan,
            mean_abs_diff=float(np.abs(m_r - m_d).mean()),
            max_abs_diff=float(np.abs(m_r - m_d).max()),
        ))

    cmp_df = pd.DataFrame(rows)
    sm = pd.DataFrame(summ).sort_values("n")
    cmp_df.to_csv(C.RESULTS / "encoder_compare.csv", index=False, encoding="utf-8")
    sm.to_csv(C.RESULTS / "encoder_summary.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 160)
    print("逐 run 汇总：")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 三条判据
    ok1 = (sm["spearman"] >= 0.9).all()
    ok2 = ((sm["M_last_rn50"] - sm["M_last_dino"]).abs() <= 0.05).all()
    order_d = sm.sort_values("n")["plat_dino"].is_monotonic_decreasing
    order_r = sm.sort_values("n")["plat_rn50"].is_monotonic_decreasing
    ok3 = bool(order_d and order_r)
    print(f"\n判据1 趋势相关≥0.9：{'通过' if ok1 else '未通过'}  "
          f"(最小 rho={sm['spearman'].min():.3f})")
    print(f"判据2 末点绝对差≤0.05：{'通过' if ok2 else '未通过'}  "
          f"(最大差 {(sm['M_last_rn50'] - sm['M_last_dino']).abs().max():.3f})")
    print(f"判据3 平台随 N 单调下降（两种编码器都成立）：{'通过' if ok3 else '未通过'}")
    print(f"\n写出 {C.RESULTS / 'encoder_compare.csv'}（{len(cmp_df)} 行）与 encoder_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())