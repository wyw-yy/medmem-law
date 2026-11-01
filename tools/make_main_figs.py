"""主图三联（论文 Fig.1）：一张 3 面板图，全部从 CSV 现算，不手写数字。

面板
  (a) 记忆率随训练轮次上升；N 越大升得越晚（pathmnist，7 档 N，seed 均值 ± 范围）
  (b) 平台高度随 N 的衰减（幂律斜率 -1.97）
  (c) 四条停训策略的"省了多少轮" vs "违规率"（越靠右下越好）

用法：
  python -m tools.make_main_figs
输出：
  figs/fig1_triptych.png / .pdf
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

import config as C

RESULTS = C.RESULTS
FIGS = C.FIGS

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f"]


def _style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    })


def panel_a(ax, df: pd.DataFrame):
    sub = df[(df["dataset"] == "pathmnist") & (df["encoder"].str.startswith("dinov2"))]
    Ns = sorted(sub["n"].unique())
    floor = 3.0 / C.N_SAMPLE
    eps = floor / 3.0          # 画在 log 轴上时给 0 一个落点，不伪造数值
    for i, N in enumerate(Ns):
        s = sub[sub["n"] == N]
        g = s.groupby("epoch")["M"]
        e = g.mean().index.to_numpy(float)
        m = np.maximum(g.mean().to_numpy(), eps)
        lo = np.maximum(g.min().to_numpy(), eps)
        hi = np.maximum(g.max().to_numpy(), eps)
        dense = N in (1500, 3000)   # 加密档用虚线，避免与主 5 档抢视觉
        ax.plot(e, m, color=COLORS[i % len(COLORS)], lw=1.0 if dense else 1.5,
                ls="--" if dense else "-", label=f"$N$={N}", alpha=0.85 if dense else 1.0)
        if not dense:
            ax.fill_between(e, lo, hi, color=COLORS[i % len(COLORS)], alpha=0.10, lw=0)
    ax.axhline(floor, color="0.55", lw=0.7, ls=":")
    # 2% 以下按"量化噪声区"处理并标灰：512 张样本里 1–3 张越线就是 0.002–0.006，
    # 这条带里的起伏不能读成"记忆化起跳"（否则 N≥1500 的抖动会被误读，见 docs/NOTES.md §11.4）。
    ax.axhspan(eps, 2.0 * floor, color="0.5", alpha=0.06, lw=0)
    ax.annotate("grey band: below 2% = 1–3 of 512 images\n(quantisation, not an onset)",
                xy=(0.015, 0.965), xycoords="axes fraction", xytext=(0, 0),
                textcoords="offset points", ha="left", va="top",
                fontsize=6, color="0.42")
    ax.annotate(f"detection floor $3/{C.N_SAMPLE}$", xy=(0.985, floor),
                xycoords=ax.get_yaxis_transform(), xytext=(0, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6, color="0.4")
    ax.set_xlabel("training epoch $E$")
    ax.set_ylabel("memorization rate $M$")
    ax.set_yscale("log")
    ax.set_ylim(eps * 0.8, 1.3)
    ax.legend(ncol=2, loc="lower right", handlelength=1.6,
              columnspacing=0.9, labelspacing=0.22, fontsize=6.5,
              frameon=True, facecolor="white", framealpha=0.88, edgecolor="none")
    ax.set_title("(a) memorization rises with training time;\nlarger $N$ delays it")


def panel_b(ax, df: pd.DataFrame, law: dict):
    sub = df[(df["dataset"] == "pathmnist") & (df["encoder"].str.startswith("dinov2"))]
    floor = 3.0 / C.N_SAMPLE
    rows = []
    for N in sorted(sub["n"].unique()):
        s = sub[sub["n"] == N]
        # 平台 = 曲线尾部 20% 里的最大值（再对 seed 取中位）
        per_seed = []
        for _sd, g in s.groupby("seed"):
            g = g.sort_values("epoch")
            tail = g.tail(max(3, len(g) // 5))
            per_seed.append(float(tail["M"].max()))
        rows.append((int(N), float(np.median(per_seed))))
    Ns = np.array([r[0] for r in rows], float)
    Ms = np.array([r[1] for r in rows], float)
    good = Ms > 2.0 * floor            # 高于两倍检出限才算"测到了平台"
    ax.plot(Ns[good], Ms[good], "o-", ms=5, lw=1.3, color=COLORS[3],
            label="measured plateau")
    if (~good).any():
        ax.plot(Ns[~good], np.maximum(Ms[~good], floor / 3.0), "o", ms=5, mfc="none",
                mec=COLORS[3], mew=1.1, label="below detection")
    slope = float("nan")
    if good.sum() >= 2:
        k = np.polyfit(np.log(Ns[good]), np.log(Ms[good]), 1)
        slope = float(k[0])
        xx = np.geomspace(Ns.min() * 0.85, Ns.max() * 1.2, 60)
        ax.plot(xx, np.exp(np.polyval(k, np.log(xx))), "--", lw=1.0, color="0.35",
                label=f"power law, slope {k[0]:+.2f}")
    ax.axhline(floor, color="0.55", lw=0.7, ls=":")
    ax.annotate(f"floor $3/{C.N_SAMPLE}$", xy=(0.98, floor),
                xycoords=ax.get_yaxis_transform(), xytext=(-1, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6, color="0.4")
    a0 = law.get("fits", {}).get(law.get("best", ""), {}).get("params")
    if a0:
        ax.text(0.03, 0.055, f"law amplitude $\\alpha$ = {a0[1]:.2f}",
                transform=ax.transAxes, fontsize=6.5, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dataset size $N$")
    ax.set_ylabel("plateau $M$")
    ax.legend(frameon=False, loc="upper right", labelspacing=0.25)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"(b) plateau decays faster than $1/N$" if not np.isnan(slope)
                 else "(b) plateau vs dataset size")


def panel_c(ax):
    p = RESULTS / "stop_policy_summary.csv"
    if not p.exists():
        ax.text(0.5, 0.5, "no stop_policy_summary.csv", ha="center", va="center",
                transform=ax.transAxes)
        return
    d = pd.read_csv(p)
    label = {
        "A_fixed_max": "A run to max",
        "B_fid_plateau": "B FID plateau",
        "C_open_loop": "C open-loop law",
        "D_closed_loop": "D closed-loop $M$",
        "Dp_closed_predict": "D′ D + predicted residue",
    }
    order = ["A_fixed_max", "B_fid_plateau", "C_open_loop", "Dp_closed_predict", "D_closed_loop"]
    # D 与 D′ 数值完全相同（会重叠），标签一左一右错开
    offset = {"Dp_closed_predict": (-10, 14, "right"),
              "D_closed_loop": (8, -6, "left")}
    d = d.set_index("policy")
    for i, k in enumerate(order):
        if k not in d.index:
            continue
        row = d.loc[k]
        viol = 100.0 * float(row["violation_rate"])
        save = 100.0 * float(row["saving"])
        c = COLORS[3] if viol == 0 else COLORS[i % len(COLORS)]
        ax.scatter(save, viol, s=46, color=c, zorder=3, edgecolor="white", lw=0.6)
        dx, dy, ha = offset.get(k, (5, 5, "left"))
        ax.annotate(label.get(k, k), (save, viol), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.5, color=c, ha=ha)
    ax.set_xlim(-6, 100)
    ax.set_ylim(-8, 100)
    ax.set_xlabel("training epochs saved (%)")
    ax.set_ylabel("privacy violations (%)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("(c) only the closed loop is both\nfast and compliant")


def main():
    _style()
    m = RESULTS / "metrics_long.csv"
    df = pd.read_csv(m)
    law = {}
    lp = RESULTS / "law_fit.json"
    if lp.exists():
        law = json.loads(lp.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), layout="constrained")
    panel_a(axes[0], df)
    panel_b(axes[1], df, law)
    panel_c(axes[2])
    out = FIGS / "fig1_triptych.png"
    fig.savefig(out)
    fig.savefig(FIGS / "fig1_triptych.pdf")
    print(f"写出 {out}")
    print(f"写出 {FIGS / 'fig1_triptych.pdf'}")


if __name__ == "__main__":
    main()