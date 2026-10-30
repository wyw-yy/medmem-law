"""出图：把 results/ 里的数字变成论文图。

用法：
  python -m tools.plot              # 全部能出的图都出
  python -m tools.plot --only M     # 只出 M-E 曲线

图目录 figs/，命名统一 fig_*.png（DPI 由 config.DPI 控制）。
没有数据就跳过并说明，不报错 —— 这样在任何阶段都能跑。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
import law as L

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["figure.dpi"] = C.DPI
plt.rcParams["savefig.dpi"] = C.DPI
plt.rcParams["font.size"] = 9
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
NOISE_MIN = 0.02   # 低于 2%（≈10/512）视为整档无可测记忆化


def _save(fig, name: str):
    C.FIGS.mkdir(parents=True, exist_ok=True)
    p = C.FIGS / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p.relative_to(C.ROOT)}")


def load(dataset: str | None = "pathmnist") -> pd.DataFrame | None:
    p = C.RESULTS / "metrics_long.csv"
    if not p.exists():
        print(f"[plot] 没有 {p}，先跑 tools.score_run")
        return None
    df = pd.read_csv(p)
    df = df[df["M"].notna()].copy()
    # 必须按数据集过滤：cross 跑完后 bloodmnist/pneumoniamnist 的行与 pathmnist
    # 共用同一张 CSV，不过滤的话"N=250"会把三个数据集平均在一起（曲线全错）。
    if dataset and "dataset" in df.columns:
        df = df[df["dataset"] == dataset].copy()
    df["n"] = df["n"].astype(int)
    df["epoch"] = df["epoch"].astype(int)
    if len(df) < 5:
        print(f"[plot] {dataset} 的测量点太少（{len(df)}）")
        return None
    return df


def fig_m_vs_e(df: pd.DataFrame):
    """主图之一：记忆率随训练轮次上升，N 越大升得越晚。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, N in enumerate(sorted(df["n"].unique())):
        sub = df[df["n"] == N]
        g = sub.groupby("epoch")["M"]
        e = g.mean().index.to_numpy()
        m = g.mean().to_numpy()
        s = g.std().fillna(0.0).to_numpy()
        ax.plot(e, m, color=COLORS[i % len(COLORS)], label=f"N={N}", lw=1.6)
        if (s > 0).any():
            ax.fill_between(e, np.clip(m - s, 0, 1), m + s, color=COLORS[i % len(COLORS)], alpha=0.18)
    ax.set_xlabel("training epoch $E$")
    ax.set_ylabel("memorization rate $M$")
    ax.set_yscale("symlog", linthresh=0.005)
    ax.axhline(3.0 / C.N_SAMPLE, color="0.6", lw=0.7, ls=":")
    # 参考线：单张样本即 1/N_SAMPLE，是这条曲线能取到的最小非零值（离散噪声地板）
    ax.axhline(1.0 / C.N_SAMPLE, ls=":", lw=1.0, color="0.55")
    ax.text(0.99, 1.0 / C.N_SAMPLE, " 1 sample ", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=6, color="0.45")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Memorization rises with training time; larger $N$ delays it")
    _save(fig, "fig_M_vs_E")


def _law_params(law_json: dict) -> dict:
    best = law_json["best"]
    return dict(zip(L.CANDIDATES[best]["pnames"], law_json["fits"][best]["params"]))


def _onset_table(df: pd.DataFrame, eps_list, need_consecutive: int = 2):
    """每个 N 上，记忆率**连续** ≥ ε 的起点轮次。

    要求连续点数：单点越过 ε 很可能是 1–2 张样本的量化噪声（512 样本里 1 张 = 0.002），
    用它当"发作时间"会让大 N 档出现虚假的早发作点。
    另外 N 档的 M 最大值 < NOISE_MIN 时视为"整档无可测记忆化"，返回 None。
    """
    out = {}
    for N in sorted(df["n"].unique()):
        sub = df[df["n"] == N].groupby("epoch")["M"].mean().sort_index()
        vals = sub.to_numpy()
        for eps in eps_list:
            hit = None
            if vals.size and vals.max() >= NOISE_MIN:
                flags = vals >= eps
                for i in range(len(flags) - need_consecutive + 1):
                    if flags[i:i + need_consecutive].all():
                        hit = int(sub.index[i])
                        break
            out[(int(N), eps)] = hit
    return out


def fig_collapse(df: pd.DataFrame, law_json: dict):
    """头图（实证版）：记忆化"发作时间"随 N 的移动。

    为什么不用 M vs E/N^γ 的塌缩图：L1/L2/L3 假设平台高度与 N 无关，
    而实测平台 0.80/0.40/0.07/≤0.01/≤0.02 随 N 陡降，单轴塌缩必然拉开；
    改成"到达 ε 所需轮次 vs N"后，纵轴本身就是要交付的量（安全窗口），
    不需要任何模型假设，再用拟合律叠加成虚线做一致性检验。
    """
    d = _law_params(law_json)
    eps_list = [0.01, 0.05, 0.10, 0.20]
    tab = _onset_table(df, eps_list)
    Ns = sorted(int(x) for x in df["n"].unique())

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    B = float(C.BATCH)
    slopes = {}
    for i, eps in enumerate(eps_list):
        col = COLORS[i % len(COLORS)]
        xs = [N for N in Ns if tab.get((N, eps))]
        ys = [tab[(N, eps)] * N / B for N in xs]          # 换算到优化步
        if len(xs):
            ax.plot(xs, ys, "o", ms=5, color=col, label=rf"measured  $\epsilon$={eps:g}")
            for x, y in zip(xs, ys):
                ax.annotate(int(x), (x, y), textcoords="offset points", xytext=(0, 5),
                            fontsize=5.5, ha="center", color=col)
            if len(xs) >= 2:
                slopes[eps] = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
        # 律的预测（epoch 反解后同样换算到步）
        gx, gy = [], []
        for N in Ns:
            e = L.solve_E_star(law_json["best"], law_json["fits"][law_json["best"]]["params"], N, eps)
            if e == e and e is not None:
                gx.append(N)
                gy.append(e * N / B)
        if gx:
            ax.plot(gx, gy, "--", lw=1.0, color=col, alpha=0.85,
                    label=rf"law      $\epsilon$={eps:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_xlabel("dataset size $N$")
    ax.set_ylabel("optimization steps to reach $\\epsilon$")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    if slopes:
        txt = ", ".join(rf"$\epsilon$={e:g}: {s:.2f}" for e, s in sorted(slopes.items()))
        ax.text(0.03, 0.97, "fitted slope vs $N$:  " + txt, transform=ax.transAxes,
                fontsize=5.8, va="top", color="0.25")
    ax.set_title("Memorization onset grows with data size (in steps)")
    _save(fig, "fig_collapse")


def fig_amplitude(df: pd.DataFrame, law_json: dict):
    """平台高度随 N 的衰减（纯实测，不叠模型线）。"""
    rows = []
    for N in sorted(df["n"].unique()):
        sub = df[df["n"] == N].sort_values("epoch")
        if sub.empty:
            continue
        tail = sub.tail(max(3, len(sub) // 5))
        rows.append((int(N), float(tail["M"].max())))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    Ns = np.array([r[0] for r in rows], float)
    Ms = np.array([r[1] for r in rows], float)
    ax.plot(Ns, Ms, "o-", ms=5, lw=1.2, color=COLORS[0], label="measured")
    floor = 3.0 / C.N_SAMPLE
    ax.axhline(floor, color="0.6", lw=0.7, ls=":")
    ax.text(Ns.min() * 0.9, floor * 1.15, f"detection floor 3/{C.N_SAMPLE}",
            fontsize=6, color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dataset size $N$")
    ax.set_ylabel("plateau memorization rate")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("More data, less memorization")
    _save(fig, "fig_amplitude")


def _fig_collapse_plain(df: pd.DataFrame, gamma: float):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, N in enumerate(sorted(df["n"].unique())):
        sub = df[df["n"] == N].sort_values("epoch")
        g = sub.groupby("epoch")["M"].mean()
        ax.plot(g.index.to_numpy() / N ** gamma, g.to_numpy() + 1e-4,
                color=COLORS[i % len(COLORS)], marker="o", ms=2.5, lw=1.2, label=f"N={N}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E / N^{\gamma}$")
    ax.set_ylabel(r"$M$ (offset $10^{-4}$)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(rf"Data collapse with $\gamma={gamma:.2f}$")
    _save(fig, "fig_collapse")


def fig_parity(df: pd.DataFrame, law_json: dict):
    """拟合优度：预测 vs 实测。"""
    best = law_json["best"]
    params = law_json["fits"][best]["params"]
    f = L.CANDIDATES[best]["f"]
    pred = L._pred(f, params, df["n"].to_numpy(float), df["epoch"].to_numpy(float))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for i, N in enumerate(sorted(df["n"].unique())):
        m = df["n"] == N
        ax.scatter(df.loc[m, "M"], pred[m], s=9, alpha=0.75,
                   color=COLORS[i % len(COLORS)], label=f"N={N}", edgecolor="none")
    lim = [1e-4, max(df["M"].max(), 1.0) * 1.2]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured $M$")
    ax.set_ylabel("predicted $M$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{best}\n$R^2_{{log}}$={law_json['fits'][best]['r2_log']:.3f}, "
                 f"LOO $R^2$={law_json['fits'][best]['loo']['r2_mean']:.3f}")
    _save(fig, "fig_parity")


def fig_quality(df: pd.DataFrame, law_json: dict | None = None):
    """质量-记忆权衡：FID 随 E 下降，M 随 E 上升，两条曲线交叉处即是"想要的停点"。"""
    if "fid" not in df.columns or df["fid"].isna().all():
        print("[plot] 没有 fid 列，跳过 fig_quality")
        return
    Ns = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, N in enumerate(Ns):
        sub = df[(df["n"] == N) & df["fid"].notna()]
        g = sub.groupby("epoch")[["M", "fid"]].mean()
        axes[0].plot(g.index, g["fid"], color=COLORS[i % len(COLORS)], label=f"N={N}", lw=1.4)
        axes[1].plot(g.index, g["M"], color=COLORS[i % len(COLORS)], label=f"N={N}", lw=1.4)
    axes[0].set_xlabel("epoch $E$")
    axes[0].set_ylabel("FID (lower = better quality)")
    axes[0].set_title("sample quality")
    axes[1].set_xlabel("epoch $E$")
    axes[1].set_ylabel("memorization rate $M$")
    axes[1].set_yscale("symlog", linthresh=0.005)
    axes[1].set_title("privacy cost")
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "fig_quality")


def fig_stop_policy():
    p = C.RESULTS / "stop_policy_summary.csv"
    if not p.exists():
        print("[plot] 没有 stop_policy_summary.csv，跳过 fig_stop")
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    x = np.arange(len(d))
    axes[0].bar(x, d["epochs"], color=COLORS[: len(d)])
    axes[0].set_xticks(x, d["policy"], fontsize=8)
    axes[0].set_ylabel("epochs used")
    axes[0].set_title("training cost")
    axes[1].bar(x, d["violation_rate"], color=COLORS[: len(d)])
    axes[1].set_xticks(x, d["policy"], fontsize=8)
    axes[1].set_ylabel("fraction of runs exceeding $\\epsilon$")
    axes[1].set_title("privacy violations")
    _save(fig, "fig_stop_policies")


def fig_e_star(law_json: dict):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    est = law_json["E_star"]
    Ns = sorted(int(k) for k in next(iter(est.values())))
    for i, (eps, d) in enumerate(est.items()):
        ys = [d[str(N)] for N in Ns]
        ax.plot(Ns, ys, marker="o", ms=3, lw=1.4, color=COLORS[i % len(COLORS)], label=f"$\\epsilon$={eps}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dataset size $N$")
    ax.set_ylabel("safe stopping epoch $E^*$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Safe early-stopping schedule")
    _save(fig, "fig_E_star")


def fig_cross_dataset():
    """跨数据集：同一套预算下，不同医学子域的记忆化倾向差多少。

    每个数据集一个面板，横轴 epoch、纵轴 M，按 N 分色。目的是让读者看到
    "函数族相同、参数不同"（bloodmnist 在 N=1000 时记忆率就已经 0.34，
    pathmnist 同预算只有 0.07）。
    """
    p = C.RESULTS / "metrics_long.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["M"].notna()].copy()
    ds_list = [d for d in ("pathmnist", "bloodmnist", "pneumoniamnist")
               if (df["dataset"] == d).any()]
    if len(ds_list) < 2:
        print("  [skip] fig_cross_dataset：还没有第二个数据集的结果")
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.2 * len(ds_list), 3.0), sharey=True)
    if len(ds_list) == 1:
        axes = [axes]
    # 只画所有数据集都有的 N 档 —— 跨数据集比较必须"同 N 同预算"才成立，
    # 否则 pathmnist 的 2000/4000（全是检出限以下的噪声）会把面板画花。
    common_n = sorted(set.intersection(*[set(df[df["dataset"] == d]["n"].unique()) for d in ds_list]))
    for ax, ds in zip(axes, ds_list):
        sub = df[df["dataset"] == ds]
        for i, N in enumerate(common_n):
            g = sub[sub["n"] == N].groupby("epoch")["M"].mean()
            # 纵轴用 symlog（不是 log）：M=0 的点要在 0 处画成平的，
            # 用 log(M+1e-4) 会让"0 与非 0 交替"变成假的锯齿振荡。
            ax.plot(g.index.to_numpy(), g.to_numpy(), color=COLORS[i % len(COLORS)],
                    lw=1.5, label=f"N={N}")
        ax.set_yscale("symlog", linthresh=0.005)
        ax.axhline(3.0 / C.N_SAMPLE, color="0.6", ls=":", lw=0.8)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("epoch $E$")
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("memorization rate $M$")
    fig.suptitle("Same budget, different memorization propensity", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_cross_dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--dataset", default="pathmnist",
                    help="画哪一份数据集的主图（cross 数据集单独出 fig_cross_dataset）")
    a = ap.parse_args()
    only = a.only.lower()

    df = load(a.dataset)
    lawjson = None
    lp = C.RESULTS / "law_fit.json"
    if lp.exists():
        lawjson = json.loads(lp.read_text(encoding="utf-8"))

    if df is None:
        if lawjson:
            fig_e_star(lawjson)
        return

    if not only or only in "m,me":
        fig_m_vs_e(df)
    if lawjson:
        if not only or only in "all,collapse,c":
            fig_collapse(df, lawjson)
        if not only or only in "all,amp,a":
            fig_amplitude(df, lawjson)
        if not only or only in "all,parity,p":
            fig_parity(df, lawjson)
        if not only or only in "all,estar,e":
            fig_e_star(lawjson)
    if not only or only in "all,quality,q":
        fig_quality(df)
    if not only or only in "all,stop,s":
        fig_stop_policy()
    if not only or only in "all,cross,x":
        fig_cross_dataset()


if __name__ == "__main__":
    main()