"""把 tools/plot.py 里的 fig_collapse 换成"逐 run onset"版本（一次性补丁）。"""
from pathlib import Path

p = Path("tools/plot.py")
src = p.read_text(encoding="utf-8")
start = src.index("def fig_collapse(")
end = src.index('_save(fig, "fig_collapse")') + len('_save(fig, "fig_collapse")')

new = '''def fig_collapse(df: pd.DataFrame, law_json: dict):
    """头图（实证版）：记忆化"发作时间"随 N 的移动 —— 每个点 = 一个训练 run。

    为什么用逐 run 的 onset（results/grid_summary.csv）而不是"种子均值曲线的 onset"：
      * 种子均值曲线会把"某个 seed 早早起跳"抹平，得到的 onset 既不代表任何一次训练，
        也不等于 onset 的均值；
      * 工程上要回答的是"我这一次训练什么时候会不安全"，逐 run 才是对的统计单位，
        而且能把 seed 方差直接画出来（同一 N 的点在竖直方向散开）。
    斜率只在**平台 ≥ 5%**的档上拟合：平台只有 1–2% 时，"首越 1%"由 1–3 张图的
    量化噪声决定，会把斜率系统性压低（此前 0.64 就是这么来的，见 docs/NOTES.md §15）。
    """
    gs_path = C.RESULTS / "grid_summary.csv"
    if not gs_path.exists():
        return
    gs = pd.read_csv(gs_path)
    gs = gs[gs["dataset"] == "pathmnist"]

    B = float(C.BATCH)
    series = [("onset_1pct", 0.01), ("onset_5pct", 0.05)]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    law_p = _law_params(law_json)
    slopes = {}

    for i, (col, eps) in enumerate(series):
        colr = COLORS[i % len(COLORS)]
        rows = []
        for N in sorted(int(x) for x in gs["n"].unique()):
            sub = gs[gs["n"] == N]
            ons = sub[col].dropna()
            if len(ons) == 0:
                continue
            plateau = float(sub["M_max"].max())
            steps = ons.to_numpy() * N / B
            rows.append(dict(N=N, k=len(ons), mean=float(steps.mean()),
                             plateau=plateau, steps=steps))
        if not rows:
            continue
        for r in rows:
            cred = r["plateau"] >= CRED_MIN
            ax.plot([r["N"]] * len(r["steps"]), r["steps"], "o", ms=2.2,
                    color=colr, alpha=0.30, mec="none")
            ax.plot([r["N"]], [r["mean"]], "o", ms=6, color=colr,
                    mfc=colr if cred else "white", mew=0.9,
                    label=rf"measured  $\\epsilon$={eps:g}")
            ax.annotate(int(r["N"]), (r["N"], r["mean"]), textcoords="offset points",
                        xytext=(0, 6), fontsize=5.5, ha="center", color=colr)
        cred_rows = [r for r in rows if r["plateau"] >= CRED_MIN]
        if len(cred_rows) >= 2:
            slopes[eps] = float(np.polyfit(np.log([r["N"] for r in cred_rows]),
                                           np.log([r["mean"] for r in cred_rows]), 1)[0])
        # 律的预测（epoch 反解后同样换算到步）
        gx, gy = [], []
        for N in sorted(int(x) for x in gs["n"].unique()):
            e = L.solve_E_star(law_json["best"], law_json["fits"][law_json["best"]]["params"], N, eps)
            if e == e and e is not None:
                gx.append(N)
                gy.append(e * N / B)
        if gx:
            ax.plot(gx, gy, "--", lw=1.0, color=colr, alpha=0.85,
                    label=rf"law      $\\epsilon$={eps:g}")

    Ns = sorted(int(x) for x in gs["n"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_ylabel("optimization steps to reach $\\\\epsilon$")
    if slopes:
        txt = ", ".join(rf"$\\epsilon$={e:g}: {s:.2f}" for e, s in sorted(slopes.items()))
        ax.set_xlabel("dataset size $N$\\n"
                      + "fitted slope vs $N$ (means, plateau $\\\\geq$ 5%):  " + txt,
                      fontsize=8)
    else:
        ax.set_xlabel("dataset size $N$")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    ax.text(0.98, 0.04,
            "small dots = individual runs (seed variance)\\n"
            "hollow mean = plateau < 5% (onset set by 1–3 images, excluded from slope)",
            transform=ax.transAxes, fontsize=5.5, color="0.45", ha="right", va="bottom")
    ax.set_title("Memorization onset grows with data size (in steps)")
    _save(fig, "fig_collapse")'''

src = src[:start] + new + src[end:]
p.write_text(src, encoding="utf-8")
print("patched fig_collapse")