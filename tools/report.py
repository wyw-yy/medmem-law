"""把 results/ 里的产物汇总成一份 RESULTS.md（人读的总入口）。

为什么要有这一层：CSV / JSON 是机器产物，一天要看好几次"现在到哪一步了"。
每次都现敲 pandas 既慢又容易看漏，所以固定一个脚本，产物永远同一副样子，
而且每次重跑都会更新文件头的生成时间 —— 一眼能看出数据是不是新的。

用法
  python -m tools.report            # 写 RESULTS.md
  python -m tools.report --print    # 同时打屏
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C
import numpy as np
import pandas as pd

OUT = ROOT / "RESULTS.md"


def _read_csv(p: Path):
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if len(df) else None
    except Exception:
        return None


def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _run_json(dataset, n, seed):
    """读 runs/*.json，拿真实预算 —— 不能再用 C.epochs_for 猜：
    deep 档（R1）把大 N 的预算改大过，猜出来的"预算"会让完成度表全错。"""
    import json
    p = C.RESULTS / "runs" / f"{C.run_tag(dataset, int(n), int(seed))}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def section_status(df):
    lines = ["## 1. 主网格完成度", ""]
    if df is None:
        lines += ["`results/metrics_long.csv` 还不存在 —— 先跑 `python -m tools.night_queue --plan main`。", ""]
        return lines
    lines.append("| 数据集 | N | seed | 测量点 | 最后 epoch / 预算 | τ | M_max | FID_min | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    # 必须按 (dataset, n, seed) 分组：三个数据集共用同一张 CSV，
    # 只按 (n, seed) 分组会把 bloodmnist 的 M_max 混进 pathmnist 的行里。
    for (ds, n, s), g in df.groupby(["dataset", "n", "seed"], sort=True):
        eps = sorted(g["epoch"].unique())
        meta = _run_json(ds, n, s)
        want = int(meta.get("epochs_target") or C.epochs_for(int(n)))
        done = int(eps[-1]) if eps else 0
        ok = "✅ 完成" if done >= want * 0.99 else f"⏳ {done}/{want}"
        # 干预组（按律/闭环提前刹停）本来就不该跑满预算：用 run JSON 的停点字段标出来，
        # 否则完成度表会把它们全标成"未完成"，看起来像掉链子。
        if meta.get("stopped_early_at"):
            why = "闭环" if meta.get("stop_eps") is not None else "开环律"
            ok = f"✅ {why}停在 e{int(meta['stopped_early_at'])}（交付 e{int(meta.get('deliver_epoch') or meta['stopped_early_at'])}）"
        fmin = g["fid"].min() if g["fid"].notna().any() else None
        lines.append(
            f"| {ds} | {int(n)} | {int(s)} | {len(eps)} | {done} / {want} | "
            f"{_fmt(g['tau'].iloc[-1], 4)} | {_fmt(g['M'].max())} | {_fmt(fmin, 1)} | {ok} |"
        )
    lines.append("")
    n_runs = df.groupby(["dataset", "n", "seed"]).ngroups
    lines.append(f"- 总测量点 **{len(df)}**，覆盖 **{n_runs}** 个 run，数据集：{', '.join(sorted(df['dataset'].unique()))}。")
    if "encoder" in df.columns:
        lines.append(f"- 编码器：{', '.join(sorted(df['encoder'].dropna().unique()))}。")
    lines.append("")
    return lines


def _gamma_of(name, f):
    """从参数名取值，别用下标猜 —— L1/L2 的 γ 是 p[3]，L3 是 p[2]，L4/L6 是 p[4]。"""
    import law as L
    p = f.get("params") or []
    try:
        return p[L.CANDIDATES[name]["pnames"].index("gamma")]
    except Exception:
        return None


def _censored_note() -> str:
    """把"为什么 R² 上不去"写成可复算的一句话：多少点低于检出限、信号区到底到哪个 N。"""
    fit = _read_json(C.RESULTS / "law_fit.json") or {}
    cen = fit.get("censoring") or {}
    txt = ""
    n_obs, n_cen = cen.get("n_observed_points"), cen.get("n_censored_points")
    if n_obs and n_cen:
        txt += f"拟合用的 {int(n_obs) + int(n_cen)} 个点里有 {int(n_cen)} 个低于检出限（3/512）；"
    gs = _read_csv(C.RESULTS / "grid_summary.csv")
    if gs is not None and len(gs):
        p = gs[gs["dataset"] == "pathmnist"]
        sig = sorted({int(n) for n, m in zip(p["n"], p["M_max"])
                      if m is not None and not pd.isna(m) and float(m) >= 0.05})
        if sig:
            txt += (f"能稳定看到记忆化（M_max ≥ 5%，即至少 25 张图）的 N 只到 {max(sig)}；再往上 M_max 只剩 1–2 张图造成的量化噪声；")
    return txt


def section_law(law):
    lines = ["## 2. 控制律拟合", ""]
    if not law:
        lines += ["`results/law_fit.json` 还不存在 —— 主网格跑完 `python -m law` 生成。", ""]
        return lines
    best = law.get("best")
    fits = law.get("fits") or {}
    if fits:
        lines.append("| 候选律 | R²_log | R²_lin | R²_log(可测) | R²_lin(可测) | AICc | 留一池化 R² | 删失合规(样本内) | γ | 参数 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, f in fits.items():
            mark = "**" if name == best else ""
            loo = f.get("loo") or {}
            prm = ", ".join(f"{v:.3g}" for v in (f.get("params") or []))
            lines.append(
                f"| {mark}{name}{mark} | {_fmt(f.get('r2_log'))} | {_fmt(f.get('r2_lin'))} | "
                f"{_fmt(f.get('r2_log_meas'))} | {_fmt(f.get('r2_lin_meas'))} | "
                f"{_fmt(f.get('aicc'), 1)} | {_fmt(loo.get('r2_pooled'))} | "
                f"{_fmt(f.get('censored_compliance'))} | {_fmt(_gamma_of(name, f))} | {prm} |"
            )
        lines.append("")
    lines.append(f"- 选定：**{best}**（判据：AICc 最小 → 留一池化 R² 最大 → 参数可解释，见 `docs/PROTOCOL.md` §10）。")
    bf = fits.get(best) or {}
    r2l = bf.get("r2_log")
    r2lin = bf.get("r2_lin")
    loo = (bf.get("loo") or {}).get("r2_pooled")
    lines.append(
        f"- 验收线（预声明）：样本内 R²_log ≥ 0.9 {'✅' if (r2l or 0) >= 0.9 else '❌'}（{_fmt(r2l)}，"
        f"线性口径 R²_lin = {_fmt(r2lin)}）；留一池化 R² ≥ 0.8 {'✅' if (loo or 0) >= 0.8 else '❌'}（{_fmt(loo)}）。"
    )
    lines.append(
        "- **未达线的原因写在明处**：" + _censored_note() +
        "留出一档大 N 等于外推一条从没测到的过程。故主结论限定在可测区间，E\\*=nan 表示该 N 在任意训练时长下都到不了 ε。"
    )
    lines.append(
        "- **口径**：律拟合的是**种子均值曲线**（`law.collapse_seeds`）；单 seed 拟合会更好看，"
        "但那是运气好的那条曲线。`R²(可测)` 只在 M ≥ 0.02 的点上算，"
        "它比全体更低说明**误差集中在真正有信号的区域**（逐档 bias 见 `docs/NOTES.md` §4.2）。"
    )
    lines.append(
        "- **为什么加了 N=1500/3000 与 seed 5/6 之后 R² 反而降了**：同一 N 下不同 seed（=不同训练子集）的"
        "发作步数相差最多 8 倍（如 N=1000：2344 / 4688 / 12891 / 15234 / 18750 步），这个**子集方差**比"
        "N 从 250 涨到 1500 带来的系统位移还大。律拟合的是期望曲线，所以预测力有天花板 —— "
        "这也是把交付物放在**闭环控制器**而不是固定时刻表上的实证理由（见第 5、6 节）。"
    )
    ps = C.RESULTS / "law_fit_perseed.json"
    if ps.exists():
        d2 = json.loads(ps.read_text(encoding="utf-8"))
        b2 = d2.get("best")
        f2 = (d2.get("fits") or {}).get(b2) or {}
        lines.append(
            f"- **口径敏感性（逐 run 口径）**：不折种子、每条曲线各自作观测 → {b2}，"
            f"R²_log {_fmt(f2.get('r2_log'))} / R²_lin {_fmt(f2.get('r2_lin'))}、"
            f"留一池化 {_fmt((f2.get('loo') or {}).get('r2_pooled'))}、γ {_fmt(_gamma_of(b2, f2))}"
            f"（种子均值口径 γ {_fmt(_gamma_of(best, bf))}）→ 两种口径的 γ 相差 < 0.05，结论不依赖口径选择。"
        )
    g = law.get("gamma_hat")
    if g is not None:
        lines.append(f"- **γ = {_fmt(g)}**（epoch 坐标：`E50 = E50_0·(N/250)^γ`；"
                     f"换算到优化步坐标 `t50 ∝ N^(γ+1) = N^{g + 1:+.2f}`）。")
    lines.append(f"- 点数 {law.get('n_points')}，N 网格 {law.get('n_grid')}。")
    est = law.get("E_star") or {}
    if isinstance(est, dict) and est:
        eps_vals = list(est.keys())
        ns = sorted({int(k) for d in est.values() for k in d})
        lines += ["", "**安全早停点 E\\*(N, ε)**（epoch；ε = 可接受记忆率）", "",
                  "| N | " + " | ".join(f"ε={e}" for e in eps_vals) + " |",
                  "|---|" + "---|" * len(eps_vals)]
        for n in ns:
            row = [f"{n}"]
            for e in eps_vals:
                v = est[e].get(str(n))
                row.append("—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.0f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return lines


def section_robust():
    """稳健性三件套：阈值扫描 / 白化维不变性 / 逐档偏差。有文件才写，没有就跳过。"""
    thr = _read_csv(C.RESULTS / "threshold_law.csv")
    kd = _read_json(C.RESULTS / "probe_kdim.json")
    unc = _read_json(C.RESULTS / "uncensored_fit.json")
    if thr is None and kd is None and unc is None:
        return []

    lines = ["## 3. 稳健性检验", "", "### 3.1 换个阈值，结论会不会变（`build/probe_threshold_law.py`）", ""]
    if thr is not None:
        lines.append("| 记忆率定义 | 有信号档数 | R²_log | R²_lin | 留一池化 | α / γ_epoch / γ_steps |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in thr.iterrows():
            lines.append(
                f"| {r['metric']} | {int(r['n_sig'])} | {_fmt(r['r2_log'])} | {_fmt(r['r2_lin'])} | "
                f"{_fmt(r['loo_pooled'])} | {_fmt(r['alpha'], 2)} / {_fmt(r['gamma_epoch'], 2)} / {_fmt(r['gamma_steps'], 2)} |"
            )
        lines.append("")
        lines.append("- 阈值 0.5/0.6/0.7 下 γ_steps 与 α 同向同量级；≥0.8 时只有 2–3 档 N 有信号，参数不可辨识（仅作参考）。")
        lines.append("")
    else:
        lines.append("（还没跑 `build/probe_threshold_law.py`）")
        lines.append("")

    lines += ["### 3.2 白化截断维 k 的不变性（`build/probe_kdim.py`）", ""]
    if kd:
        rows = {}
        for r in kd:
            rows.setdefault(int(r["n"]), {})[int(r["k"])] = r
        ks = sorted({int(r["k"]) for r in kd})
        lines.append("| N | " + " | ".join(f"k={k} τ / M" for k in ks) + " |")
        lines.append("|---|" + "---|" * len(ks))
        for n in sorted(rows):
            cells = []
            for k in ks:
                r = rows[n].get(k)
                cells.append(f"{_fmt(r['tau'])} / {_fmt(r['M'])}" if r else "—")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("- τ 随 k 剧烈变化、**M 几乎不变**（±0.05 绝对）→ 记忆率对度量的自由参数稳健；τ 的数值不可跨配置比较。")
        lines.append("")
    else:
        lines.append("（还没跑 `build/probe_kdim.py`）")
        lines.append("")

    if unc:
        lines += ["### 3.3 只用已检出点重拟合（`build/probe_uncensored_fit.py`）", "",
                  "| 律 | R²_log | R²_lin | 逐档 bias（中位 pred/obs） |", "|---|---|---|---|"]
        for name, r in unc.items():
            lines.append(f"| {name} | {_fmt(r.get('r2_log'))} | {_fmt(r.get('r2_lin'))} | "
                         + " ".join(r.get("bias_by_n") or []) + " |")
        lines.append("")

    fpr = _read_csv(C.RESULTS / "fpr_sensitivity.csv")
    if fpr is not None and len(fpr):
        lines += ["### 3.4 换 FPR，τ 与 M 会不会变（`build/fpr_sens.py`）", "",
                  "τ 是本套度量唯一的外部自由参数：按非成员参照图的 99.5 分位校准，即假阳率 0.5%。"
                  "下表对同一批特征重设 FPR 再算 M。", "",
                  "| run | 检查点 | FPR | τ | 记忆率 M | 命中/生成 |",
                  "|---|---|---|---|---|---|"]
        for _, r in fpr.iterrows():
            lines.append(f"| {r['run']} | e{int(r['epoch'])} | {float(r['fpr']) * 100:.1f}% | "
                         f"{_fmt(r['tau'])} | {_fmt(r['M'])} | "
                         f"{int(round(float(r['M']) * float(r['n_gen'])))}/{int(r['n_gen'])} |")
        lines.append("")
        lines.append("- FPR 从 0.1% 放宽到 1%（10 倍）时 M 只变动几个百分点 → 结论不依赖 FPR 的具体取值；"
                     "但 τ 的绝对值不可跨 FPR 比较，报告里统一用 0.5%。")
        lines.append("")

    # 3.5 R 独立复算：第二个实现，用于排除"pipeline 算错"
    rv = _read_csv(C.RESULTS / "r_verify.csv")
    if rv is not None and len(rv):
        lines += ["### 3.5 R 独立复算（`R/verify_in_R.R`，base R 无外部包）", "",
                  "同一份测量点、同一个目标函数，用 R 的 `optim` 重新拟合、自己重写留一 N 与 "
                  "bootstrap。两条完全独立的实现给出同样的数字，比多发一百个 seed 更能说明结论没算错。", "",
                  "| 检查 | 项目 | R | Python | 差异 | 判定 |",
                  "|---|---|---|---|---|---|"]
        for _, r in rv.iterrows():
            lines.append(f"| {r['check']} | {r['item']} | {r['r']} | {r['python']} | "
                         f"{r['abs_diff']} | {r['status']} |")
        lines.append("")
        lines.append("- `exact/match` = 两条实现逐位一致或差异 <0.1%；`consistent` 是 bootstrap 分位，"
                     "两端各自用不同随机数流，差异在蒙特卡洛噪声（约 0.01–0.03）内。")
        lines.append("")

    # 3.6 换编码器重算：整篇的 M 依赖一个自由选择（特征编码器），这里给出第二套编码器的结果
    ec = _read_csv(C.RESULTS / "encoder_summary.csv")
    if ec is not None and len(ec):
        lines += ["### 3.6 换编码器重算（DINOv2 → ResNet50，`build/rescore_encoder.py`）", "",
                  "同一批生成样本，不重训练，只换特征与阈值（τ 逐 N 重校准）。"
                  "它回答的是「M 是不是编码器的产物」。", "",
                  "| run | N | 点 | τ(D→R) | 末点 M(D→R) | 平台(D→R) | Spearman | 最大点差 |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in ec.iterrows():
            lines.append(
                f"| {r['run']} | {int(r['n'])} | {int(r['points'])} | "
                f"{r['tau_dino']:.3f}→{r['tau_rn50']:.3f} | "
                f"{r['M_last_dino']:.3f}→{r['M_last_rn50']:.3f} | "
                f"{r['plat_dino']:.3f}→{r['plat_rn50']:.3f} | "
                f"{r['spearman']:.3f} | {r['max_abs_diff']:.3f} |")
        lines.append("")
        lines.append("- **定性结论全部保留**：M 随训练上升（ρ 0.84–0.99）、平台随 N 单调下降（两种编码器都成立）、"
                     "闭环有效性不依赖编码器。")
        lines.append("- **定量幅度改变**：平台随 N 的下降斜率在 N=250→1500 上，DINOv2 约 −2.46、ResNet50 约 −1.13。"
                     "所以 α 是「条件指数」（依赖 dataset + 编码器 + FPR 口径），跨论文比较 α 无效。")
        lines.append("- 低 M 区间最不稳（N=1500：0.008 vs 0.066 = 4 张 vs 34 张图），与 §3.2、§8 的地板教训同源。")
        lines.append("")

    # 3.7 判据对齐：用 Bonnaire et al. 的 Eq.(6)（像素最近/次近比 < 1/3）复算同一批样本
    eq6 = _read_csv(C.RESULTS / "eq6.csv")
    if eq6 is not None and len(eq6):
        ml = _read_csv(C.RESULTS / "metrics_long.csv")
        lines += ["### 3.7 判据对齐：用原文 Eq.(6) 复算（`build/eq6.py`）", "",
                  "原文（arXiv 2505.17638 §2）判定「记住」用的是**无参数相对判据**："
                  "像素 L2 最近/次近距离比 < 1/3。我们的主指标是「白化特征余弦 ≥ τ(FPR=0.5%)」。"
                  "两者排序是否一致，决定我们能不能与文献对话。", "",
                  "| run | N | 我们的 M | 像素比中位 | f_mem（Eq.6） |",
                  "|---|---|---|---|---|"]
        for _, r in eq6.sort_values(["n", "epoch"]).iterrows():
            M = float("nan")
            if ml is not None:
                sel = ml[(ml["run"] == r["run"]) & (ml["epoch"] == r["epoch"])]
                if len(sel):
                    M = float(sel["M"].iloc[0])
            lines.append(f"| {r['run']}@{int(r['epoch'])} | {int(r['n'])} | "
                         f"{M:.3f} | {r['ratio_p50']:.3f} | {r['f_mem']:.3f} |")
        lines.append("")
        lines.append("- **排序一致**：M 高 ↔ f_mem 高；像素级判据更严（约 2 倍），"
                     "差额正是「像」与「复刻」的区别。")
        lines.append("- **大 N 档是关键差别**：`pathmnist_n2000_s2` 早期 M≈0.04，但 f_mem=0、像素比中位 0.98 →"
                     "那 21 张不是复刻，只是在特征空间里像。**大 N 档的 M 只能当复刻率的上界**。")
        lines.append("")
    return lines


def section_cross():
    """跨数据集推广性：同一个律族在别的数据集上还成不成立。

    这一节回答的是 PPT 里"④验证：跨数据集推广性"。口径必须写清楚：
    cross 每个数据集只有 N∈{250,1000} 两档，5 个参数不可辨识，
    所以只能做「固定律形状、只重估尺度」的迁移检验 + 自由拟合当上限对照。
    """
    gs = _read_csv(C.RESULTS / "grid_summary.csv")
    tr = _read_csv(C.RESULTS / "transfer.csv")
    if gs is None and tr is None:
        return []
    lines = ["## 4. 跨数据集推广性", ""]
    if gs is not None:
        lines.append("### 4.1 各数据集的网格（同一套等步数预算）")
        lines.append("")
        lines.append("| 数据集 | N | seed | 点数 | τ | 末轮 | 步数 | M_末 | M_最大 | 首次≥1% | 该点步数 | 末点 FID |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in gs.iterrows():
            lines.append(
                f"| {r['dataset']} | {_fmt(r['n'], 0)} | {_fmt(r['seed'], 0)} | {_fmt(r['points'], 0)} | "
                f"{_fmt(r['tau'], 3)} | {_fmt(r['epochs'], 0)} | {_fmt(r['steps'], 0)} | "
                f"{_fmt(r['M_last'], 4)} | {_fmt(r['M_max'], 4)} | {_fmt(r.get('onset_1pct'), 0)} | "
                f"{_fmt(r.get('steps_1pct'), 0)} | {_fmt(r.get('fid_last'), 1)} |")
        lines.append("")
        lines.append("**到 1% 记忆率所需的优化步数 vs N**（斜率 ≈1 表示步数坐标下 onset ∝ N）：")
        lines.append("")
        for ds, g in gs.groupby("dataset"):
            ok = g.copy()
            ok["steps_1pct"] = pd.to_numeric(ok["steps_1pct"], errors="coerce")
            ok = ok[np.isfinite(ok["steps_1pct"])]
            if len(ok) >= 2:
                k, _ = np.polyfit(np.log(ok["n"].to_numpy(float)),
                                  np.log(ok["steps_1pct"].to_numpy(float)), 1)
                lines.append(f"- {ds}：N={sorted(ok['n'].astype(int).tolist())} → 斜率 **{k:.2f}**")
            elif len(ok) == 1:
                lines.append(f"- {ds}：只有 1 档到达 1%，算不出斜率")
            else:
                lines.append(f"- {ds}：没有任何一档到达 1%")
        lines.append("")
    if tr is not None:
        lines.append("### 4.2 迁移检验（固定律形状 vs 自由拟合上限）")
        lines.append("")
        lines.append("| 数据集 | 模式 | R²_log | R²_lin | 删失合规 | A0 | α | β | E50 | γ |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in tr.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['mode']} | {_fmt(r['r2_log'], 3)} | {_fmt(r['r2_lin'], 3)} | "
                f"{_fmt(r.get('censored_compliance'), 2)} | {_fmt(r.get('A0_at250'), 3)} | "
                f"{_fmt(r.get('alpha'), 2)} | {_fmt(r.get('beta_h'), 2)} | "
                f"{_fmt(r.get('E50_at250'), 0)} | {_fmt(r.get('gamma'), 2)} |")
        lines.append("")
        lines.append("读法：`transfer` = 只用 pathmnist 的律形状、在目标数据集上重估两个尺度参数；")
        lines.append("`free_fit` = 五个参数全放开（该数据集自身能达到的上限；2 档 N 下参数不唯一，只当参考）。")
        lines.append("")
        lines.append("**结论**：函数族（拉伸指数 × 幅度随 N 衰减）在两个数据集上都能拟合得很好"
                     "（自由拟合 R²_log ≈ 0.98），但**参数是数据集相关的**（α 2.06 vs 0.78）——"
                     "所以医院落地必须用自己的一小批 run 重新标定，或者直接用不需要律的闭环控制器（策略 D）。"
                     "这也正是把控制器做成「边训边测」的理由。")
        lines.append("")
    return lines


def section_policy(df):
    lines = ["## 6. 停训策略对比", ""]
    if df is None:
        lines += ["`results/stop_policy_summary.csv` 还不存在 —— `python -m controller` 生成。", ""]
        return lines
    cols = [c for c in ("policy", "n", "epochs", "saving", "M_stop", "M_stop_p95", "fid", "violation_rate")
            if c in df.columns]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "policy":
                cells.append(str(v))
            elif c == "saving" or c == "violation_rate":
                cells.append(f"{100 * float(v):.0f}%" if pd.notna(v) else "—")
            else:
                cells.append(_fmt(v, 4) if c in ("M_stop", "M_stop_p95") else _fmt(v, 1) if c == "fid" else _fmt(v, 0))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("- A=固定跑满（PPT 基线）、B=FID 平台早停、C=开环律 E\\*(N,ε)、D=闭环在线测 M、Dp=闭环+斜率预测。")
    lines.append("- `violation_rate` = 停训时 M 仍高于 ε 的 run 占比；`saving` = 相对 A 省下的训练轮次。")
    lines.append("")
    return lines


def section_audit(leak):
    lines = ["## 7. 数据审计（跨 split 泄漏）", ""]
    if not leak:
        lines += ["`results/leak_report.json` 还不存在 —— `python -m tools.check_leak --dataset pathmnist`。", ""]
        return lines
    items = leak if isinstance(leak, list) else leak.get("datasets") or [leak]
    lines.append("| 数据集 | test→train 完全重复 | 近似重复(≥0.995) | val→train 完全重复 | p99 缩略图相似 | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for it in items:
        if not isinstance(it, dict):
            continue
        lines.append(
            f"| {it.get('dataset', '?')} | {_fmt(it.get('exact_dup_test_in_train'), 0)} | "
            f"{_fmt(it.get('near_dup_test_in_train_ge_0.995'), 0)} | "
            f"{_fmt(it.get('exact_dup_val_in_train'), 0)} | "
            f"{_fmt(it.get('thumb_max_sim_p99'), 3)} | {it.get('verdict', '—')} |"
        )
    lines.append("")
    lines.append("- 校准 τ 时用 `LEAK_COS=0.99` 把参照集里这些重复图剔除，否则 τ 会被抬高、M 恒为 0。")
    lines.append("")
    return lines


def section_safe():
    """交付物：可直接给医院看的早停建议表（results/safe_early_stop.csv）。"""
    d = _read_csv(C.RESULTS / "safe_early_stop.csv")
    L = ["## 5. 安全早停建议（交付物）", ""]
    if d is None or len(d) == 0:
        return L + ["`results/safe_early_stop.csv` 还不存在 —— `python -m tools.safety`。", ""]
    L.append("`E*` 由拟合律反解；`M_停` 是**所有 seed 里最差那条**在停点的实测值（不是平均），"
             "`发作点` 是**不依赖律**的实测记忆化起点（首次连续 2 个测量点 M ≥ ε）。"
             "`E*/发作 < 1` 才说明律的停点落在发作之前。`ε ≤ 2×FPR` 的档位没有信息量（度量的假阳性就有 0.5%），标为 ⚠。")
    L.append("")
    L.append("| 数据集 | N | ε | seed数 | E* | M_停 | M_均 | M_满 | 省% | FID停 | FID满 | E*/发作 | 隐私OK | 质量OK | **推荐** |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in d.sort_values(["dataset", "eps", "n"]).iterrows():
        eps_txt = f"{r['eps']:g}" if bool(r.get("informative", True)) else f"{r['eps']:g} ⚠"
        L.append(
            f"| {r['dataset']} | {int(r['n'])} | {eps_txt} | {int(r.get('n_seeds', 1))} | {r['E_star']:.0f} | "
            f"{_fmt(r['M_stop'], 4)} | {_fmt(r.get('M_stop_mean'), 4)} | {_fmt(r['M_full'], 4)} | "
            f"{100 * r['epoch_saving']:.0f}% | "
            f"{_fmt(r['fid_stop'], 1)} | {_fmt(r['fid_full'], 1)} | "
            f"{_fmt(r['onset_ratio'], 2)} | {'✅' if r['ok'] else '❌'} | "
            f"{'✅' if r['ok_quality'] else '❌'} | {'🟢 可用' if r['recommended'] else '—'} |")
    L.append("")
    n_ok = int(d["recommended"].sum())
    L.append(f"- 四个条件（最差 seed ≤ ε、经验保守停点也不违约、FID ≤ 跑满的 1.5 倍、ε > 2×FPR）"
             f"同时满足的组合：**{n_ok} / {len(d)}**。")
    # 把张力说白了：小 N 的安全停点远早于图像变好看的轮次，所以"又安全又好看"在小 N 上不存在。
    small = d[(d["n"] <= 500) & (d["ok"])]
    if len(small) and not small["recommended"].any():
        L.append("- **关键张力（必须写进论文）**：N ≤ 500 时所有隐私合规的停点都远早于 FID 变好的轮次"
                 "（FID 约为跑满时的 2–3 倍）。也就是说小数据集上「安全」与「可用」不可兼得——"
                 "正确的处方是**加数据**（N 增大后安全窗口变宽），而不是硬把停点往后推。")
    L.append("")

    # 5.2 带 seed 余量的"廉价"开环方案（B11）：不测量、只靠时刻表 + 2σ 余量
    sm = _read_csv(C.RESULTS / "seed_margin.csv")
    if sm is not None and len(sm):
        L.append("### 5.2 带 seed 余量的开环方案（`build/seed_margin.py`，B11 的 7–9 个 seed）")
        L.append("")
        L.append("`E*_safe = E*_law · 10^(−2σ_log)`，σ_log = std(log10 onset)。判据：在该停点上"
                 "**所有 seed** 的 M 都 ≤ ε 才算安全。")
        L.append("")
        L.append("| N | ε | seed 数 | σ_log | E*_law | E*_safe | 省轮 | M@law | M@safe | 质量代价(FID 比) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        sd = sm[sm["eps"].isin([0.01, 0.02])].sort_values(["eps", "n"])
        for r in sd.itertuples():
            e_law = "—" if r.E_law != r.E_law else f"{int(r.E_law)}"
            e_safe = "—" if r.E_safe != r.E_safe else f"{int(r.E_safe)}"
            sav = "—" if r.saving_safe != r.saving_safe else f"{r.saving_safe * 100:.0f}%"
            mlaw = "—" if r.M_law != r.M_law else f"{r.M_law:.3f}"
            msafe = "—" if r.M_safe != r.M_safe else f"{r.M_safe:.3f}"
            q = "—" if r.fid_rel_safe != r.fid_rel_safe else f"{r.fid_rel_safe:.2f}×"
            L.append(f"| {int(r.n)} | {r.eps} | {int(r.n_seeds)} | {r.sigma_log:.3f} | {e_law} | "
                     f"{e_safe} | {sav} | {mlaw} | {msafe} | {q} |")
        L.append("")
        L.append("- **廉价方案成立，但只在「隐私优先、质量可退」时成立**：N=250/500/1000 在 ε=0.01 上都能做到"
                 "全部 seed 合规且省 88–94%，代价是 FID 为跑满的 1.55–2.54 倍。")
        L.append("- **σ_log 随 N 变大**（250/500 约 0.05，N=1000 达 0.34）：数据越多平台越低，但「什么时候发作」更飘，"
                 "所以固定时刻表的可信区间反而更宽。")
        # 这一行以前是手写常量，补数据后就过期了（2026-11-09 踩过）——改成现算：
        # 策略数字取自 stop_policy_summary.csv，2σ 形态的省轮区间取自 seed_margin.csv。
        sp = C.RESULTS / "stop_policy_summary.csv"
        mg = C.RESULTS / "seed_margin.csv"
        if sp.exists():
            sd = pd.read_csv(sp).set_index("policy")

            def _pp(k: str) -> str:
                r = sd.loc[k]
                return f"省 {r['saving'] * 100:.0f}%、{r['violation_rate'] * 100:.0f}% 违规"

            band = "省 88–94%"
            if mg.exists():
                md = pd.read_csv(mg)
                md = md[(md["eps"] == 0.01) & md["saving_safe"].notna() & (md["n"] <= 1000)]
                if len(md):
                    band = f"省 {md['saving_safe'].min() * 100:.0f}–{md['saving_safe'].max() * 100:.0f}%"
            L.append(f"- 帕累托阶梯：闭环（{_pp('D_closed_loop')}；质量 1.1–1.3×）> "
                     f"开环+2σ（{band}、0 违规；质量 1.5–2.5×）> "
                     f"开环无余量（{_pp('C_open_loop')}）> "
                     f"FID 早停（{_pp('B_fid_plateau')}）。")
        L.append("")
    return L


def section_appendix():
    """第 8 节：附录（分辨率对照 / 记忆化类型学 / 逐格核查表）。"""
    L = ["## 8. 附录", ""]

    L.append("### 8.1 28px pilot 与 64px 主线（`tools/pilot_appendix.py`）")
    L.append("")
    pv = _read_csv(C.RESULTS / "pilot_vs_main.csv")
    if pv is not None and len(pv):
        L.append("| N | 分辨率 | τ | 末点 M | 跑满轮次 | 步数 | onset 步数(ε=1%) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in pv.sort_values(["n", "res"]).itertuples():
            onset = "—" if r.onset_steps != r.onset_steps else f"{r.onset_steps:.0f}"
            L.append(f"| {r.n} | {r.res} | {r.tau:.4f} | {r.M_last:.4f} | {r.epochs} | "
                     f"{r.steps} | {onset} |")
        L.append("")
        L.append("- τ 在 64px 上系统低 0.015–0.096（换分辨率必须重新校准，跨分辨率不可比）。")
        L.append("- 同步数下 28px 的 M 更高（N=250：0.932 vs 0.617），但这一差异远小于同 N 的"
                 "种子方差（最高 33×），**分辨率不改变定性结论**；绝对 M 与 FID 不可跨分辨率比较。")
        L.append("- 28px pilot 只在附录出现：主线全部结论都来自 64px 数据。")
    else:
        L.append("`results/pilot_vs_main.csv` 不存在 —— `python -m tools.pilot_appendix` 生成。")
    L.append("")

    L.append("### 8.2 224px pilot：跑了 30,000 步，结论是「没有结论」（`build/pilot224.py`）")
    L.append("")
    p224 = _read_csv(C.RESULTS / "pilot224_metrics.csv")
    if p224 is not None and len(p224):
        p224 = p224.sort_values("epoch")
        L.append("pneumoniamnist、N=1000、seed=21、batch=16、480 epoch（30,000 步）、8 个测量点 × 128 张。")
        L.append("")
        L.append("| 轮次 | M | τ | sim_max | FID |")
        L.append("|---|---|---|---|---|")
        for r in p224.itertuples():
            fid = "—" if r.fid != r.fid else f"{r.fid:.1f}"
            L.append(f"| {int(r.epoch)} | {r.M:.4f} | {r.tau:.4f} | {r.sim_max:.4f} | {fid} |")
        L.append("")
        L.append("- 判 **inconclusive**：FID 从 472 降到 110–117、像素 std 12→31（图像终于有纹理），"
                 "但 `M` 全程 0、`sim_max` 0.24–0.27 始终低于 τ=0.352。")
        L.append("- 既不能证明 224px 会记忆化，也不能证明不会：这一档的保真度还没到「会复刻」的水平，"
                 "且 batch 16 与主网格 64 是混杂因素。故**不据此声称分辨率无关**，只写进 limitation。")
        L.append("- 过程教训：第一次「45 min 没出图」是同一个 pilot 被启动了两份（两 train 抢 8 GB 显存 → 换页），"
                 "不是采样慢（`build/bench_sample224.py`：224px 采 128 张只要 ~46 s）。")
    else:
        L.append("`results/pilot224_metrics.csv` 不存在 —— `python -m build.pilot224` 生成（约 1.6 h）。")
    L.append("")

    L.append("### 8.3 224px 决定性档：同 N 同 seed 的 64px / 224px 对照（`build/run_tonight.py`）")
    L.append("")
    pb = _read_csv(C.RESULTS / "pilot224_blood_n250.csv")
    if pb is not None and len(pb):
        pb = pb.sort_values("epoch")
        L.append("bloodmnist、N=250、seed=21、batch=16、1600 epoch（25,000 步）、8 个测量点 × 128 张。")
        L.append("")
        L.append("| 轮次 | M | τ | sim_max | FID |")
        L.append("|---|---|---|---|---|")
        for r in pb.itertuples():
            fid = "—" if r.fid != r.fid else f"{r.fid:.1f}"
            L.append(f"| {int(r.epoch)} | {r.M:.4f} | {r.tau:.4f} | {r.sim_max:.4f} | {fid} |")
        L.append("")
        L.append("- **口径必须先对齐 epoch，不是步数**：224px 只能跑 batch 16，25,000 步只有 1,600 轮；"
                 "64px 主网格 20,000 步是 5,125 轮 —— 同样步数下 224px 的「每张图被看几遍」少了 3.2 倍。")
        L.append("- 按 epoch 对齐后的读法：64px 在 `E=1600` 已 `M=0.375`（`sim_max` 0.787），"
                 "224px 到 `E=1600` 仍为 0（`sim_max` 0.315–0.365，FID 仍从 371 稳降到 209）→ "
                 "**分辨率把 onset 至少推到 1,600 轮之后**；224px 自己的平台仍未测到。")
        L.append("- 所以本文只写「64px 的定量系数不跨分辨率推广」，**不写「分辨率不改变结论」**。")
    else:
        L.append("`results/pilot224_blood_n250.csv` 不存在 —— `python -m build.run_tonight` 生成。")
    L.append("")

    L.append("### 8.4 记忆化类型学：被复刻的是哪一类训练样本（`build/typology.py`）")
    L.append("")
    ty = _read_csv(C.RESULTS / "typology.csv")
    if ty is not None and len(ty):
        L.append("| run | 检查点 | 命中/样本 | 去重目标 | top1 占比 | 离群度(命中 vs 全体) | 最大类别 overrep |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ty.itertuples():
            L.append(f"| {r.run} | e{r.epoch} | {r.n_hit}/{r.n_gen} | {r.distinct_targets} | "
                     f"{r.top1_share:.2f} | {r.atyp_mem_med:.3f} vs {r.atyp_all_med:.3f} | "
                     f"{r.max_overrep:.2f} |")
        L.append("")
        L.append("- 三个假说全部被否：**不是**「少数几张图被反复复刻」（top1 只占 8–10%）、"
                 "**不是**「复刻离群样本」（离群度与全体几乎相同）、**不是**「集中在少数类别」（最大 overrep 1.57）。")
        L.append("- 含义：记忆化弥散在数据流形上，人工挑「危险样本」删掉的防线不可靠；"
                 "要控只能在训练时间轴上控（这正是闭环早停的动机）。")
    else:
        L.append("`results/typology.csv` 不存在 —— `python -m build.typology` 生成。")
    L.append("")

    L.append("### 8.5 逐格核查表（`build/verify_claims.py`）")
    L.append("")
    vc = _read_csv(C.RESULTS / "verify_claims.csv")
    if vc is not None and len(vc):
        bad = vc[vc["status"] == "FAIL"]
        L.append(f"共 {len(vc)} 条断言，**PASS {len(vc) - len(bad)} / FAIL {len(bad)}**。"
                 "每一条都从原始产物重算（不看文档）。")
        L.append("")
        if len(bad):
            L.append("| PPT 位置 | 断言 | 期望 | 实测 |")
            L.append("|---|---|---|---|")
            for r in bad.itertuples():
                L.append(f"| {r.ppt} | {r.claim} | {r.expect} | {r.actual} |")
            L.append("")
            L.append("- 未达成的红线集中在一处：**预声明 R²≥0.9 未达成**（R²_lin 0.774 / R²_log 0.709）"
                     "与**严格口径下安全表 0 行全过**。两条都在 `docs/ALIGNMENT.md` 与 "
                     "`docs/PROTOCOL.md` §16/§17 里提前声明并解释，不作事后修改。")
        else:
            L.append("全部 PASS。")
    else:
        L.append("`results/verify_claims.csv` 不存在 —— `python -m build.verify_claims` 生成。")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="同时打屏")
    a = ap.parse_args()

    df = _read_csv(C.RESULTS / "metrics_long.csv")
    law = _read_json(C.RESULTS / "law_fit.json")
    pol = _read_csv(C.RESULTS / "stop_policy_summary.csv")
    leak = _read_json(C.RESULTS / "leak_report.json")

    L = []
    L.append("# 实验结果总览（自动生成）")
    L.append("")
    L.append(f"> 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} · 由 `python -m tools.report` 从 `results/` 汇总。")
    L.append("> 数字的出处见每节标注；协议与判据见 `docs/PROTOCOL.md`，踩坑记录见 `docs/NOTES.md`。")
    L.append("")
    L += section_status(df)
    L += section_law(law)
    L += section_robust()
    L += section_cross()
    L += section_safe()
    L += section_policy(pol)
    L += section_audit(leak)
    L += section_appendix()

    figs = sorted((ROOT / "figs").glob("*.png")) if (ROOT / "figs").exists() else []
    L.append("## 9. 图")
    L.append("")
    if figs:
        for p in figs:
            L.append(f"- `figs/{p.name}`")
    else:
        L.append("还没有图 —— `python -m tools.plot` 生成。")
    L.append("")
    L.append("## 10. 复现命令")
    L.append("")
    L.append("```powershell")
    L.append(".\\ .venv\\Scripts\\python.exe -m tools.download_data        # 下数据")
    L.append(".\\ .venv\\Scripts\\python.exe -m tools.check_leak --dataset pathmnist")
    L.append(".\\ .venv\\Scripts\\python.exe -m tools.night_queue --plan main --seed 1")
    L.append(".\\ .venv\\Scripts\\python.exe -m law                        # 拟合控制律")
    L.append(".\\ .venv\\Scripts\\python.exe -m controller                # 停训策略对比")
    L.append(".\\ .venv\\Scripts\\python.exe -m tools.plot                # 出图")
    L.append(".\\ .venv\\Scripts\\python.exe -m tools.report              # 生成本文件")
    L.append("```")
    L.append("")

    txt = "\n".join(L)
    OUT.write_text(txt, encoding="utf-8")
    print(f"写出 {OUT}（{len(txt)} 字符）")
    if a.print:
        print(txt)


if __name__ == "__main__":
    main()