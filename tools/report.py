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
        "- **未达线的原因写在明处**：可测范围只覆盖 N ≤ 1000（N≥2000 的记忆率低于检出限 3/512），"
        "留出一档大 N 等于外推一条从没测到的过程。故主结论限定在可测区间，E\\*=nan 表示该 N 在任意训练时长下都到不了 ε。"
    )
    lines.append(
        "- **口径**：律拟合的是**种子均值曲线**（`law.collapse_seeds`）；单 seed 拟合能到 R²_lin≈0.90，"
        "但那是运气好的那条曲线。剧集 `R²(可测)` 只在 M ≥ 0.02 的点上算，"
        "它比全体更低说明**误差集中在真正有信号的区域**（逐档 bias 0.42–1.72，见 `docs/NOTES.md` §4.2）。"
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

    figs = sorted((ROOT / "figs").glob("*.png")) if (ROOT / "figs").exists() else []
    L.append("## 8. 图")
    L.append("")
    if figs:
        for p in figs:
            L.append(f"- `figs/{p.name}`")
    else:
        L.append("还没有图 —— `python -m tools.plot` 生成。")
    L.append("")
    L.append("## 9. 复现命令")
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