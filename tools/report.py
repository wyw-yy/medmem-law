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
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def section_status(df):
    lines = ["## 1. 主网格完成度", ""]
    if df is None:
        lines += ["`results/metrics_long.csv` 还不存在 —— 先跑 `python -m tools.night_queue --plan main`。", ""]
        return lines
    lines.append("| N | seed | 测量点 | 最后 epoch / 预算 | τ | M_max | FID_min | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (n, s), g in df.groupby(["n", "seed"], sort=True):
        eps = sorted(g["epoch"].unique())
        want = C.epochs_for(int(n))
        done = int(eps[-1]) if eps else 0
        ok = "✅ 完成" if done >= want * 0.99 else f"⏳ {done}/{want}"
        fmin = g["fid"].min() if g["fid"].notna().any() else None
        lines.append(
            f"| {int(n)} | {int(s)} | {len(eps)} | {done} / {want} | "
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
    p = f.get("params") or []
    try:
        return p[2] if "L3" in name else p[3]
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
        lines.append("| 候选律 | R²_log | R²_lin | AICc | 留一 N R²（均值/最差） | 拟合 γ | 参数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, f in fits.items():
            mark = "**" if name == best else ""
            loo = f.get("loo") or {}
            prm = ", ".join(f"{v:.3g}" for v in (f.get("params") or []))
            lines.append(
                f"| {mark}{name}{mark} | {_fmt(f.get('r2_log'))} | {_fmt(f.get('r2_lin'))} | "
                f"{_fmt(f.get('aicc'), 1)} | {_fmt(loo.get('r2_mean'))} / {_fmt(loo.get('r2_min'))} | "
                f"{_fmt(_gamma_of(name, f))} | {prm} |"
            )
        lines.append("")
    lines.append(f"- 选定：**{best}**（判据：AICc 最小 → 留一 N R² 最大 → 参数可解释，见 `docs/PROTOCOL.md`）。")
    bf = fits.get(best) or {}
    r2l = bf.get("r2_log")
    loo = (bf.get("loo") or {}).get("r2_mean")
    v1 = "✅" if (r2l if r2l is not None else 0) >= 0.9 else "❌"
    v2 = "✅" if (loo if loo is not None else 0) >= 0.8 else "❌"
    lines.append(f"- 验收线：样本内 R²_log ≥ 0.9 {v1}（{_fmt(r2l)}）；留一 N R² ≥ 0.8 {v2}（{_fmt(loo)}）。")
    g = law.get("gamma_hat")
    if g is not None:
        lines.append(f"- **γ = {_fmt(g)}**（L1/L2 里 `E50 = c·N^γ`）。γ≈1 意味着"
                     "「按 epoch 计的起点随 N 线性推迟」，即 τ_mem 随 N 线性增长。")
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


def section_policy(df):
    lines = ["## 3. 停训策略对比", ""]
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
            if c == "saving" or c == "violation_rate":
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
    lines = ["## 4. 数据审计（跨 split 泄漏）", ""]
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
    L += section_policy(pol)
    L += section_audit(leak)

    figs = sorted((ROOT / "figs").glob("*.png")) if (ROOT / "figs").exists() else []
    L.append("## 5. 图")
    L.append("")
    if figs:
        for p in figs:
            L.append(f"- `figs/{p.name}`")
    else:
        L.append("还没有图 —— `python -m tools.plot` 生成。")
    L.append("")
    L.append("## 6. 复现命令")
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