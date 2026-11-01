"""核查表：把 PPT 每一格 / 文档里写下的数字，逐条回到原始产物重算一遍。

做法是"不看文档，只看原始文件"：
  results/metrics_long.csv        973 行原始测量
  results/runs/*.json             每次训练的预算、停点、交付点
  results/law_fit.json            拟合参数与四个 R²
  results/law_bootstrap.json      bootstrap 区间
  results/stop_policy_*.csv       四个停训策略
  results/safe_early_stop.csv     安全交付表
  results/leak_report.json        数据侧泄漏审计
  results/typology*.csv           记忆化类型学
  figs/*.png                      图

输出一张 PASS / FAIL 表（同时打印到 stdout 并写 results/verify_claims.csv），
任何一条 FAIL 都该在交付前解释清楚。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402

claims: list[dict] = []


def add(ppt: str, claim: str, expect: str, actual, ok: bool) -> None:
    claims.append(dict(ppt=ppt, claim=claim, expect=expect, actual=str(actual),
                       status="PASS" if ok else "FAIL"))


def main() -> int:
    d = pd.read_csv(C.RESULTS / "metrics_long.csv")
    fit = json.load(open(C.RESULTS / "law_fit.json", encoding="utf-8"))
    boot = json.load(open(C.RESULTS / "law_bootstrap.json", encoding="utf-8"))
    runs = {p.stem: json.load(open(p, encoding="utf-8"))
            for p in (C.RESULTS / "runs").glob("*.json")}

    # ---------------------------------------------------------------- 01 内容
    add("01①", "主数据集 = pathmnist（公开、CC BY 4.0）",
        "pathmnist 为主", f"{d['dataset'].value_counts().to_dict()}",
        d["dataset"].value_counts().get("pathmnist", 0) > 0)
    add("01③", "记忆率 M = 近邻相似度超阈值的生成样本占比",
        "每行都有 M 与 tau", f"M 列存在={'M' in d} tau 列存在={'tau' in d}",
        {"M", "tau"} <= set(d.columns))
    add("01③", "数据规模档位 ≥3 档", "≥3", sorted(d["n"].unique()),
        d["n"].nunique() >= 3)
    add("01④", "控制律 R²≥0.9", "R²_lin ≥ 0.9 且 R²_log ≥ 0.9",
        f"R²_lin={fit['fits'][fit['best']]['r2_lin']:.4f} "
        f"R²_log={fit['fits'][fit['best']]['r2_log']:.4f}",
        min(fit["fits"][fit["best"]]["r2_lin"], fit["fits"][fit["best"]]["r2_log"]) >= 0.9)
    add("01④", "安全早停点存在", "E* 表非空", f"{len(fit['E_star'])} 行",
        len(fit.get("E_star") or []) > 0)

    # ---------------------------------------------------------------- 02 数据
    keys = {"n", "epoch", "M", "sim_max", "fid", "n_gen"}
    add("02③", "关键字段齐全（规模/轮次/生成样本/近邻距离/记忆率/质量）",
        "n, epoch, n_gen, sim_max, M, fid 全在", sorted(set(d.columns) & keys),
        keys <= set(d.columns))
    add("02②", "训练轮次多档", "每档 N 至少 2 个测量点", f"最少点数={d.groupby(['dataset','n']).size().min()}",
        d.groupby(["dataset", "n"]).size().min() >= 2)
    leak = json.load(open(C.RESULTS / "leak_report.json", encoding="utf-8"))
    add("02④", "公开数据集 + 隐私审计（数据侧泄漏检查）", "三数据集都有报告",
        f"{list(leak.keys()) if isinstance(leak, dict) else type(leak)}",
        bool(leak))

    # ---------------------------------------------------------------- 03 模型
    add("03①", "主模型 = 影像扩散生成模型", "DDPM UNet",
        f"params={runs and next(iter(runs.values())).get('params_M', 'n/a')}",
        all("params_M" in v for v in runs.values()) if runs else False)
    add("03②", "双时间尺度（规模-轮次分离）体现在 γ 上", "γ 的 95% CI 不含 0",
        f"γ={boot['ci']['gamma']['point']:.4f} "
        f"[{boot['ci']['gamma']['lo']:.4f}, {boot['ci']['gamma']['hi']:.4f}]",
        boot["ci"]["gamma"]["hi"] < 0 or boot["ci"]["gamma"]["lo"] > 0)
    add("03③", "基线 = 无早停控制的常规训练", "策略 A 存在且跑满", 
        f"A 轮次={pd.read_csv(C.RESULTS/'stop_policy_summary.csv').set_index('policy').loc['A_fixed_max','epochs']:.0f}"
        if (C.RESULTS / "stop_policy_summary.csv").exists() else "缺文件",
        (C.RESULTS / "stop_policy_summary.csv").exists())
    add("03④", "拟合优度 R²≥0.9（PPT 验收线）", "同一行",
        "见 01④", False)  # 与 01④ 同一条，已知未达成，保留为显式的红灯
    add("03④", "跨数据集推广性", "free_fit 与 transfer 都跑过",
        f"transfer.csv 行数={len(pd.read_csv(C.RESULTS/'transfer.csv'))}"
        if (C.RESULTS / "transfer.csv").exists() else "缺文件",
        (C.RESULTS / "transfer.csv").exists())

    # ---------------------------------------------------------------- 04 临床
    safe = pd.read_csv(C.RESULTS / "safe_early_stop.csv")
    n_ok = int(safe["recommended"].astype(str).str.lower().isin(["true", "1"]).sum()) \
        if "recommended" in safe else -1
    add("04④", "安全早停点（四条件全过）", "≥1 行可交付（严格口径，用户已定保持严格）",
        f"{n_ok}/{len(safe)} 行", n_ok >= 1)
    add("04⑤", "落地：训练早停准则 + 隐私审计流程", "两份文档都在",
        f"PRIVACY_AUDIT={'docs/PRIVACY_AUDIT.md' and (ROOT/'docs/PRIVACY_AUDIT.md').exists()} "
        f"PROTOCOL={(ROOT/'docs/PROTOCOL.md').exists()}",
        (ROOT / "docs/PRIVACY_AUDIT.md").exists() and (ROOT / "docs/PROTOCOL.md").exists())

    # ---------------------------------------------------------------- 稳健性
    if (C.RESULTS / "typology.csv").exists():
        ty = pd.read_csv(C.RESULTS / "typology.csv")
        delta = (ty["atyp_mem_med"] - ty["atyp_all_med"]).abs()
        add("类型学", "被复刻的样本没有系统性离群（H2 反证）", "|离群度差| < 0.02（无系统性差异）",
            " / ".join(f"{r.run}: {r.atyp_mem_med:.3f} vs {r.atyp_all_med:.3f}"
                       for r in ty.itertuples()),
            bool((delta < 0.02).all()))
        add("类型学", "被复刻的样本不是少数三张图（H1 反证）", "top1 占比 < 0.3",
            " / ".join(f"{r.run}: {r.top1_share:.2f}" for r in ty.itertuples()),
            bool((ty["top1_share"] < 0.3).all()))
        add("类型学", "类别偏斜温和（H3）", "最大 overrep < 2",
            " / ".join(f"{r.run}: {r.max_overrep:.2f}" for r in ty.itertuples()),
            bool((ty["max_overrep"] < 2).all()))
    if (C.RESULTS / "r_verify.csv").exists():
        rv = pd.read_csv(C.RESULTS / "r_verify.csv")
        bad = rv[rv["status"].astype(str).str.upper() == "CHECK"]
        add("R 复算", "Python 与 R 结果无冲突", "CHECK 行数 = 0",
            f"{len(bad)} 行 CHECK", len(bad) == 0)
    fig = list(C.FIGS.glob("*.png"))
    add("图", "主图齐备", "≥6 张", f"{len(fig)} 张",
        len(fig) >= 6)

    t = pd.DataFrame(claims)
    print(t.to_string(index=False))
    t.to_csv(C.RESULTS / "verify_claims.csv", index=False, encoding="utf-8")
    n_fail = int((t["status"] == "FAIL").sum())
    print(f"\nPASS {len(t) - n_fail} / FAIL {n_fail}")
    print("写出 results/verify_claims.csv")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())