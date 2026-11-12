"""把"权威数字"集中到 docs/NUMBERS.md，并检查各文档里是否还留着过期数字。

为什么需要：这个项目里同一组数字（α/γ/R²/省轮%/违规率/平台值）出现在 8 份文档里，
每跑一批新数据就要同步改 8 遍；漏改一处，读者看到的就是自相矛盾的两个版本。

用法：
  python -m build.doc_numbers            # 重新生成 docs/NUMBERS.md 并做一次过期扫描
  python -m build.doc_numbers --check    # 只扫描，不写文件（CI 里跑这个）
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "NUMBERS.md"

SCAN = [
    "README.md", "START_HERE.md", "docs/ABSTRACT.md", "docs/ALIGNMENT.md",
    "docs/OUTLINE.md", "docs/HANDOFF.md", "docs/RESULTS_TEXT.md",
    "docs/COMPARISON.md", "docs/METHODS.md",
]
# 这些文件里合法地会出现"历史值"（改了也说明改过），扫描时给出提示而不是错误
HISTORY_OK = {"docs/PROTOCOL.md", "docs/NOTES.md", "docs/NUMBERS.md"}


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def load() -> dict:
    d: dict = {}
    fit = json.loads((RES / "law_fit.json").read_text(encoding="utf-8"))
    best = fit["best"]
    f = fit["fits"][best]
    d["law_name"] = best
    d["params"] = [float(v) for v in f["params"]]
    d["r2_log"] = float(f["r2_log"])
    d["r2_lin"] = float(f.get("r2_lin", float("nan")))
    loo = f.get("loo", {}) or {}
    d["loo"] = float(loo.get("r2_pooled", float("nan")))
    d["aicc"] = float(f.get("aicc", float("nan")))

    boot = json.loads((RES / "law_bootstrap.json").read_text(encoding="utf-8"))
    d["ci"] = {k: (float(v["lo"]), float(v["hi"])) for k, v in boot["ci"].items()}

    ml = pd.read_csv(RES / "metrics_long.csv")
    d["n_rows"] = int(len(ml))
    d["n_runs"] = int(ml["run"].nunique())
    d["n_datasets"] = int(ml["dataset"].nunique()) if "dataset" in ml else 0
    if "M" in ml:
        d["n_measurable"] = int((ml["M"] >= 0.02).sum())

    gs = pd.read_csv(RES / "grid_summary.csv")
    plat = gs.groupby("n")["M_max"].agg(["mean", "count"]).reset_index()
    d["plateau"] = [(int(r.n), float(r["mean"]), int(r["count"])) for _, r in plat.iterrows()]

    sp = pd.read_csv(RES / "stop_policy_summary.csv")
    d["policy"] = {
        r["policy"]: dict(epochs=float(r["epochs"]), saving=float(r["saving"]),
                          M_stop=float(r["M_stop"]), viol=float(r["violation_rate"]),
                          n=int(r["n"]))
        for _, r in sp.iterrows()
    }

    se = pd.read_csv(RES / "safe_early_stop.csv")
    d["safe_rows"] = int(len(se))
    for col in ("ok", "ok_emp", "ok_quality", "recommended"):
        if col in se:
            d[f"safe_{col}"] = int(se[col].astype(bool).sum())

    sm = pd.read_csv(RES / "seed_margin.csv")
    d["seed_margin"] = sm

    enc = RES / "encoder_summary.csv"
    if enc.exists():
        d["encoder"] = pd.read_csv(enc)

    fpr = RES / "fpr_sensitivity.csv"
    if fpr.exists():
        d["fpr"] = pd.read_csv(fpr)

    vc = RES / "verify_claims.csv"
    if vc.exists():
        v = pd.read_csv(vc)
        d["verify_pass"] = int((v["status"] == "PASS").sum())
        d["verify_fail"] = int((v["status"] != "PASS").sum())
        d["verify_fails"] = v[v["status"] != "PASS"][["ppt", "claim"]].values.tolist()

    d["pilot224"] = {}
    for p in sorted(RES.glob("pilot224_*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "M" not in df:
            continue
        d["pilot224"][p.name] = dict(rows=len(df), M_last=float(df["M"].iloc[-1]),
                                    M_max=float(df["M"].max()))
    # 下界探针：N<FIT_MIN_N 的两档（128/192）。它们不进主拟合（白化维数 k=N/2 与主网格不同、
    # 只有 1 个 seed、检查点间隔 1250/825 轮），只用来回答"小数据是否也记忆化"和给出 onset 上界。
    d["probe"] = []
    ml = RES / "metrics_long.csv"
    if ml.exists():
        m = pd.read_csv(ml)
        m = m[m["n"] < 250]
        for run, g in m.groupby("run"):
            g = g.sort_values("epoch")
            steps_per_epoch = float(g["n"].iloc[0]) / 64.0
            first = g.iloc[0]
            d["probe"].append(dict(
                run=run, n=int(g["n"].iloc[0]), seeds=int(g["seed"].nunique()),
                points=len(g), tau=float(first["tau"]), k=int(max(32, min(384, int(g["n"].iloc[0]) // 2))),
                M_max=float(g["M"].max()), epochs=int(first["epoch"]),
                M_first=float(first["M"]), steps_first=int(round(first["epoch"] * steps_per_epoch)),
                bound_steps=round(steps_per_epoch * int(first["epoch"]) / int(g["n"].iloc[0]), 1),
            ))
        d["probe"].sort(key=lambda r: r["n"])

    # 同子集、同 seed、按 epoch 对齐的 64px vs 224px 对照（控制变量：子集与种子相同）
    d["res_cmp"] = []
    p64 = RES / "pilot64_blood_n250_s21.csv"
    p224 = RES / "pilot224_blood_n250.csv"
    if p64.exists() and p224.exists():
        a = pd.read_csv(p64).set_index("epoch")
        b = pd.read_csv(p224).set_index("epoch")
        d["tau224"] = float(b["tau"].iloc[0])
        for ep in sorted(set(a.index) & set(b.index)):
            d["res_cmp"].append(dict(epoch=int(ep),
                                     m64=float(a.loc[ep, "M"]), m224=float(b.loc[ep, "M"]),
                                     s64=float(a.loc[ep, "sim_max"]), s224=float(b.loc[ep, "sim_max"])))

    # R22（容量 / 隐私训练 / 224px 自身平台）：训练规则与主网格不同，独立 CSV、不进主律拟合
    d["r22"] = []
    r22 = RES / "r22_summary.csv"
    if r22.exists():
        df = pd.read_csv(r22)
        for _, r in df.iterrows():
            row = dict(run=str(r["run"]), points=int(r["points"]), emax=int(r["epochs_max"]),
                       tau=float(r["tau"]), M_max=float(r["M_max"]), sim=float(r["sim_max_max"]),
                       e1=None if pd.isna(r["first_1pct_epoch"]) else int(r["first_1pct_epoch"]),
                       fid0=float(r["fid_first"]), fid1=float(r["fid_last"]),
                       eps=None if pd.isna(r["sigma"]) else float(r["eps_ub_steps4800"]),
                       sigma=None if pd.isna(r["sigma"]) else float(r["sigma"]), pairs=[])
            for ep in (160, 320, 480, 640, 800, 1600, 2400, 5125):
                k, kr = f"M@{ep}", f"ref@{ep}"
                if k in r and not pd.isna(r[k]):
                    row["pairs"].append((ep, float(r[k]),
                                         None if pd.isna(r.get(kr)) else float(r.get(kr))))
            d["r22"].append(row)

    return d


def render(d: dict) -> str:
    A0, alpha, beta, E50, gamma = d["params"]
    L: list[str] = []
    L.append("# 权威数字（自动生成，别手改）")
    L.append("")
    L.append("> 本文件由 `python -m build.doc_numbers` 从 `results/` 现算生成。")
    L.append("> 任何文档引用数字时，**以本文件为准**；数字对不上就是那份文档过期了。")
    L.append("")
    L.append("## 1. 数据规模")
    L.append("")
    L.append(f"- 测量点（`metrics_long.csv` 行数）：**{d['n_rows']}**")
    L.append(f"- 训练 run 数：**{d['n_runs']}**；数据集：**{d['n_datasets']}**")
    if "n_measurable" in d:
        L.append(f"- 其中可测点（M ≥ 0.02）：**{d['n_measurable']}**")
    L.append("")
    L.append("## 2. 控制律")
    L.append("")
    L.append(f"- 选中模型：`{d['law_name']}`")
    L.append(f"- 参数：A₀={_fmt(A0)}（在 N=250 处）、α={_fmt(alpha)}、β={_fmt(beta)}、"
             f"E50={_fmt(E50, 1)}（在 N=250 处）、γ={_fmt(gamma)}")
    ci = d["ci"]
    L.append("- 95% CI：" + "；".join(
        f"{k} [{_fmt(v[0])}, {_fmt(v[1])}]" for k, v in ci.items()))
    L.append(f"- R²_log = **{_fmt(d['r2_log'])}**，R²_lin = {_fmt(d['r2_lin'])}，"
             f"留一池化 R² = **{_fmt(d['loo'])}**，AICc = {_fmt(d['aicc'], 1)}")
    L.append("")
    L.append("## 3. 逐档平台（各 seed 的 M_max 均值）")
    L.append("")
    L.append("| N | 平台 M_max | run 数 |")
    L.append("|---|---|---|")
    for n, m, c in d["plateau"]:
        L.append(f"| {n} | {_fmt(m)} | {c} |")
    L.append("")
    L.append("## 4. 停训策略（ε=0.01，所有 run 取最差）")
    L.append("")
    L.append("| 策略 | 平均轮次 | 省轮 | M_stop | 违规率 |")
    L.append("|---|---|---|---|---|")
    for k, v in d["policy"].items():
        L.append(f"| {k} | {v['epochs']:.1f} | {v['saving']*100:.1f}% | "
                 f"{_fmt(v['M_stop'])} | {v['viol']*100:.1f}% |")
    L.append("")
    L.append("## 5. 安全表")
    L.append("")
    parts = [f"共 {d['safe_rows']} 行"]
    for col, label in (("ok", "隐私达标"), ("ok_emp", "经验达标"),
                       ("ok_quality", "质量达标"), ("recommended", "四条件全过")):
        if f"safe_{col}" in d:
            parts.append(f"{label} {d[f'safe_{col}']} 行")
    L.append("- " + "；".join(parts))
    L.append("")
    if "seed_margin" in d:
        sm = d["seed_margin"]
        L.append("## 6. 免测量的 2σ 开环形态（ε=0.01）")
        L.append("")
        L.append("| N | seed 数 | σ_log | E*_law | E*_safe | 省轮 | M@safe | FID 相对跑满 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in sm.iterrows():
            if int(r["n"]) > 1500:
                continue
            L.append(f"| {int(r['n'])} | {int(r['n_seeds'])} | {_fmt(r['sigma_log'])} | "
                     f"{r['E_law']:.0f} | {r['E_safe']:.0f} | {r['saving_safe']*100:.0f}% | "
                     f"{_fmt(r['M_safe'])} | {r['fid_rel_safe']:.2f}× |")
        L.append("")
    if "encoder" in d:
        e = d["encoder"]
        L.append("## 7. 换编码器的稳健性（DINOv2 → ResNet50）")
        L.append("")
        L.append("| run | Spearman | 末点差 | 平台 DINO | 平台 RN50 |")
        L.append("|---|---|---|---|---|")
        for _, r in e.iterrows():
            L.append(f"| {r['run']} | {r['spearman']:.3f} | {_fmt(float(r['max_abs_diff']))} | "
                     f"{_fmt(float(r['plat_dino']))} | {_fmt(float(r['plat_rn50']))} |")
        L.append("")
    if "fpr" in d:
        L.append("## 8. FPR 敏感性")
        L.append("")
        L.append("| FPR | τ | M |")
        L.append("|---|---|---|")
        for _, r in d["fpr"].iterrows():
            L.append(f"| {r['fpr']} | {_fmt(float(r['tau']))} | {_fmt(float(r['M']))} |")
        L.append("")
    if d.get("pilot224"):
        L.append("## 9. 224px 决定性档")
        L.append("")
        L.append("| 文件 | 行数 | 末点 M | M_max |")
        L.append("|---|---|---|---|")
        for k, v in d["pilot224"].items():
            L.append(f"| {k} | {v['rows']} | {_fmt(v['M_last'])} | {_fmt(v['M_max'])} |")
        L.append("")
        cmp = d.get("res_cmp") or []
        if cmp:
            L.append("**按 epoch 对齐的同子集对照**（bloodmnist N=250 seed=21；224px 受 batch=16 限制，"
                     "同一步数对应的轮数少 3.2 倍，所以只能比轮）：")
            L.append("")
            L.append("| epoch | 64px M | 224px M | 64px sim_max | 224px sim_max |")
            L.append("|---|---|---|---|---|")
            for r in cmp:
                L.append(f"| {r['epoch']} | {_fmt(r['m64'])} | {_fmt(r['m224'])} | "
                         f"{_fmt(r['s64'])} | {_fmt(r['s224'])} |")
            L.append("")
            L.append("- 判定口径：只看两者**都测过**的 epoch；224px 的 sim_max 若始终低于 τ（"
                     f"{_fmt(d.get('tau224', 0.0))}），说明它连「接近重复」的门都没摸到，"
                     "这比「M=0」更接近事实。")
            L.append("")
    if d.get("probe"):
        L.append("## 10. 下界探针 N=128/192（**不进主拟合**）")
        L.append("")
        L.append("| run | N | seed | 点 | k(白化维) | τ | M_max | 首个检查点 | 该点 M | 步数/N 上界 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in d["probe"]:
            L.append(f"| {r['run']} | {r['n']} | {r['seeds']} | {r['points']} | {r['k']} | "
                     f"{_fmt(r['tau'])} | {_fmt(r['M_max'])} | e{r['epochs']} | "
                     f"{_fmt(r['M_first'])} | ≤{r['bound_steps']} |")
        L.append("")
        L.append("- 两档都**会**记忆化（M_max 0.99 / 0.87，高于 N=250 的 0.68），方向与律一致。")
        L.append("- 但它们的白化维数只有 64/96（主网格 125/384），τ 与 M 不与主网格同尺；"
                 "且检查点间隔 1250/825 轮，首个检查点就已经越线 → 只能给 onset 的**上界**，"
                 "无法确认或推翻「步数/N≈10」这个常数。故 `config.FIT_MIN_N=250` 把它们挡在主拟合之外。")
        L.append("")
    if "verify_pass" in d:
        L.append("## 11. 逐格核查")
        L.append("")
        L.append(f"- {d['verify_pass']} PASS / {d['verify_fail']} FAIL")
        for ppt, claim in d["verify_fails"]:
            L.append(f"  - {ppt}：{claim}")
        L.append("")
    if d.get("r22"):
        L.append("## 12. R22：容量 / 隐私训练 / 224px 自身平台（**均不进主拟合**）")
        L.append("")
        L.append("| run | 点 | 末轮 | τ | M_max | sim_max 峰 | 首个 ≥1% | FID 首→末 | σ / ε 上界 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in d["r22"]:
            e1 = "—" if r["e1"] is None else f"e{r['e1']}"
            dp = "—" if r["sigma"] is None else f"σ={r['sigma']:.1f} / ε≤{r['eps']:.1f}"
            L.append(f"| {r['run']} | {r['points']} | {r['emax']} | {_fmt(r['tau'])} | "
                     f"{_fmt(r['M_max'])} | {_fmt(r['sim'])} | {e1} | "
                     f"{r['fid0']:.0f} → {r['fid1']:.0f} | {dp} |")
        L.append("")
        for r in d["r22"]:
            if not r["pairs"]:
                continue
            L.append(f"**{r['run']}** 与对照同 epoch（`{r['run']}` 的顺序即预声明顺序 A→E）：")
            L.append("")
            L.append("| epoch | 本 run M | 对照 M |")
            L.append("|---|---|---|")
            for ep, m, ref in r["pairs"]:
                L.append(f"| {ep} | {_fmt(m)} | {'—' if ref is None else _fmt(ref)} |")
            L.append("")
        L.append("- ε 只报强组合保守上界（Abadi Thm.1，δ=1e-5，q=0.256，T=4800 步）；本实现没做矩账户。")
        L.append("- DP 两组的 M 全 0 但 FID 从约 120 劣到 309–353 → 「记忆化消失」不等于「可以这样用」。")
        L.append("- 容量档（base=64）平台与 onset 都比主网格高 → α 只能作「本数据 + 本模型」的条件指数。")
        L.append("")
    L.append("## 13. 改数字时要连带改哪里")
    L.append("")
    L.append("- 律参数/区间 → `docs/ALIGNMENT.md` 04 行、`docs/ABSTRACT.md`、`docs/OUTLINE.md` C1 行、`docs/RESULTS_TEXT.md` §5.2");
    L.append("- 停训策略/安全表 → `docs/ALIGNMENT.md` 补 1 行、`docs/OUTLINE.md` C2 行、`docs/RESULTS_TEXT.md` §5.4/§5.5、`README.md` 三条结论表");
    L.append("- 平台与 N → `figs/fig1_triptych.png`（跑 `tools.make_main_figs`）与 `docs/COMPARISON.md`");
    L.append("- 全部重生成：`python -m build.finalize --skip-score`，然后 `python -m build.doc_numbers`，最后 `python -m tools.manifest`");
    L.append("")
    return "\n".join(L)


NUM_RE = re.compile(r"(?<![\d.])(\d+\.\d{2,4})(?![\d])")

# 只扫"紧挨着我们自己的符号"的数字，避免把文献里的数字（COMPARISON.md 大量引用别人的
# τ_mem/n、α 等）误判成过期。每个模式给出 (正则, 期望值 key)。
TARGETS: list[tuple[str, str]] = [
    (r"α\s*[=＝≈]\s*([0-9]\.[0-9]{2,4})", "alpha"),
    (r"γ\s*[=＝≈]\s*[−-]?\s*([0-9]\.[0-9]{2,4})", "gamma_abs"),
    (r"β\s*[=＝≈]\s*([0-9]\.[0-9]{2,4})", "beta"),
    (r"A₀\s*[=＝≈]\s*([0-9]\.[0-9]{2,4})", "A0"),
    (r"R²_log\s*[=＝≈]?\s*([0-9]\.[0-9]{2,4})", "r2_log"),
    (r"R²_lin\s*[=＝≈]?\s*([0-9]\.[0-9]{2,4})", "r2_lin"),
    (r"留一池化\s*(?:R²)?\s*[=＝≈]?\s*([0-9]\.[0-9]{2,4})", "loo"),
]


def _expected(d: dict) -> dict[str, float]:
    A0, alpha, beta, _E50, gamma = d["params"]
    return {"alpha": alpha, "gamma_abs": abs(gamma), "beta": beta, "A0": A0,
            "r2_log": d["r2_log"], "r2_lin": d["r2_lin"], "loo": d["loo"]}


def check(d: dict) -> int:
    """只检查"α=1.97"这种带我们自己的符号的数字；差 >0.5% 视为不一致。"""
    exp = _expected(d)
    hits = 0
    for rel in SCAN:
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith(">"):
                continue
            # "自由拟合" 指跨数据集单独拟合的 R²（另一个量），不是主律的 R²，跳过
            if "自由拟合" in line:
                continue
            # "逐 run 口径" 指不折 seed 的拟合（另一组参数），与主律（seed 均值）不同源，跳过
            if "逐 run" in line:
                continue
            for pat, key in TARGETS:
                for m in re.finditer(pat, line):
                    val = float(m.group(1))
                    want = exp[key]
                    if abs(val - want) <= max(1e-3, abs(want) * 0.005):
                        continue
                    print(f"[过期] {rel}:{i}  {key}={m.group(1)}（当前 {want:.4f}）")
                    hits += 1
    if hits:
        print(f"\n共 {hits} 处过期数字。历史值只应留在 docs/PROTOCOL.md 与 docs/NOTES.md；"
              f"正文/README 里请改成 docs/NUMBERS.md 的当前值。")
    else:
        print("扫描通过：正文里没有与权威值不一致的 α/γ/β/A₀/R²/留一。")
    return 0 if hits == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="不改文件，只比对 docs/NUMBERS.md 是否与当前 results 一致（CI 用）")
    a = ap.parse_args()
    d = load()
    if a.verify:
        want = render(d)
        have = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if want.replace("\r\n", "\n") != have.replace("\r\n", "\n"):
            print(f"[STALE] {OUT.relative_to(ROOT)} 与 results 不一致；"
                  f"请跑 `python -m build.doc_numbers` 重新生成")
            return 1
        print(f"{OUT.relative_to(ROOT)} 与 results 一致（{len(want)} 字符）")
        return 0
    if not a.check:
        OUT.write_text(render(d), encoding="utf-8")
        print(f"写出 {OUT.relative_to(ROOT)}（{len(render(d))} 字符）")
    return check(d)


if __name__ == "__main__":
    sys.exit(main())