"""把 R23 的 24 组结果写进 docs/NOTES.md（§21）。一次性脚本，跑完即删。"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

R = C.RESULTS


def row(tag: str) -> pd.Series:
    d = pd.read_csv(R / f"{tag}.csv")
    f = d["fid"].dropna()
    return pd.Series({
        "n_pts": len(d),
        "ep0": int(d.epoch.min()),
        "ep1": int(d.epoch.max()),
        "M_max": float(d.M.max()),
        "sim_pk": float(d.sim_max.max()),
        "sim_last": float(d.sim_max.iloc[-1]),
        "tau": float(d.tau.iloc[0]),
        "fid0": float(f.iloc[0]) if len(f) else np.nan,
        "fid1": float(f.iloc[-1]) if len(f) else np.nan,
    })


def onset(tag: str, thr: float = 0.01) -> str:
    d = pd.read_csv(R / f"{tag}.csv")
    h = d[d.M >= thr]
    return f"e{int(h.epoch.iloc[0])}" if len(h) else f">{int(d.epoch.max())}"


def table(rows: list[tuple[str, str]], head: str) -> str:
    out = [head,
           "| " + " | ".join(h for h, _ in rows[0][1:]) + " |",
           "|" + "|".join("---" for _ in rows[0][1:]) + "|"]
    return "\n".join(out)


# ---------- DP ----------
dp_tags = [("σ=1.0", "dp_s1_n250_s1_long"), ("σ=2.0", "dp_s2_n250_s1_long"),
           ("σ=4.0", "dp_s4_n250_s1_long"), ("σ=8.0", "dp_s8_n250_s1_long")]
dp_rows = []
for lab, tag in dp_tags:
    r = row(tag)
    dp_rows.append((lab, r))
base = row("pathmnist_n250_s1") if (R / "pathmnist_n250_s1.csv").exists() else None
g = pd.read_csv(R / "metrics_long.csv")
b = g[g.run == "pathmnist_n250_s1"]
base = pd.Series({"n_pts": len(b), "M_max": float(b.M.max()), "sim_pk": float(b.sim_max.max()),
                  "tau": float(b.tau.iloc[0]), "fid1": float(b.fid.dropna().iloc[-1])})

dp_tbl = "\n".join(
    f"| {lab} | {int(r.n_pts)} | {r.M_max:.4f} | {r.sim_pk:.3f} | {r.sim_last:.3f} | "
    f"{r.sim_pk / r.tau:.2f} | {r.fid0:.1f} → {r.fid1:.1f} |"
    for lab, r in dp_rows)

clip_tbl = "\n".join(
    f"| C={lab} | {int(row(t).n_pts)} | {row(t).M_max:.4f} | {row(t).sim_pk:.3f} | "
    f"{row(t).fid0:.1f} → {row(t).fid1:.1f} |"
    for lab, t in [("0.1", "dp_s8_c01_n250_s1"), ("0.5", "dp_s8_c05_n250_s1"), ("2.0", "dp_s8_c20_n250_s1")])

# ---------- 容量 ----------
cap = [("base=32 · seed1", "pathmnist_n250_s1", "metrics_long"),
       ("base=32 · seed2", "pathmnist_n250_s2", "metrics_long"),
       ("base=48 · seed1", "b48_pathmnist_n250_s1", "csv"),
       ("base=64 · seed1", "b64_pathmnist_n250_s1", "csv"),
       ("base=64 · seed2", "b64_pathmnist_n250_s2", "csv")]
cap_lines = []
for lab, tag, kind in cap:
    if kind == "csv":
        r = row(tag)
        on = onset(tag)
        cap_lines.append(f"| {lab} | {on} | {r.M_max:.4f} | {r.fid1:.1f} |")
    else:
        sub = g[g.run == tag]
        h = sub[sub.M >= 0.01]
        on = f"e{int(h.epoch.iloc[0])}" if len(h) else "—"
        cap_lines.append(f"| {lab} | {on} | {float(sub.M.max()):.4f} | {float(sub.fid.dropna().iloc[-1]):.1f} |")

# ---------- 分辨率 ----------
res_lines = []
for lab, tag in [("64px · seed21", "pilot64_blood_n250_s21"), ("64px · seed22", "pilot64_blood_n250_s22"),
                 ("128px · seed21", "pilot128_blood_n250"), ("128px · seed22", "pilot128_blood_n250_s22"),
                 ("224px · seed21 (accum4)", "pilot224_accum_blood_n250"),
                 ("224px · seed21 (batch16)", "pilot224_blood_n250")]:
    r = row(tag)
    res_lines.append(f"| {lab} | {int(r.n_pts)} | {r.M_max:.4f} | {r.sim_pk:.3f} | {r.tau:.3f} | {onset(tag)} |")
res2_lines = []
for lab, tag in [("64px", "pilot64_blood_n1000"), ("128px", "pilot128_blood_n1000"),
                 ("224px (accum4)", "pilot224_blood_n1000_a4")]:
    r = row(tag)
    res2_lines.append(f"| {lab} | {int(r.n_pts)} | {r.M_max:.4f} | {r.sim_pk:.3f} | {r.tau:.3f} | {onset(tag)} |")

text = f"""

## 21 R23：DP 剂量四档、容量的第二次复现、分辨率阶梯补齐（60.4 h 跑批）

这一轮补的是三处"结论已经写了、但证据只有一半"的地方：

1. 上一轮 DP 只跑到 1,200 轮，而**同一预算下无隐私基线的记忆率才 0.0605** —— 在这个量级上说
   "DP 把记忆率压到 0"没有说服力，因为基线本身也不高。必须四档 σ 同跑满 5,125 轮。
2. 容量效应（base=32 → 64）只有 seed 1 一条证据，且 N=250 与 N=1000 的 batch 不一致。
3. 分辨率阶梯在 N=1000 上缺两块（128px、224px），而且 224px 的"0"用的是 batch 16。

### 21.1 DP：M 全 0，但 sim_max 是干净的剂量-反应

四档同预算（5,125 轮、N=250、pathmnist、seed 1、batch 64、micro 8、C=1.0）：

| 噪声 | 点数 | M_max | sim_max 峰值 | 末点 | 峰值/τ | FID |
|---|---|---|---|---|---|---|
{dp_tbl}

对照：**无隐私基线 `pathmnist_n250_s1` 同 5,125 轮 —— M={base.M_max:.4f}、sim_max 峰值 {base.sim_pk:.3f}、
τ={base.tau:.4f}、FID {base.fid1:.1f}**。

判读三句话：

- **M 在所有档位都是 0**（512 张采样，检出限 0.2%），比上一轮"1,200 轮的 0"硬得多；
- **sim_max 随 σ 单调下降**（0.625 → 0.554），峰值/τ 从 0.85 降到 0.75 → 噪声是在**连续量上**
  把生成样本推离训练图，不是开关式的"切断"；
- **σ=1 已到 τ 的 85%**：这是四档里最逼近边界的一档，"DP 在这个预算下压得住"这句话要带
  "σ=1 时只剩 15% 余量"这个注脚。

代价：FID 从基线 123.2 抬到 **284–312（2.3–2.5×）**，且**与 σ 非单调**（284/312/296/302）——
说明瓶颈在"每步加噪 + 逐样本裁剪"这个结构本身，而不是隐私预算的大小。

裁剪范数（σ=8、1,200 轮）换三个值：

| 裁剪范数 C | 点数 | M_max | sim_max 峰值 | FID |
|---|---|---|---|---|
{clip_tbl}

→ **σ=8 时裁剪范数不改变结论**，sim_max 全在 0.56–0.58 的窄带里（噪声主导）。

限制：**只有 seed 1**。σ=1 的余量只有 15%，换 seed 是否仍不越线没有证据（正在补 seed 2）。

### 21.2 容量：第二次复现，且暴露出一个 batch 混淆

| 配置 | 首次 ≥1% | M_max | 末点 FID |
|---|---|---|---|
{chr(10).join(cap_lines)}

- **平台随容量几乎不动**（0.820 → 0.887；seed 2：0.906 → 0.885），**onset 提前**（seed 1：e800 → e640；
  seed 2：e640 → e480）。两个 seed 都是这个形态 → "**容量改变何时开始记，不改变最终记多少**"站得住。
- **但 N=1000 那一档有 batch 混淆**：base=32 当时用 batch 64（每张图看 ≈153 遍），base=48/64 用
  batch 32（≈77 遍）。所以"b48 vs b64 差 25 倍"是干净的（两者重复次数相同），而"base=32 落在中间"
  混着"多一倍重复次数"的增量。已用 batch 32 重跑 base=32 对齐（R24，见 §22）。

### 21.3 分辨率阶梯补齐：两个 seed + N=1000

同一子集（bloodmnist N=250、seed 21/22）、同 161 万像素、等效 batch 64：

| 配置 | 点数 | M_max | sim_max 峰值 | τ | 首次 ≥1% |
|---|---|---|---|---|---|
{chr(10).join(res_lines)}

N=1000（同一子集、seed 21）：

| 分辨率 | 点数 | M_max | sim_max 峰值 | τ | 首次 ≥1% |
|---|---|---|---|---|---|
{chr(10).join(res2_lines)}

- **两个 seed 完全一致**：64px 平台（1.000 / 0.969）→ 128px 只剩 3–5% → 224px 为 0。
  这不是"某个 seed 恰好不记"，是配置的效应。
- **N=1000 在 128px 上是 0**，但**不能解释成"onset 太晚"**：64px 同 N 的 M_max 只有 0.586（比 N=250 的
  1.000 低一半），说明 N 本身已经把平台压低；到 128px 平台进一步降到检出限以下，所以全程 0。
  sim_max/τ = 0.77 是这个解释的旁证。

### 21.4 三处口径，必须写进正文（否则会被当成硬伤）

1. **检出限随采样张数变**：主网格每点 512 张 → 0.2%；224px × N=1000 每点 **128 张 → 2.3%**。
   跨配置比较 M 之前必须先对齐这个下限。
2. **τ 不可跨配置比较**：N=250 的 k=125（白化维数）给 τ≈0.479，N=1000 的 k=384 给 τ≈0.315。
   能比的是 `sim_max/τ` 这类无量纲量。
3. **onset 只能给不等式**：224px 是 `>6,400`（batch16）与 `>16,000`（accum4），128px × N=1000 是 `>6,400`。

### 21.5 工程账（60.4 h、14 组，全部 rc=0）

- 组间停机 30 min；K 的预算改过三次（9,600 → 2,400 → 1,200 轮），原因写在
  `build/run_campaign3.py` 的注释里：224px × N=1000 每 epoch 有 **63 个 micro-batch**，实测
  **14.9 s/epoch**，9,600 轮要 40 h+。
- 日志断过三次：`Start-Process` 没重定向 stdout，`build/campaign3b.log` 从 00:11 起再没写过，
  K 的 ▶/✔ 行全丢 → 改成 `cmd /c ... >> log 2>&1`，并让看门狗**自动选最新的 `campaign3*.log`**。
- 被外部打断两次：11-12 23:33 关机（Kernel-Power 109，18.5 h 前的睡眠事故同一台机器，损失约 1 h）、
  11-12 夜间的睡眠（18.5 h）。两次都从最近的检查点续跑，没有数据损坏 —— **这是 checkpoint + 断点续训
  这套设计第一次真正救了场**。
- 看门狗 `--fix` 一度把 DSH 宿主一起杀掉：`wmic` 在 Win11 24H2 已移除，退路返回"所有 python 进程"，
  `taskkill /T /F` 于是连解析器一起杀。修法见 `build/watch_campaign3.py` 的 `_cim_pids` / `_kill_candidates`：
  **只杀命令行里带训练关键字的进程**，并且 `discard(os.getpid())`、`discard(os.getppid())`。
"""

p = pathlib.Path("docs/NOTES.md")
s = p.read_text(encoding="utf-8")
if "## 21 R23" in s:
    raise SystemExit("已经写过 §21，先手动确认")
p.write_text(s.rstrip() + text, encoding="utf-8")
print("写出 docs/NOTES.md §21，", len(text), "字符")