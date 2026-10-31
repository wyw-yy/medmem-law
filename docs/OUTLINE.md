# 论文提纲（v1：按实测结果改写）

> 目标刊：Medical Image Analysis / Nature Machine Intelligence（方法型）。
> 这是"每章写什么 / 每张图从哪个文件来 / 哪个数字已经定稿"的清单，不是正文。
> 所有数字都来自 `results/`，可用 `build/finalize.py` 一键重跑（见 `docs/PROTOCOL.md`）。

## 一句话主张（v1 修正版）

合成医学影像的**记忆率 M** 可以用一个校准过的近邻度量测出来，并且随**数据规模 N** 与**训练轮次 E**
按一条幅度衰减的拉伸指数律演化：**数据越多，平台越低（∝ N^-2.0）、发作越晚（步数坐标 τ ∝ N^0.34）**。
把这条律反解就得到停机判据 E*(N, ε)；但实测的**种子间发作时间差约 20 倍**，所以单靠开环律兜不住每一条轨迹 ——
**律给规划，闭环给兜底**，这是本文两个贡献的关系。

## 三个贡献（按证据强度排序）

| # | 贡献 | 证据 | 状态 |
|---|---|---|---|
| C1 | **记忆化控制律**：`M = A0·(N/250)^-α · [1-exp(-(E/E50(N))^β)]`，`E50 = E50_0·(N/250)^γ`；α≈1.97、γ≈−0.69（epoch）/ +0.31（步数） | `results/law_fit.json`、`figs/fig_M_vs_E.png`、`figs/fig_amplitude.png` | 方向稳，拟合优度未达预声明 0.9（见下） |
| C2 | **记忆感知早停 + 闭环兜底**：固定时刻表会因种子方差违约（实测 1/2 例），在线测 M 再停的违规率可压到 0 | `results/stop_policy_summary.csv`、`figs/fig_stop_policies.png`、`RESULTS.md` §5 | 强（这就是"为什么不能用固定公式"的实证） |
| C3 | **可复现的测量协议**：FPR 校准的 τ、检出限、白化维不变性、EMA 采样 | `results/metrics_long.csv`、`docs/PRIVACY_AUDIT.md`、`docs/NOTES.md` §1.5/§2 | 强（三个数据集都跑通） |

## 必须诚实写出来的四件事（审稿人一定会问）

1. **R² 没有 0.9**：预声明是 R²_log ≥ 0.9 且留一池化 ≥ 0.8；实测（种子均值口径）R²_log 0.756 / R²_lin 0.848 /
   留一池化 0.473（逐 run 口径 0.666）。单 seed 拟合能到 R²_lin 0.90，但那是运气好的那条曲线。已试 7 种函数形式（含"形状随 N 变"
   的 6 参数版，参数撞边界被否），瓶颈是**可测范围只覆盖 N ≤ 1000**。
2. **γ 的符号与坐标要讲清**：epoch 坐标 γ≈−0.69（大 N 的曲线在 epoch 轴上"看起来更早起跳"是假象，
   因为一个大 epoch 里步数更多）；换到**优化步**坐标 τ ∝ N^+0.34，与 Bonnaire et al. 的 τ_mem ∝ N 同向但更平。
3. **度量有地板**：τ 按 FPR=0.5% 校准，一个都没记住时也有约 0.5% 的样本越线（512 张里 2–3 张）。
   ε ≤ 1% 的档位结论强度有限，交付档位从 ε=2% 起。
4. **早停有质量代价**：小 N 档早停后 FID 是跑满时的 2–3 倍（`figs/fig_quality.png`）；
   只有 N=1000、ε=2% 这一档同时满足隐私与质量（E*=603，省 75%，FID 1.10×）。

## 章节骨架

### Abstract（最后写，150 词）
问题（合成影像可能复刻训练病例）→ 缺口（缺定量、可校准的规律）→ 做法（等步数网格 5 档 N × 3 seed，
每 20000 步一点，校准 FPR 的近邻记忆率）→ 结果（α≈1.97、γ≈−0.69、E* 可使 M≤2% 且省 75% 训练）→
限制（可测范围 N≤1000；固定时刻表需闭环兜底）。

### 1. Introduction
- 钩子：医院用扩散模型"补充"影像，法务要回答"这张合成图会不会就是某位病人的片子"。
- 现有做法：按 FID 或固定 epoch 停 → 与隐私无关（FID 好 ≠ 不记忆，见 `figs/fig_quality.png` 双面板）。
- 本文：把 M 变成可测、可拟合、可反解的量，并给出"什么时候必须用闭环"的判据。

### 2. Related work
- 机制：Bonnaire et al. 2025（NeurIPS Oral；τ_gen 与 N 无关、τ_mem ∝ N 的双时间尺度）。
- 攻击与度量：Carlini 2023、Somepalli 2023、Deo et al. ISBI 2026（MRI 上校准过的 MI）、
  SSCD（自监督描述子）、Kynkäänniemi 2019（密度/覆盖）。
- 数据侧干预：去重/选样（本组 datadiet-ehr 经验）。

### 3. Method
- 3.1 **测量**：M = 与训练集近邻余弦相似度 > τ 的生成样本比例；τ 由非成员参照集（test split）
  的最大相似度 99.5 百分位校准（FPR=0.5%）；DINOv2-S + 白化截断 `k=max(32,min(d,N/2))`；
  另报去重感知 `M_uniq` 与固定阈值 `M_050…M_095`（回答"结论是不是靠调阈值"）。
- 3.2 **实验设计**：**等优化步预算**（20000 步/run，epoch 按 N 反比缩放），5 档 N × 3 seed，
  每 run 32 个测量点；EMA 权重采样（否则 M(E) 出假锯齿，`docs/NOTES.md` §1.5）。
- 3.3 **律拟合**：6 个预声明候选（L1 拉伸指数 / L2 Hill / L3 双时间尺度 / L4 幅度衰减 / L5 幅度衰减·γ=0 /
  L6 幅度指数衰减），删失数据单侧惩罚（M=0 的点只进罚项不进 R²），多起点 + 留一 N（池化）。
- 3.4 **早停策略**：A 固定跑满（基线）、B FID 平台、C 开环律 E*(N,ε)、D 闭环在线测 M；
  判据 = M_stop ≤ ε（**最差 seed**）、省 epoch、FID 相对代价 ≤ 1.5。

### 4. Experiments
- 4.1 主结果：M(E) 五条曲线（`fig_M_vs_E.png`）+ 平台高度随 N 的衰减（`fig_amplitude.png`）。
- 4.2 律的选择与稳健性：6 候选表（R²_log/R²_lin/可测区间 R²/AICc/留一池化）+ 阈值扫描 + 白化维不变性。
- 4.3 跨数据集：`fig_cross_dataset.png` + `results/transfer.csv`
  （**函数族可迁移、参数不可迁移**：bloodmnist 自由拟合 R²_log 0.98，但 α 0.78 vs pathmnist 2.06）。
- 4.4 停训策略对比：`fig_stop_policies.png` + `RESULTS.md` §5（最差 seed 口径）。
- 4.5 安全早停交付表：`results/safe_early_stop.csv`（每行都标 ε 是否高于假阳性地板）。

### 5. Discussion
- 临床落法：三个阶段 —— ①小规模标定（每档 N 跑 1 条曲线）②按律规划 ③上线后用闭环守卫。
- 为什么"加数据"比"调停点"更有效：平台高度 ∝ N^-2.0，而停点只挪动时间轴。
- 限制：28px→64px 已升级但仍是 MedMNIST 而非 DICOM；13M 参数容量偏小；只测图像近邻（不是成员推断攻击）。

### 6. Limitations & Ethics
- 复刻判定是"度量"而非"证明"：相似 ≠ 同一位病人，需结合院内 DICOM 复核。
- 院内数据（伦理审批后）留作 future work：`docs/PRIVACY_AUDIT.md` 给了五步审计流程。

## 图表 → 文件对照

| 论文图 | 文件 | 说明 |
|---|---|---|
| Fig.1 概念图 | — | 手绘：双时间尺度 + 我们的网格 |
| Fig.2 M(E) 五条曲线 | `figs/fig_M_vs_E.png` | 主结果，symlog 纵轴 + 检出限线 |
| Fig.3 平台随 N 衰减 | `figs/fig_amplitude.png` | α≈2.0 的直接证据 |
| Fig.4 质量-隐私权衡 | `figs/fig_quality.png` | FID 与 M 双面板 |
| Fig.5 跨数据集 | `figs/fig_cross_dataset.png` | bloodmnist 记忆化更早更强 |
| Fig.6 停训策略 | `figs/fig_stop_policies.png` | A/B/C/D 的耗轮与违规率 |
| Fig.7 E*(N,ε) | `figs/fig_E_star.png` | 交付曲线 |
| Table 1 候选律 | `RESULTS.md` §2 | 6 行 |
| Table 2 网格总览 | `results/grid_summary.csv` | 每 run 的 τ/末点 M/onset |
| Table 3 早停建议 | `results/safe_early_stop.csv` | 交付物 |
| Table 4 迁移检验 | `results/transfer.csv` | transfer vs free_fit |

## 写作顺序
Method（3.1–3.4）→ Experiments（4.1, 4.4）→ Related work → Intro → Discussion/Abstract。