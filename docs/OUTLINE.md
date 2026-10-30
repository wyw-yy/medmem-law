# 论文提纲（v0 骨架）

> 目标刊：Medical Image Analysis / Nature Machine Intelligence（方法型）。
> 这一份不是正文，是**每个章节写什么、每张图表从哪个文件来**的清单。
> 数据/代码全部在 `medmem-law`，所有数字都能一键重跑（见 `docs/PROTOCOL.md`）。

## 一句话主张

合成医学影像的**记忆化率 M** 不是玄学，它服从一条可测量的控制律
`M(N, E)`；把这条律反解，就得到**训练该在第几个 epoch 停**的判据，
而且这个判据比"看 FID 涨不动了再停"更早、更安全。

## 三个贡献（对应三张主图）

| # | 贡献 | 主图 | 证据文件 |
|---|---|---|---|
| C1 | **记忆化控制律**：`M = M∞·(1-exp(-(E/E50)^β))`，`E50 = c·N^γ`，样本内 R²_log≥0.9 且留一 N 的 R²≥0.8 | Fig.2 数据塌缩 | `results/law_fit.json`、`figs/fig_collapse.png` |
| C2 | **记忆感知早停控制器**：从开环律（策略 C）到在线测量闭环（策略 D），比 FID 平台早停更早且违约率更低 | Fig.4 停训策略对比 | `results/stop_policy_summary.csv`、`figs/fig_stop_policies.png` |
| C3 | **可复现的测量协议**：FPR 校准的近邻记忆率（含去重、编码器锁死、EMA 采样），跨 3 个数据集验证 | Table 1 + Fig.3 | `results/metrics_long.csv`、`figs/fig_M_vs_E.png` |

**novelty 的落点（审稿人会问）**：Bonnaire et al.（NeurIPS 2025 Oral）给的是
"存在隐式动力学正则化"这个**机制**，且是在理论/玩具设定下；我们给的是
**给从业者用的、带置信区间的停机判据**，并且在真实医学影像数据集上验证——
即把"为什么会背"变成"什么时候该停"。

## 章节骨架

### Abstract（最后写，150 词）
模板：问题（合成影像可能复刻训练病例）→ 缺口（缺定量规律）→ 做法（等步数网格 × 记忆率度量 × 律拟合）→
结果（R²_log=?、γ=?±?、E* 使 M≤1% 能省 ??% 训练）→ 意义（隐私审计可落地）。

### 1. Introduction
- 钩子：医院用 GAN/扩散生成"补充"影像，法务需要回答"这张合成图会不会就是某位病人的片子"。
- 现有做法：按 FID 或固定 epoch 停 → 与隐私无关（FID 好 ≠ 不记忆；见 `figs/fig_quality.png` 的双面板）。
- 本文：把 M 变成可测、可拟合、可反解的量。

### 2. Related work
- 记忆化的机制研究：Bonnaire 2025（双时间尺度 τ_gen / τ_mem，τ_mem ∝ N）。
- 提取攻击与度量：Carlini 2023（扩散模型的提取）、Somepalli 2023（GAN 复制）、
  Deo et al. ISBI 2026（MRI 上校准过的 MI）、SSCD（自监督描述子去重）、Kynkäänniemi 2019（密度/覆盖）。
- 数据侧干预：去重/选样（本组 datadiet-ehr 经验）。

### 3. Method
3.1 **测量**：M = 生成样本中"与训练集近邻余弦相似度 > τ"的比例；
    τ 由**非成员参照集**（同数据集 test split，从未训练）对训练集 bank 的最大相似度
    的 99.5 百分位校准 → 固定假阳性率 0.5%。编码器 DINOv2-S + 白化截断到
    `k=max(32,min(d,N/2))` 维；校准前剔除与训练集近乎重复的参照图（`LEAK_COS=0.99`）。
    另报去重感知的 `M_uniq`（同近邻只算一次）以防"多采几张一样的"被误当记忆。
3.2 **实验设计**：等**步数**预算（20000 步/run），5 档 N × 3 seed；
    每 run 32 个测量点；EMA 权重采样（否则 M(E) 出假锯齿，见 `docs/NOTES.md` §1.5）。
3.3 **律拟合**：三个候选（拉伸指数 L1 / Hill L2 / 双时间尺度 erf L3），
    在 log10(M+1e-4) 空间多起点最小二乘；预声明选型判据 AICc → 留一 N R² → 可解释性。
3.4 **早停策略**：A 固定跑满（基线）、B FID 平台、C 开环律 E*(N,ε)、D 闭环在线测 M。
    判据：M_stop ≤ ε=1%、耗 epoch 显著更低、FID 不差于 B。

### 4. Experiments
- 4.1 主结果：数据塌缩图（Fig.2）—— 5 条 N 曲线叠成一条，横轴 E/N^γ。
- 4.2 律的选择与稳健性：Table（三候选 × R²_log/R²_lin/AICc/LOO）。
- 4.3 跨数据集：bloodmnist（多类别小数据）、pneumoniamnist（灰度）→ γ 是否可迁移。
- 4.4 停训策略对比：Fig.4 + Table（省多少 epoch、违约率）。
- 4.5 与 FID 早停的正面对撞：**FID 平台时 M 已经多高**（这是最有力的一句）。

### 5. Discussion / Limitations
直接抄 `docs/NOTES.md` §4：28px、MedMNIST 非原始 DICOM、13M 参数容量偏小、
FID 绝对值不可跨分辨率比、单一模型族（DDPM/UNet）。

### 6. Ethics / 可用性
合成影像的临床使用边界；本方法给的是**训练期**控制，不替代**发布前**审计；
院内数据需伦理审批 + 去标识化（本工作全部用公开数据集）。

## 表格清单（先用 CLI 生成 CSV，再排版）
1. Table 1 数据集与网格（`config.epochs_for` 预算表）。
2. Table 2 三候选律拟合指标（`results/law_fit.json`）。
3. Table 3 停训策略（`results/stop_policy_table.csv` / `_summary.csv`）。
4. Table 4 跨数据集 γ 与 R²。

## 图清单
- Fig.1 方法示意（测量管线：生成 → 编码 → 白化 → 近邻 → 阈值 → M）。可用 `archify` 出图。
- Fig.2 M(E) 与塌缩（`figs/fig_M_vs_E.png`、`figs/fig_collapse.png`）。
- Fig.3 预测 vs 实测 + 留一 N（`figs/fig_parity.png`）。
- Fig.4 质量 vs 记忆（`figs/fig_quality.png`）+ 停训策略（`figs/fig_stop_policies.png`）。
- Fig.5 E*(N) 与安全区（`figs/fig_E_star.png`）。

## 写作顺序（省时间的顺序，不是编号顺序）
1. Method（有代码就有文字）→ 2. Experiments（跑出来就能填）→ 3. Related work（读 5 篇）→
4. Intro/Discussion → 5. Abstract（最后）。