# 补充材料证明（claim → file 映射）

本文件的作用只有一个：**让审稿人（或任何第三方）能逐条检查论文里的每个数字，而不必相信作者**。
每条声明都对应一个仓库内文件、一个筛选条件、一条可执行命令。

- 机械核对表：`results/verify_claims.csv`（20 条，当前 **17 PASS / 3 FAIL**）
- 命令：`.\.venv\Scripts\python.exe -m build.verify_claims`（FAIL 时 exit 1 是设计如此）
- 全仓库哈希：`MANIFEST.sha256`（受控文件）+ `MANIFEST.data.sha256`（数据集）
- 环境快照：`docs/ENV.md`
- 数字的当前权威值：`docs/NUMBERS.md`（本表的"数字（最终）"列必须与它一致；`python -m build.doc_numbers --check` 会扫本文件）

## 1. 声明 → 证据

| # | 论文声明 | 数字（最终） | 出处（文件 / 筛选） | 命令 |
|---|---|---|---|---|
| 1 | 数据规模 7 档、每档 2–12 个 seed，共 **49 条记忆率曲线 / 1573 个测量点**（另有对照训练，累计 66 组） | 250/500/1000/1500/2000/3000/4000；主网格每档 run 数 12/11/12/2/2/2/2（合计 **43**）+ 2 个下界探针（N=128/192）+ bloodmnist/pneumoniamnist 各 2 个 = **49** | `results/metrics_long.csv`（逐行 = 一个检查点） | `python -m tools.grid_summary` |
| 2 | 记忆率 M = 近邻相似度超阈值的生成样本占比 | 每行都有 `M` 与 `tau` | 同上；τ 逐 run 校准（FPR=0.5%） | `python -m tools.audit` |
| 3 | 平台高度随数据量下降：平台 ∝ N^−1.89 | α = 1.8898 [1.8034, 1.9780] | `results/law_fit.json` → `fits.L4_amp_decay.params[1]` | `python -m law --dataset pathmnist` |
| 4 | 双时间尺度分离（γ 的 95% CI 不含 0） | γ = −0.8362 [−0.9044, −0.7395] | `results/law_bootstrap.json` | `python -m tools.law_ci --n-boot 200` |
| 5 | 控制律拟合优度 | R²_lin 0.7097 / R²_log 0.6834；可测区间（M≥0.02 的 80 点子集 R²）0.5638 / 0.4260；**只用这 80 点真重拟合 = 0.8677** | `results/law_fit.json`（`r2_*` 与 `r2_*_meas`）；重拟合见 `docs/NOTES.md` §23.1 #4 | 同 #3 |
| 6 | 律的外推能力（留一 N） | 留一池化 R² **0.4673**（分折平均 −2.0836，7 折中 6 折为负，唯一正折是内插的 N=500） | `results/law_fit.json` → `fits.L4_amp_decay.loo` | 同 #3 |
| 7 | 拟合与区间可被独立语言复现 | 逐点预测差 **0**、四个 R² 差 ≤1.1e-16；独立重拟合五参数差 **1e-5 级**（判 `match`）、残差平方和 56.773816（差 2.16e-9）、留一池化 0.467348 vs 0.467349 | `results/r_verify.csv`（状态 exact/match/consistent） | `python -m tools.r_verify` |
| 8 | 与 Bonnaire et al. 2025 的定量对齐 | onset/N ≈ 10.4–11.1（N=250–1000）；原文 τ_mem/n ≈ 300 | `docs/COMPARISON.md`、`results/fid_onset_vs_n.csv` | `python -m build.comparison_numbers` |
| 9 | 像素判据（原文 Eq.6）独立复算 | pathmnist N=250 末点 f_mem **0.412**（像素最近/次近比 p50=0.383）vs 我们 M 0.805 | `results/eq6.csv` | `python -m build.eq6 --runs ...` |
| 10 | 闭环停训 0 违规、省 69.6% | 660.7 / 2860.5 轮，违规 **0/43** | `results/stop_policy_summary.csv`（列 `violation_rate`） | `python -m controller --dataset pathmnist` |
| 11 | FID 早停不可靠 | 违规 **83.7%**（省 20.2%） | 同上（`policy=B_fid_plateau`） | 同上 |
| 12 | 开环律只给期望：省 68.8%，违规 11.6% | 635.2 轮 | 同上（`policy=C_open_loop`） | 同上 |
| 13 | 2σ 余量形态：省 88–94%、违规 0、FID 1.52–2.50× | 见表 | `results/seed_margin.csv` | `python -m build.seed_margin` |
| 14 | 严格四条件安全表：0/28 行全过（**未达标项**） | ok 22 / ok_emp 28 / ok_quality 6 / informative 11 / recommended 0 | `results/safe_early_stop.csv` | `python -m tools.safety` |
| 15 | 度量对白化维数稳健（k 不变性） | M 0.789→0.730（τ 0.748→0.128，**τ 不可跨 k 比**） | `docs/NOTES.md` §4 表 | `python -m build.probe_kdim` |
| 16 | 度量对 FPR 稳健（10× 变化只动 6.7 个百分点） | M 0.7734 / 0.8047 / 0.8398（FPR 0.1/0.5/1%） | `results/fpr_sensitivity.csv` | `python -m build.fpr_sens` |
| 17 | 换编码器：排序稳健、幅度不稳健 | 斜率 DINOv2 **−2.41** vs ResNet50 **−1.15**；Spearman 0.836–0.994 | `results/encoder_summary.csv` | `python -m build.rescore_encoder --encoder resnet50` |
| 18 | 记忆化不集中于少数样本 / 非离群 / 类别温和（**限 3 个 run**） | top1 0.08–0.10；离群差 <0.02；overrep <1.6 | `results/typology.csv` | `python -m build.typology` |
| 19 | 跨数据集：函数族可迁移、参数不可迁移 | bloodmnist free_fit R²_log **0.984** vs 共享参数 0.602（α 0.78 vs 1.89）；pneumoniamnist 0.958 vs 0.833（α 2.21 vs 1.89） | `results/transfer.csv` | `python -m tools.transfer` |
| 20 | 数据侧审计（重复 / 跨 split 泄漏） | bloodmnist：test→train **5 张**完全重复 + **4 张**近似（≥0.995），val→train 3 张完全重复；pneumoniamnist：test 干净但 **val→train 8 张**完全重复；pathmnist 全 0 | `results/leak_report.json` | `python -m tools.check_leak` |
| 21 | 分辨率：onset 随分辨率单调后移（同子集、同 seed、等效 batch 64） | 64px **e800** → 128px **e4600** → 224px **>16,000 轮仍为 0**（`sim_max` 0.29–0.40 无趋势，FID 234）→ "≥20 倍延迟"，不能写"不记忆" | `results/pilot64_blood_n250_s21.csv`、`results/pilot128_blood_n250.csv`、`results/pilot224_accum_blood_n250.csv`、`results/pilot224_blood_n1000_a4.csv` | `python -m build.summary_r23` |
| 22 | 阳性对照：度量在 N=64 上有效（欠训练才会 M=0） | 19k 步 M 0.406 / f_mem 0.664 | `results/poscontrol_64_deep.csv`（M）与 `results/eq6.csv`（f_mem） | `python -m build.poscontrol` |

## 2. 未达标的 3 条（刻意保留在表里）

| 声明 | 现状 | 为什么没有被"修好" |
|---|---|---|
| 控制律 R² ≥ 0.9 | 0.7097 / 0.6834 | 数据所能支撑的 N 范围只到 ≈1000（N≥2000 全部低于检出限）；可测区内单独重拟合能到 0.8677，仍 <0.9，且换更花哨的律在留一上更差（六候选律 + 一个形状随 N 变的探索性版本，全部记录在 `PROTOCOL.md` §14/§19/§25） |
| 同上（PPT 第二个措辞） | 同上 | 同一件事，两种写法 |
| 安全早停点表 ≥1 行可交付 | 0/28 | 四条件（最差 seed ≤ ε、经验停点不违约、FID ≤ 1.5×、ε > 2×FPR）同时成立的组合在当前模型容量与 64px 下不存在；放宽任一条就能"有结果"，用户明确要求不放宽（m03226） |

## 3. 审计链（四层，逐层可独立复核）

1. **CSV（唯一真相）**：`results/metrics_long.csv` 一行一个检查点，含 `run/dataset/n/seed/epoch/M/tau/sim_*/n_gen/fid/encoder`。任何图与表都只能由它派生。
2. **审计脚本**：`tools/audit.py` 检查 7 件事（重复点、run 内 encoder/τ 唯一、取值越界、n_gen 一致、点数下限、CSV 末轮 vs `runs/*.json`）。干预组（早停 run）豁免"点数下限"。
3. **哈希**：`tools/manifest.py` 对全部受控文件与全部数据集分别生成清单；`--verify` 重算并报告改动/缺失/清单外新增。
   实测（2026-11-18）：`MANIFEST.sha256` **287 个文件 / 24.7 MB**；文本文件在算哈希前先做 CRLF→LF 归一，所以能在 Windows CRLF 与克隆后 LF 之间复现。
4. **独立语言复算**：`R/verify_in_R.R` 用 base R 重写模型、删失罚、优化、留一、bootstrap。**逐点预测逐位一致（差 0）、四个 R² 差 ≤1.1e-16**；独立重拟合的参数落在 1e-5 级、留一池化差 9.4e-7——即"同盆最优解"，不是"参数逐位相同"（`results/r_verify.csv`）。

## 4. 不随仓库分发的东西

| 内容 | 为什么不进 git | 怎么拿到 | 怎么验 |
|---|---|---|---|
| 数据集 npz（64px，约 1.8 GB） | 体积；且应回源头获取 | `python -m build.get_hf64 --name pathmnist --size 64`（源：HF `dz-osamu/MedMNIST` 的 224px 公开 JPEG） | `MANIFEST.data.sha256` |
| 训练产生的样本/检查点（约 22 GB） | 体积 | 重跑训练（`tools/night_queue.py`） | 见 `REPRODUCE.md` §5 |
| 编码器权重（DINOv2 88 MB、Inception 109 MB） | 第三方许可 | `REPRODUCE.md` §3 的离线预取 | `docs/ENV.md` 里的 sha256 |

## 5. 版本与归档

- 论文引用的每一个数字都应绑定一个 **git tag**（`v0.1.0` 起）；正文数字与 tag 不符即为错。
- **已归档**：当前版本 `v0.1.3` → Zenodo 记录 [`10.5281/zenodo.24028919`](https://doi.org/10.5281/zenodo.24028919)（含 `THIRD_PARTY_NOTICES.md` 与数据集署名）；
  概念 DOI `10.5281/zenodo.24012026` 永远指向最新版本。
  历史版本：`v0.1.2` → `10.5281/zenodo.24012549`（修作者署名）；`v0.1.1` → `10.5281/zenodo.24012027`（首个归档版，作者为占位）。三者代码与数字完全一致。
  Zenodo 生成的是**源码 zip**（约 4 MB），不含数据集与训练产物 —— 那两类的哈希另见 `MANIFEST.data.sha256` 与 §4。
- DOI 已写回 `CITATION.cff`（`doi` = 概念 DOI + `identifiers` 列出各版本）与 `README.md` 顶部徽章。
- 提交信息前缀约定：`exp:` 跑批结果 / `law:` 拟合 / `docs:` 文档 / `fix:` 修 bug / `audit:` 审计与清单。任何"改变数字"的提交都必须在 `docs/PROTOCOL.md` 留下设计修订编号（R1…R25）。
