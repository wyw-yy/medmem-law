# 补充材料证明（claim → file 映射）

本文件的作用只有一个：**让审稿人（或任何第三方）能逐条检查论文里的每个数字，而不必相信作者**。
每条声明都对应一个仓库内文件、一个筛选条件、一条可执行命令。

- 机械核对表：`results/verify_claims.csv`（20 条，当前 **17 PASS / 3 FAIL**）
- 命令：`.\.venv\Scripts\python.exe -m build.verify_claims`（FAIL 时 exit 1 是设计如此）
- 全仓库哈希：`MANIFEST.sha256`（受控文件）+ `MANIFEST.data.sha256`（数据集）
- 环境快照：`docs/ENV.md`

## 1. 声明 → 证据

| # | 论文声明 | 数字（最终） | 出处（文件 / 筛选） | 命令 |
|---|---|---|---|---|
| 1 | 数据规模 7 档、每档 5–9 seed，共 38 个 run、1265 个测量点 | 250/500/1000/1500/2000/3000/4000 | `results/metrics_long.csv`（逐行 = 一个检查点） | `python -m tools.grid_summary` |
| 2 | 记忆率 M = 近邻相似度超阈值的生成样本占比 | 每行都有 `M` 与 `tau` | 同上；τ 逐 run 校准（FPR=0.5%） | `python -m tools.audit` |
| 3 | 平台高度随数据量下降：平台 ∝ N^−1.90 | α = 1.896 [1.810, 1.980] | `results/law_fit.json` → `fits.L4_amp_decay.params[1]` | `python -m law --dataset pathmnist` |
| 4 | 双时间尺度分离（γ 的 95% CI 不含 0） | γ = −0.8308 [−0.8991, −0.7337] | `results/law_bootstrap.json` | `python -m tools.law_ci --n-boot 200` |
| 5 | 控制律拟合优度 | R²_lin 0.7098 / R²_log 0.6824 / 可测区间 0.5568 / 0.4431 | `results/law_fit.json` | 同 #3 |
| 6 | 律的外推能力（留一 N） | 留一池化 R² 0.464 | `results/law_fit.json` → `fits.L4_amp_decay.loo.r2_pooled` | 同 #3 |
| 7 | 拟合与区间可被独立语言复现 | 残差平方和 50.669525（差 3e-9） | `results/r_verify.csv`（状态 exact/match/consistent） | `python -m tools.r_verify` |
| 8 | 与 Bonnaire et al. 2025 的定量对齐 | onset/N ≈ 10.4–11.1（N=250–1000）；原文 τ_mem/n ≈ 300 | `docs/COMPARISON.md`、`results/fid_onset_vs_n.csv` | `python -m build.comparison_numbers` |
| 9 | 像素判据（原文 Eq.6）独立复算 | 例如 N=250 末点 f_mem 0.412 vs 我们 M 0.805 | `results/eq6.csv` | `python -m build.eq6 --runs ...` |
| 10 | 闭环停训 0 违规、省 66.6% | 683.5 / 2728.7 轮，违规 0/34 | `results/stop_policy_summary.csv`（列 `violation_rate`） | `python -m controller --dataset pathmnist` |
| 11 | FID 早停不可靠 | 违规 79.4%（省 27.7%） | 同上（`policy=B_fid_plateau`） | 同上 |
| 12 | 开环律只给期望：省 64.8%，违规 11.8% | 682.1 轮 | 同上（`policy=C_open_loop`） | 同上 |
| 13 | 2σ 余量形态：省 88–94%、违规 0、FID 1.55–2.54× | 见表 | `results/seed_margin.csv` | `python -m build.seed_margin` |
| 14 | 严格四条件安全表：0/28 行全过（**未达标项**） | ok 22 / ok_emp 28 / ok_quality 5 / recommended 0 | `results/safe_early_stop.csv` | `python -m tools.safety` |
| 15 | 度量对白化维数稳健（k 不变性） | M 0.789→0.730（τ 0.748→0.128） | `docs/NOTES.md` §4 表 | `python -m build.probe_kdim` |
| 16 | 度量对 FPR 稳健（10× 变化只动 4%） | M 0.7734/0.8047/0.8398 | `results/fpr_sensitivity.csv` | `python -m build.fpr_sens` |
| 17 | 换编码器：排序稳健、幅度不稳健 | 斜率 −2.46 vs −1.13；N=1500 末点差 0.105 | `results/encoder_summary.csv` | `python -m build.rescore_encoder --encoder resnet50` |
| 18 | 记忆化不集中于少数样本 / 非离群 / 类别温和 | top1 0.08–0.10；离群差 <0.02；overrep <1.6 | `results/typology.csv` | `python -m build.typology` |
| 19 | 跨数据集：函数族可迁移、参数不可迁移 | free_fit R²_log 0.984 vs transfer 0.534 | `results/transfer.csv` | `python -m tools.transfer` |
| 20 | 数据侧审计（重复 / 跨 split 泄漏） | bloodmnist 5 完全 + 4 近似；另两集干净 | `results/leak_report.json` | `python -m tools.check_leak` |
| 21 | 分辨率实验（224px）**不确定**，不作为结论 | M=0、std 30.8、FID 116.7（30k 步） | `results/pilot224_metrics.csv` | 见 `docs/PROTOCOL.md` §23 |
| 22 | 阳性对照：度量在 N=64 上有效（欠训练才会 M=0） | 19k 步 M 0.406 / f_mem 0.664 | `results/poscontrol_64_deep.csv` | `python -m build.poscontrol` |

## 2. 未达标的 3 条（刻意保留在表里）

| 声明 | 现状 | 为什么没有被"修好" |
|---|---|---|
| 控制律 R² ≥ 0.9 | 0.7098 / 0.6824 | 数据所能支撑的 N 范围只到 ≈1000（N≥2000 全部低于检出限）；改用更花哨的律在留一上更差（六候选 + 两个探索性候选全部记录在 `PROTOCOL.md` §14/§19） |
| 同上（PPT 第二个措辞） | 同上 | 同一件事，两种写法 |
| 安全早停点表 ≥1 行可交付 | 0/28 | 四条件（最差 seed ≤ ε、经验停点不违约、FID ≤ 1.5×、ε > 2×FPR）同时成立的组合在当前模型容量与 64px 下不存在；放宽任一条就能"有结果"，用户明确要求不放宽（m03226） |

## 3. 审计链（四层，逐层可独立复核）

1. **CSV（唯一真相）**：`results/metrics_long.csv` 一行一个检查点，含 `run/dataset/n/seed/epoch/M/tau/sim_*/n_gen/fid/encoder`。任何图与表都只能由它派生。
2. **审计脚本**：`tools/audit.py` 检查 7 件事（重复点、run 内 encoder/τ 唯一、取值越界、n_gen 一致、点数下限、CSV 末轮 vs `runs/*.json`）。干预组（早停 run）豁免"点数下限"。
3. **哈希**：`tools/manifest.py` 对全部受控文件与全部数据集分别生成清单；`--verify` 重算并报告改动/缺失/清单外新增。
   实测（2026-11-06）：`git clone` 到一台干净目录后运行 `tools/manifest.py --verify` → **204 个文件哈希一致**，
   即清单能在换机器、换换行符（Windows CRLF → 克隆后 LF）的情况下复现；文本文件在算哈希前会先做 CRLF→LF 归一。
4. **独立语言复算**：`R/verify_in_R.R` 用 base R 重写模型、删失罚、优化、留一、bootstrap，与 Python 逐位对照。

## 4. 不随仓库分发的东西

| 内容 | 为什么不进 git | 怎么拿到 | 怎么验 |
|---|---|---|---|
| 数据集 npz（64px，约 1.8 GB） | 体积；且应回源头获取 | `python -m build.get_hf64 --name pathmnist --size 64`（源：HF `dz-osamu/MedMNIST` 的 224px 公开 JPEG） | `MANIFEST.data.sha256` |
| 训练产生的样本/检查点（约 22 GB） | 体积 | 重跑训练（`tools/night_queue.py`） | 见 `REPRODUCE.md` §5 |
| 编码器权重（DINOv2 88 MB、Inception 109 MB） | 第三方许可 | `REPRODUCE.md` §3 的离线预取 | `docs/ENV.md` 里的 sha256 |

## 5. 版本与归档

- 论文引用的每一个数字都应绑定一个 **git tag**（`v0.1.0` 起）；正文数字与 tag 不符即为错。
- **已归档**：`v0.1.1` → Zenodo 记录 [`10.5281/zenodo.24012027`](https://doi.org/10.5281/zenodo.24012027)（版本 DOI）；
  概念 DOI `10.5281/zenodo.24012026` 永远指向最新版本。
  Zenodo 生成的是**源码 zip**（约 4 MB），不含数据集与训练产物 —— 那两类的哈希另见 `MANIFEST.data.sha256` 与 §4。
- DOI 已写回 `CITATION.cff`（`doi` + `identifiers`）与 `README.md` 顶部徽章。
- 提交信息前缀约定：`exp:` 跑批结果 / `law:` 拟合 / `docs:` 文档 / `fix:` 修 bug / `audit:` 审计与清单。任何"改变数字"的提交都必须在 `docs/PROTOCOL.md` 留下设计修订编号（R1…R16）。