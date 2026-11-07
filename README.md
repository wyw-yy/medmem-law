# medmem-law

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24012026.svg)](https://doi.org/10.5281/zenodo.24012026)
[![CI](https://github.com/wyw-yy/medmem-law/actions/workflows/ci.yml/badge.svg)](https://github.com/wyw-yy/medmem-law/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

**Memorization control law for medical image diffusion models — data scale, training epochs, and memorization rate.**

一个可复现的实验仓库：在公开医学影像上扫描「数据规模 × 训练轮次」，测量合成样本对训练集近邻的复刻率，
拟合一条带删失处理的控制律，并给出三条可交付的停训策略。**结论以「哪条站得住、哪条没达标」的形式写清，不修饰。**

> 完整交接文档见 [`docs/HANDOFF.md`](docs/HANDOFF.md)（一页看完：数字、结论强度、剩余工作）。
> 复现步骤见 [`REPRODUCE.md`](REPRODUCE.md)；方法学见 [`docs/METHODS.md`](docs/METHODS.md)。

## 三条主结论

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | **记忆化随训练时间上升，但平台高度随数据量急剧下降**：平台 ∝ N^-1.90 [−1.98, −1.81]；N≥2000 时跑 8 万步仍 <2%（低于检出限） | 强（7 档 N，40 run，1282 个测量点） |
| 2 | **控制律只能给"期望时刻表"，给不了保证**：L4_amp_decay 的 R² 为 0.716（线性）/ 0.685（对数），**未达预声明的 0.9**；留一池化 R² 0.464，也低于 0.8 | 诚实未达标（已写进 limitation） |
| 3 | **闭环（在线测记忆率再停）是唯一能保证的形态**：34 个 run 上省 66.6% 轮次、违规 **0/34**；FID 早停省 27.7% 但违规 79.4% | 强（逐 run 判定，含交付点语义） |

## 仓库结构

```
medmem-law/
├── config.py            全局常量（路径、预算、阈值、环境变量开关）
├── data.py              数据集加载 / 嵌套子集 / 参照集
├── model.py             小型 UNet + DDPM（13.02M 参数）
├── train.py             训练循环（EMA、滚动 ckpt、--stop-law / --stop-eps 两种早停）
├── metric.py            记忆率度量（白化特征近邻 + τ 校准 + FID）
├── law.py               六个候选律的拟合、删失处理、留一、E* 反解
├── controller.py        四种停训策略的离线评估
├── tools/               评分、出图、报告、审计、清单、迁移检验
├── build/               一次性探针脚本（每个都对应 NOTES 里的一条结论）
├── R/                   独立语言的复算（verify_in_R.R，base R，无依赖）
├── results/             ★ 所有数字的原始 CSV（审计链只认这里）
├── figs/                定稿图（fig1_triptych 为主图）
└── docs/                方法、协议（含全部设计修订）、笔记、交接、补充材料映射
```

## 从零复现（三步）

```powershell
# 1. 环境（本机 pip 网络栈不可用，必须手工装 wheel；见 REPRODUCE.md §1）
python -m venv .venv ; .\.venv\Scripts\python.exe -m build.get_wheels ...

# 2. 数据（64px 由 224px 公开源本地转换，幂等）
.\.venv\Scripts\python.exe -m build.get_hf64 --name pathmnist --size 64

# 3. 一条龙：评分 → 拟合 → 区间 → 跨数据集 → 控制器 → 安全表 → 出图 → 审计 → 报告 → 清单
.\.venv\Scripts\python.exe build\finalize.py --skip-score
.\.venv\Scripts\python.exe -m tools.manifest --verify
```

## 可复现性声明（本仓库的硬约束）

- 每个数字都能追到 `results/*.csv` 的某一行，且该行由已提交的脚本产出；
- 关键拟合用 **Python + 独立 R 实现**双跑，残差平方和逐位一致（差 3e-9）；
- `MANIFEST.sha256` 覆盖全部受控文件；`MANIFEST.data.sha256` 覆盖不进 git 的数据集；
- 环境快照在 [`docs/ENV.md`](docs/ENV.md)（GPU/驱动/CUDA/包版本/离线权重哈希）；
- 数据审计（完全重复 / 近似重复 / 跨 split 泄漏）见 `results/leak_report.json`。

## 引用

见 [`CITATION.cff`](CITATION.cff)。代码 MIT，文档与图 CC BY 4.0（可改，见 `LICENSE`）。

概念 DOI：**`10.5281/zenodo.24012026`**（永远指向最新归档版）。

## 许可与署名

| 内容 | 许可 |
|---|---|
| 代码（`*.py` / `*.R` / `*.ps1`） | **MIT** — 见 [`LICENSE`](LICENSE) |
| 文档、图、结果表（`docs/`、`figs/`、`results/`） | **CC BY 4.0** |
| 数据集（MedMNIST v2 三子集） | **CC BY 4.0**，**不随本仓库分发**，只给哈希清单 `MANIFEST.data.sha256` |
| 第三方依赖与预训练权重 | 各自许可（BSD-3 / Apache-2.0 / MIT / PSF-2.0），明细见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |

本仓库**不 vendor 任何第三方源码**。图中医学图像来自 MedMNIST v2，复用这些图时须同时署名数据集与本仓库：

> MedMNIST v2 — Yang, J., Shi, R., Ni, B. *MedMNIST Classification Decathlon* (ISBI 2021) / *MedMNIST v2* (Scientific Data 2023)，CC BY 4.0（https://medmnist.com/）。本仓库做了 224px→64px 转换与子集抽样，未修改图像内容。