# 第三方组件与数据来源（许可与署名）

本文件只做一件事：把本项目**用到的**第三方东西、它们的许可、以及我们是否分发它们，一次说清。

**一句话结论**：

- 本仓库**不 vendor 任何第三方源代码**（`git grep` 全文无外部版权头；`model.py` 只是"结构与某公开基线同构"的设计参照，代码为本项目自写）。
- 9 个直接 Python 依赖**全部是宽松许可**（BSD-3 / Apache-2.0 / MIT / PSF-2.0 / MPL-2.0+MIT），与本项目 MIT 兼容，可再分发、可用于商业与论文补充材料。
- **数据不分发**：MedMNIST v2 子集在运行时从公开源下载，我们只发布**哈希清单**（`MANIFEST.data.sha256`）与**由数据派生出的图**。数据本身按其 CC BY 4.0 署名（见 §3）。
- 预训练特征编码器 DINOv2（代码 + 权重）是 Apache-2.0，FID 用的 Inception-v3 权重随 torchvision 以 BSD-3 发布；两者都**不随本仓库分发**，只在本机缓存（哈希记在 `docs/ENV.md`）。

重新生成依赖表：

```powershell
.\.venv\Scripts\python.exe -m build.license_audit        # 终端表格
.\.venv\Scripts\python.exe -m build.license_audit --md   # Markdown（即 §1 的表）
```

## 1. Python 依赖（实测自 `.venv` 的 `dist-info/METADATA`）

| 包 | 版本 | 许可 | 直接依赖 | 是否随仓库分发 |
|---|---|---|---|---|
| matplotlib | 3.11.2 | PSF-2.0 (Matplotlib licence, BSD-compatible) | 是 | 否 |
| medmnist | 3.0.2 | Apache-2.0 | 是 | 否 |
| numpy | 2.5.4 | BSD-3-Clause | 是 | 否 |
| pandas | 3.0.6 | BSD-3-Clause | 是 | 否 |
| pillow | 13.0.0 | MIT-CMU (HPND) | 是 | 否 |
| scipy | 1.18.1 | BSD-3-Clause | 是 | 否 |
| torch | 2.11.0+cu128 | BSD-3-Clause | 是 | 否 |
| torchvision | 0.26.0+cu128 | BSD-3-Clause | 是 | 否 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | 是 | 否 |
| contourpy | 1.4.0 | BSD-3-Clause | 传递 | 否 |
| cycler | 0.12.1 | BSD-3-Clause（matplotlib 项目） | 传递 | 否 |
| fonttools | 4.68.0 | MIT | 传递 | 否 |
| kiwisolver | 1.5.1 | BSD | 传递 | 否 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | 传递 | 否 |
| pip | 24.3.1 | MIT | 传递 | 否 |
| pyparsing | 3.3.3 | MIT | 传递 | 否 |
| python-dateutil | 3.0.0 | Dual License (Apache-2.0 / BSD-3) | 传递 | 否 |
| six | 1.17.0 | MIT | 传递 | 否 |
| typing_extensions | 4.16.0 | PSF-2.0 | 传递 | 否 |
| tzdata | 2026.5 | Apache-2.0 | 传递 | 否 |

> 关于 **tqdm**：它是 `MPL-2.0 AND MIT` 双许可。我们只用它做进度条，**未修改其代码**，MPL-2.0 的"文件级 copyleft"因此不触发；若要更保守，也可以直接改用 `pip install tqdm` 之外的极简打印（本项目已尽量在关键日志里自带时间戳）。

## 2. 运行时非 Python 组件

| 组件 | 用途 | 许可 | 是否随仓库分发 |
|---|---|---|---|
| DINOv2 代码（`facebookresearch/dinov2`） | 记忆率度量的特征编码器 | Apache-2.0 | 否（`torch.hub` / 离线缓存） |
| DINOv2 ViT-S/14 权重 `dinov2_vits14_pretrain.pth`（88 MB） | 同上 | Apache-2.0 | 否（哈希见 `docs/ENV.md`） |
| Inception-v3 ImageNet 权重（torchvision 分发） | FID | BSD-3-Clause | 否（哈希见 `docs/ENV.md`） |
| R 4.6.1 + base R | 独立语言复算 `R/verify_in_R.R` | GPL-2.0-or-later | 否（只在本地运行，不链接、不分发） |

> R 是 **GPL**，但我们只是**调用**它跑一个独立脚本，既不链接进本项目、也不分发 R 本体或修改它的源码，因此不会把本项目"传染"成 GPL。论文里报告"用另一个语言的独立实现复算过"不涉及许可问题。

## 3. 数据：MedMNIST v2（CC BY 4.0）

| 项 | 说明 |
|---|---|
| 数据集 | MedMNIST v2 的三个 2D 子集：`pathmnist`（9 类）、`bloodmnist`（8 类）、`pneumoniamnist`（2 类） |
| 许可 | **CC BY 4.0** |
| 我们分发的 | **不分发图像**。只分发：①由数据派生出的表格/图（`results/`、`figs/`，见 LICENSE 的范围说明）；②**哈希清单** `MANIFEST.data.sha256`（证明我们训练用的就是这一份） |
| 我们怎么拿到 | Hugging Face 镜像 `dz-osamu/MedMNIST` 的 **224px 原始 JPEG**（继承 MedMNIST v2 的 CC BY 4.0），本地用 LANCZOS 转成 64px；转换脚本 `build/get_hf64.py` |
| 必须附的署名 | 见下方引用块，已同时写进 `README.md` 与 `docs/PRIVACY_AUDIT.md` |

> **Attribution（CC BY 4.0 要求）**
> MedMNIST v2 — Jiancheng Yang, Rui Shi, Bingbing Ni.
> *MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis*, ISBI 2021；
> 以及 *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*, Scientific Data 2023.
> 数据集以 CC BY 4.0 发布（https://medmnist.com/）。本仓库对其做了分辨率转换（224px → 64px）与子集抽样，**未修改图像内容**；派生结果同样按 CC BY 4.0 署名。

**关于 `figs/` 里的样本图**：图中的医学图像来自上述数据集，属 CC BY 4.0 的衍生使用。按 CC BY 4.0 要求，**复用这些图时必须同时署名数据集与本仓库**。LICENSE 里已把 `figs/` 定为 CC BY 4.0，与此一致。

**关于 `bloodmnist` 的重复样本**：该子集自身含 5 对完全重复、4 对近似重复（`results/leak_report.json`），这是数据集的问题、不是我们引入的；我们在校准 τ 时按 `LEAK_COS=0.99` 剔除，并在 `docs/PRIVACY_AUDIT.md` 里写明。这不影响数据许可，但影响"记忆化"的可解释性，审稿人问到时直接指向该文件。

## 4. 未 vendor、但"影子"存在的东西（主动声明）

| 东西 | 关系 | 许可风险 |
|---|---|---|
| `lucidrains/denoising-diffusion-pytorch` | `model.py` 的文件头注释写"与 lucidrains 基线同构但更小"。这指的是**结构设计**（eps-prediction + 线性 β + ResBlock/Attention-free UNet），代码为本项目自写，**没有复制其源码**（`git grep` 无外部版权头） | 无。该仓库本身是 MIT；即使有借鉴也不构成问题 |
| DDPM 论文（Ho et al., 2020） | 噪声调度、`q_sample`、ε-MSE 目标的**方法**来自论文 | 方法/算法不受版权保护，仅专利与论文表述受约束；我们未复制论文文字 |
| Bonnaire et al. 2025（arXiv 2505.17638） | 我们**独立重实现**了其 Eq.(6) 的像素近邻判据（`build/eq6.py`），用于交叉验证自己的度量 | 无。论文方法可自由实现；我们在 `docs/COMPARISON.md` 明确标注"判据来自该文 §2 定义，实现独立编写" |

## 5. 这张表怎么保持有效

- 依赖表由 `build/license_audit.py` **从 `.venv` 实测生成**，不是手抄；版本号与 `requirements-lock.txt`、`docs/ENV.md` 一致。
- 升级任何依赖时，重跑上面两条命令，若有许可从宽松变成 copyleft（例如某包改用 GPL 或 AGPL），CI 不会自动拦（CI 不装依赖），但本文件会被要求同步更新 —— 见 `docs/PROTOCOL.md` 的文档维护约定。
- 论文投稿时的 Data/Code availability 段落见 `docs/PUBLISH.md` §5，已包含数据集署名与"数据不分发、只给哈希"的表述。