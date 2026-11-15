# 许可范围（许可文件之外的补充说明）

`LICENSE` 是标准的 MIT 文本，覆盖本仓库的**代码**。这份文件把其余几类内容的许可讲清楚，
免得引用时含糊。

## 1. 代码 —— MIT

`*.py`、`*.ps1`、`*.R`（`data.py`、`model.py`、`train.py`、`train_dp.py`、`metric.py`、
`law.py`、`controller.py`、`tools/`、`build/`、`R/`）适用 `LICENSE` 的 MIT 条款。

## 2. 文档、图、结果表 —— CC BY 4.0

`docs/`、`figs/`、`results/` 里的文字、图与结果表，即论文的补充材料本身，
采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
允许转载与再加工，条件是署名本仓库并注明 commit hash。

## 3. 数据集 —— 不随本仓库分发

MedMNIST v2 的子集在运行时从公开源下载（Hugging Face 镜像 `dz-osamu/MedMNIST`，
224px JPEG，本地降采样到 64px）。适用其自身许可（MedMNIST v2：CC BY 4.0）。

`MANIFEST.data.sha256` 记录了训练所用文件的哈希，第三方可以据此证明自己用的是同一批数据。

## 4. 模型权重 —— MIT，但有一个刻意的例外

本仓库训练出的权重随代码一并按 MIT 发布。**唯一的例外**是：
如果某个权重能被证明复现了训练图像（这正是本工作研究的现象），
它的再分发等于分发那些图像，因此不在 MIT 覆盖范围内。
相关分析与判据见 `docs/PRIVACY_AUDIT.md`。

需要权重时请自己按 `REPRODUCE.md` 重训（单卡约 30–60 分钟一个点），
或直接引用 `results/` 里的测量结果。

## 5. 第三方组件 —— 各自许可

依赖包、预训练编码器权重、数据集都不在上述范围内，各自保留原许可，
详见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。本仓库不内置任何第三方源代码。