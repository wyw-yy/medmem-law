# 运行环境（自动生成，勿手改）

- 生成时间：`2026-11-06T19:55:59.2931439+08:00`
- 系统：Windows 11 10.0.26200（AMD64，24 逻辑核）
- Python：`3.12.8`（`D:\code\codetest\medmem-law\.venv\Scripts\python.exe`）
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU；驱动 `573.22`；显存 8151 MiB；compute capability 12.0；torch 内置 CUDA `12.8`；cuDNN `91900`
- R：`Rscript (R) version 4.6.1 (2026-06-24)`

## 关键包版本

| 包 | 版本 |
|---|---|
| torch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| numpy | `2.5.4` |
| pandas | `3.0.6` |
| scipy | `1.18.1` |
| matplotlib | `3.11.2` |
| pillow | `13.0.0` |
| tqdm | `4.70.1` |
| medmnist | `3.0.2` |
| packaging | `26.3` |

## 离线权重（不随仓库分发，需按 REPRODUCE.md 预取）

| 文件 | 大小 / sha256 |
|---|---|
| `dinov2_vits14_pretrain.pth` | 88.3 MB sha256=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 |
| `inception_v3_google-0cc3c7bd.pth` | 108.9 MB sha256=0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef |
| `resnet50-11ad3fa6.pth` | 102.5 MB sha256=11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca |

> 这台机器没有可用 GPU 时：`TORCH_HOME` 与离线权重仍必须存在，否则 `metric.load_encoder` 会以 fail-loud 方式报错（这是刻意的，禁止静默回退到 ResNet50 —— 两者的 τ 不同，会让 M 变成 0）。
