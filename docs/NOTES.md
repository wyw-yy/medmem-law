# 工程笔记（踩过的坑 + 为什么这么定）

> 这份文件记录「为什么代码长这样」。写论文的方法节和复现步骤时直接用。
> 最后更新：2026-10-30

## 1. 度量：记忆率 M 是怎么算的

M = 「生成的图里，有百分之多少和训练集某张图**足够像**」。三个环节都可能出错，逐个说清。

### 1.1 相似度空间：DINOv2 + 截断续白化

- 编码器：`DINOv2 ViT-S/14`（dim 384），图 resize 到 224，输出 CLS 特征再 L2 归一化。
- **必须白化，但不能全维白化**。N=250 < d=384 时协方差秩亏，`eigh` 把特征值 clip 到 1e-8
  后，零空间方向被放大 ~30 倍，白化后的余弦退化成随机数（实测 ≈0.05），τ 和 M 同时失效
  —— 第一次跑 score_run 就是这个症状：τ=0.0976，所有 M=0。
- 修法：`fit_whitening(feats, shrinkage=1e-3)`，`k = max(32, min(d, len(feats)//2))`，
  只保留最大 k 个特征方向，`W = v[:, -k:] / sqrt(w[-k:])`。
- 实测分辨率（`build/probe_metric.py`：A=非成员真图→训练集最大余弦，B=训练图微微增广的近复制品）：
  raw 特征 A 的 p99.5=0.965、B 的 p50=0.953，两者几乎重叠 → **不能直接用 raw 余弦**；
  白化后能分开，所以白化是必需的，不是可选的美化。

### 1.2 阈值 τ：每个 N 单独校准，锚定 FPR=0.5%

τ 取「参照集（官方 test 前 2000 张，**绝不参与训练**）对训练子集的最大余弦」的 99.5 百分位。
含义：在"没见过训练集"的图上，只有 0.5% 会误判成记忆。

实测（DINOv2 + 截断白化）：

| N | k | τ | 近复制品检出率 |
|---|---|---|---|
| 250 | 125 | 0.6721 | 80.0% |
| 1000 | 384 | 0.5570 | 68.8% |
| 4000 | 384 | 0.6584 | 57.8% |

τ 随 N 变化是**应该的**：bank 越大，最大余弦天然越高，所以每个 N 用各自的 τ。
校准前用 `LEAK_COS=0.99` 剔除参照集里与训练集重复的图（`tools/check_leak.py`：
bloodmnist 有 5 张完全重复 + 4 张近似，pathmnist 1 张近似，pneumoniamnist 干净）。

### 1.3 ⚠️ 编码器必须锁死（踩过的真坑）

同一条评分命令两次跑出 **τ=0.6721 和 τ=0.8203** 两个值，M 从 7.8% 直接变 0。
排查结论：那次进程里 DINOv2 加载失败，`load_encoder` **静默回退到了 ResNet50**；
用 resnet50 在同样数据上算就是 τ=0.8203。

- 现已改成：请求 dinov2 却加载失败 → **直接抛错**，不再静默回退；
  真要回退必须显式 `MEDMEM_ALLOW_ENCODER_FALLBACK=1`。
- `results/metrics_long.csv` 增加 `encoder` 列（`dinov2_s:384`），
  写入时若与本 run 已有行的编码器不一致 → 拒绝写入。
- 教训推广：**任何影响 M 量纲的开关都要落进结果表**（编码器、白化维度、FPR、去重阈值）。

### 1.4 FID

用 torchvision `inception_v3` (IMAGENET1K_V1) 自己算，real 一侧统计量缓存在
`cache/feats/fid_real_{dataset}.npz`。坑：
- 权重下载走 `torch.hub`，本机 `~/.cache/torch` **建目录被拒**（WinError 5），
  所以 `config.py` 里把 `TORCH_HOME` 指到 `build/torch_home`。
- scipy≥1.18 的 `sqrtm` 去掉了 `disp` 参数 → 用 try/except 兼容。
- 512 张样本对 2048 维协方差奇异 → `cov += ridge*tr(cov)/d*I`（ridge=1e-4）。
- 每个测量点约 15 s，所以只按 `--fid-stride` 抽样算（默认每 4 个点一次）。

### 1.5 ⚠️ 采样必须用 EMA 权重（否则 M(E) 出假锯齿）

第一轮主网格（N=250，非 EMA 权重采样）的 M(E) 曲线是：
`e2240 0.883 → e2400 0.092 → e2560 0.805 → e2720 0.199 → e3040 0.922`。

排查三步，排除了两个更"性感"的解释：

1. **不是训练发散**：那段 loss 平滑（2200–2450 段 mean 0.0082、max 0.0142）。
2. **不是重复计数**：去重感知的 `M_uniq` 与 M 几乎一致（`n_uniq` 371–511）。
3. **就是采样塌了**：坏点采出的是低对比度、接近数据集均值的紫块
   （`figs/probe_samples.png`），`sim_mean` 0.43 vs 好点 0.80，像素 std 27 vs 33。

验证：把最后 4 个 ckpt 的权重等权平均后采样，M=0.930、`sim_mean`=0.892、std 33.7
—— 回到"好点"水平。**权重平均能消除塌陷**，所以这是优化轨迹的高频噪声，
不是记忆化真的时有时无。

处置：`train.WeightEMA`（decay 0.999，带 `min(decay,(1+n)/(10+n))` 预热），
每个优化步更新，**采样一律用 EMA 权重**，`ckpt` 里同时存 `ema`/`ema_n`，
run json 记 `sample_from: "ema"`。改完后旧 run 全部清空重跑
（`build/reset_runs.py`）—— 两代采样方式的点绝不能混在一条曲线上。

**这条本身是论文里值得写的一段**：如果不做 EMA，你会很自然地报告
"记忆化随训练非单调"，而这其实是采样噪声。凡是报告 M(E) 曲线的工作都该说清
自己是不是 EMA 采样。

## 2. 本机环境坑（换机器时照着重装）

1. **pip 完全不可用**：`pip install -U pip` / 清华源装小包，全部卡在
   `Looking in indexes:`，进程 90% CPU 空转、零 IO、永不退出。网络本身没问题
   （`Invoke-WebRequest` 到 pypi/tuna/pytorch 全 200，urllib 下 36 MB wheel 秒级）。
   → 依赖全部改为 `urllib` 直下 wheel + `zipfile.extractall` 解包
   （`build/get_wheels.py`、`build/unpack_wheels.py`）。torch 直接从上一个项目的
   `.venv` robocopy 过来，省 2.86 GB 下载。
2. **zenodo.org 不可用**（DNS 解析失败；DoH 拿 IP 直连 403）→ MedMNIST 官方 64px npz
   拿不到，改用 `hf-mirror.com` 上 `albertvillanova/medmnist-v2` 的 **28px** npz
   （键名/切分与官方一致）。改分辨率只需改 `config.IMAGE_SIZE` 一行。
3. **PowerShell 路径含单引号**：`C:\Users\yang'yu\...` 直接写进命令会
   `ParserError: 字符串缺少终止符`。→ 用相对路径或 `workdir` 参数。
4. **`Set-Content -Encoding UTF8` 会加 BOM**，Python 解析中文注释时
   `SyntaxError: invalid non-printable character U+FEFF`。→ 用 Python 读写，或事后剥 BOM。
5. `six.py` 是单文件，robocopy 会把它复制成目录 → 必须用 `Copy-Item`。
6. Windows 上 `NUM_WORKERS=0`，多进程在夜间队列里容易挂住。

## 3. 实验设计：为什么按「步数」给预算，不按 epoch

记忆化发生在**优化步**这个时间尺度上（NeurIPS 2025 Oral 的 τ_mem ∝ N 就是步数单位）。
所以主网格给每档 N 相同的步数预算（`config.STEP_BUDGET=20000`），epoch 数按 N 反比缩放
（`config.epochs_for`），每个 run 恰好采 32 个测量点（`config.ckpt_every_for`）。

好处：
- 5 个 run 耗时几乎相同（≈20–30 min），15 个 run 一晚上能跑完；
- 横轴 `E/N` 正好是「每张图被看了多少遍 / batch」，是理论预测该塌缩的量；
- 拟合出的 `E50 = c·N^γ` 里，**γ≈+1 就是"按 epoch 计时会随 N 线性推迟"的直接证据**。

## 4. 已知限制（写 limitations 用）

- 主实验 28px（原因见 2.2），分辨率本身影响记忆难度；64px 源恢复后应在
  `IMAGE_SIZE` 一处改掉重跑，作为分辨率敏感性实验。
- 数据集是 MedMNIST（真实临床影像的降采样/去标识版本），**不是**原始 DICOM；
  结论外推到 PACS 级数据要加一句 caveat。
- 单卡 8 GB、UNet 13 M 参数，模型容量相对 N 偏小；容量是记忆化的一阶因素，
  所以这里的 M 绝对值偏保守，律的形状（而非绝对高度）才是主要结论。
- FID 用 Inception 在 28px 上采样后算，绝对值不可与文献中的 256px FID 比，只做横向比较。