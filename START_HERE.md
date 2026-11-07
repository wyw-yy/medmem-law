# START HERE

> 课题：**医学影像生成模型的记忆化控制律：数据规模、训练轮次与记忆率**
> 一句话：用少量训练 run、每个 run 读出 32 个测量点，拟合出 `M(N, E)`（数据量、训练轮次 → 记忆率）
> 的经验控制律，再由它反解出"最迟应该在多少轮停训"的准则，并证明它比常规做法更省算力、更安全。

这份文件是入口。**先读它，再读 `PLAN.md`（方案与理由），然后按第 3 节的顺序跑。**

---

## 1. 你在做什么（30 秒版）

生成模型训练久了会把训练样本"背下来"，这是医学影像落地的隐私红线。
本项目不是提新网络，而是回答一个**可测量、可拟合、可执行**的问题：

- 记忆率 `M` 随数据量 `N` 和训练轮次 `E` 怎么变？
- 能不能写成一个只有几个参数的公式，拟合 R² ≥ 0.9？
- 用它反解出的早停点 `E*(N, ε)`，比"用 FID 平台早停"更省轮次、更少泄漏吗？

三个技术贡献（**最终按证据强度排序，与 `docs/OUTLINE.md` 一致**）：

- **C2 记忆感知早停（主贡献）**：每个检查点现场采样 → 测 `M` → 在越线前的最后一个检查点交付。
  34 个 run 上省 **66.6%** 轮次、违规 **0%**；对照组 FID 平台早停省 23.9% 但违规 79.4%，跑满违规 79.4%。
  另有一个**免测量**的替代形态：开环律 + 2σ seed 余量（`E*_safe`），省 88–94%、违规 0，代价是 FID 1.55–2.54×。
- **C1 记忆化控制律**：`M = A0·(N/250)^-α·[1-exp(-(E/E50(N))^β)]`，
  `A0=0.507`、`α=1.896`（95% CI 1.810–1.980）、`β=2.32`、`E50_0=2108`、`γ=-0.831`（CI −0.899..−0.734）。
  样本内 R²_lin 0.710 / R²_log 0.682、留一池化 0.464 —— **低于预声明的 0.9 / 0.8，如实报告并解释了原因**。
- **C3 可复现的测量协议**：τ 按 FPR=0.5% 校准（FPR 从 0.1% 到 1% 只让 M 动 4 个百分点）、
  截断白化对维数不敏感、编码器更换不改变定性结论、R 独立复算与 Python 逐位一致。

> **当前状态（2026-11-02）**：A、B 两组任务基本收工，最新的数字、结论边界、还没做的事都在
> `docs/HANDOFF.md`。先读那一份，再回来看本文件第 3 节的命令。

---

## 2. 目录结构

```
medmem-law/
  START_HERE.md        ← 你在这里
  PLAN.md              方案、文献锚点、时间表、降级阶梯（决策记录）
  RESULTS.md           结果登记簿（每张图/表背后的数字来源，由 tools/report.py 生成）
  config.py            所有常量。改参数只改这里
  data.py              MedMNIST 加载 + 嵌套子集（64px 由 build/get_hf64.py 从 224px 源转出）
  model.py             小型 DDPM UNet（13.02M 参数） + DDIM 采样
  train.py             训练一个 run（每 N 轮存点+采样+EMA，可续跑）
  metric.py            记忆率度量（编码器→截断白化→近邻→校准 τ） + FID
  law.py               三个候选律拟合 + AICc + 留一 N + 反解 E*（含删失处理）
  controller.py        四种停训策略的离线对比（A/B/C/D）
  tools/
    check_leak.py      训练前泄漏审计（**必须先跑**）
    score_run.py       对已有采样逐点评分 → results/metrics_long.csv（幂等）
    night_queue.py     夜间队列（可中断、可续跑；plan: mvp/main/deep/cross/replicate/controller/ctl）
    plot.py            出图（figs/*.png）
    report.py          生成 RESULTS.md
    law_ci.py          按 N 分层的 bootstrap，给参数与 E* 的置信区间
    transfer.py        跨数据集迁移检验（固定律形状，只重估两个尺度参数）
    safety.py          把律翻译成医院能看的安全早停表（results/safe_early_stop.csv）
    machine_profile.py 实测本机速度/显存，给排期用
    download_data.py   预下载数据
  build/               安装/转换脚本、日志、队列状态（不进 git）
    get_hf64.py        从 HF 的 224px 原始 jpg 包生成 64px npz
    night_chain.ps1    夜间串行链（主网格 → deep → cross → seed2/3）
  data/raw/            *.npz（不进 git）
  cache/ckpt/          checkpoint（fp16 + 滚动保留 4 个，不进 git）
  cache/samples/       每个测量点生成的 512 张图（不进 git）
  cache/pilot28/       28px pilot 的样本/ckpt（不进 git）
  results/             metrics_long.csv / law_fit.json / law_bootstrap.json / runs/*.json（**进 git**）
  results/pilot28/     28px pilot 的结果（归档，不进主结果）
  figs/                定稿图
  docs/                PROTOCOL / NOTES / OUTLINE / GLOSSARY / ALIGNMENT / PRIVACY_AUDIT
```

---

## 3. 上手顺序（每步都是一条命令）

```powershell
cd medmem-law
$py = ".\.venv\Scripts\python.exe"

# 0) 环境自检 + 实测速度（决定排期，必跑）
& $py -m tools.machine_profile

# 1) 数据：64px 由 224px 源转换（官方 npz 在 Zenodo，本机不可达）
& $py -m build.get_hf64 --name pathmnist --size 64
& $py -m build.get_hf64 --name bloodmnist --size 64
& $py -m build.get_hf64 --name pneumoniamnist --size 64

# 2) 泄漏审计 —— 必跑。参照集里若有训练集的重复图，τ 会被抬飞，记忆率恒为 0
& $py -m tools.check_leak

# 3) 冒烟：最小 run，验证全链路通
& $py -m train --dataset pathmnist --n 250 --seed 1 --epochs 160
& $py -m tools.score_run --run pathmnist_n250_s1

# 4) 主网格（5 档 N × 1 seed，约 3 h；夜里用 night_chain.ps1 串起来跑）
& $py -m tools.night_queue --plan main --seed 1

# 5) 评分（幂等，可随时中断重跑）
& $py -m tools.score_run --all --fid --fid-stride 4

# 6) 拟合控制律 + 置信区间 + 反解早停
& $py -m law
& $py -m tools.law_ci --boot 200

# 7) 四种停训策略对比（离线回放）
& $py -m controller

# 7b) 真·开环律早停：训练循环自己按 E*(N, eps) 停（这是"优化后的代码"）
#     例：N=250 预算 5125 轮，律算出 E*=612，训练到 612 轮自动停并存点采样
& $py -m train --dataset pathmnist --n 250 --seed 4 --epochs 5125 --stop-law --eps 0.01
& $py -m tools.night_queue --plan ctl        # 两条正式的控制器验证 run

# 7c) 跨数据集推广性（需要 cross 跑完）：固定律形状，只重估尺度参数
& $py -m tools.transfer

# 8) 出图 + 结果登记簿
& $py -m tools.plot
& $py -m tools.report
```

---

## 4. 三个必须先懂的概念

**记忆率 M**：生成的 512 张图里，与训练集某张图的"截断白化后余弦相似度" ≥ τ 的比例。
τ 不是 0.5 这种拍脑袋的数，而是这样定的：拿**从没参与训练的 test 集**图，算它们各自到训练集
的最大相似度，取 99.5% 分位。含义是"一张没被训练的图，只有 0.5% 的概率被误判为记忆"。
（64px + DINOv2 下 τ ≈ 0.735；28px 下 τ ≈ 0.672 —— 所以 τ 必须逐 N、逐数据集校准。）

**嵌套子集**：N=250 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 4000。同一个 seed 下用同一个随机排列取前缀，
所以不同 N 的曲线是"同一条数据流的截断"，N 的效应不被子集噪声污染。

**等步数预算**：记忆化发生在**优化步**这个尺度上，所以 5 档 N 给相同总步数（`STEP_BUDGET=20000`），
epoch 数按 N 反比缩放（250→5125 轮，4000→325 轮）。于是横轴 `E/N` 正好是该塌缩的量，
拟合出的 γ 就是"按 epoch 计时，记忆化起点随 N 如何推迟"的直接证据。

**数据塌缩**：主图横轴不是 E，而是 `E / N^γ`。五条不同 N 的曲线会叠成一条——这就是"控制律"的图形证据。

---

## 5. 本机（RTX 5070 Laptop 8 GB）的坑，已经踩过

| 现象 | 原因 | 处理 |
|---|---|---|
| `pip install` 打印成功后**永不退出** / 卡在 `Looking in indexes` | 本机 pip 网络栈坏了 | **别用 pip**：wheel 用 `build/get_wheels.py` + `unpack_wheels.py` 直下直解；torch 从 `datadiet-ehr\.venv` 复制 |
| Zenodo 403 / DNS 失败 | 本机不可达 | 64px 用 HF 的 224px 源本地转换（`build/get_hf64.py`）；28px 用 hf-mirror |
| HF 下载 302 后 403 | 缺 User-Agent | 请求头必须带 `User-Agent: Mozilla/5.0` |
| `~/.cache/torch` 建目录被拒 | WinError 5 | `config.py` 把 `TORCH_HOME` 指到 `build/torch_home` |
| `torch.hub.load` 即使有缓存也要联网，偶发 RemoteDisconnected | hub 先查 github | `metric._load_dinov2_offline` 直接从本地 hub 目录 import + 读权重 |
| 采样出"低对比度紫块"，M 从 0.88 掉到 0.09 | 用最后 epoch 的原始权重采样 | **必须用 EMA 权重采样**（`WeightEMA`, decay 0.999） |
| 时钟从 2 GHz 掉到 200 MHz | 笔记本功耗墙 93 W | 所有时间估算以实测 s/epoch 为准 |
| PowerShell 里带单引号的绝对路径（旧位置是 `C:\Users\yang'yu\...`）报"字符串缺少终止符" | 路径里的单引号被当成字符串边界 | 用相对路径；项目现在在 `D:\code\codetest\medmem-law`，同类坑对含空格/中文的路径照样成立 |
| `Set-Content -Encoding utf8` 给 .py 加 BOM | PS 5.1 行为 | 别用它写 py；写 .ps1 要带 BOM，否则中文乱码 |

---

## 6. 验收标准（写在开题里的那个数，怎么算）

- **拟合**：`log10(M+1e-4)` 空间 R² ≥ 0.9，且留一 N 的 R² ≥ 0.8（后者才是真的"律"，前者可能只是过拟合）。
  M=0 的点按**删失数据**处理（单侧 softplus 惩罚），不当成"M 恰好 = 1e-4"。
- **早停**：存在 `E*(N, ε)` 使 `M(E*) ≤ ε` 且 `E* < E_max`；对比策略 A/B，省轮次 ≥ 30% 且违规率更低。
- **推广**：在 bloodmnist / pneumoniamnist 上重跑 2 个 N，同一族律的 γ 在 ±50% 内。
- **稳健性**：固定阈值 M_050/M_060/M_070 与校准 τ 的结论方向一致。

---

## 7. 卡住了先看这里

1. `docs/NOTES.md`：环境与踩坑流水账（含每次失败的原因）。
2. `docs/PROTOCOL.md`：每个数字怎么来的、每个阈值的理由、设计修订 R1/R2、失败模式清单。
3. `docs/PRIVACY_AUDIT.md`：数据侧 + 模型侧审计流程与结果。
4. `docs/ALIGNMENT.md`：开题 PPT 每一格 → 代码落点 → 现状。
5. `PLAN.md` 第 7 节：降级阶梯。