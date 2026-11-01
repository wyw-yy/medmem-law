# 复现指南（从零到全部图表）

> 目标：一台干净的 Windows + 一块 ≥8 GB 显存的 NVIDIA 卡，按本文顺序执行即可复现
> `RESULTS.md` 里的每一个数字。全程**不需要联网**（除首次下数据与权重，见 §2）。
> 协议与判据见 `docs/PROTOCOL.md`；踩坑记录见 `docs/NOTES.md`；逐格核查表见 `results/verify_claims.csv`。

## 0 约定

| 项 | 值 |
|---|---|
| 项目根 | `medmem-law/`（本文所有相对路径都相对它） |
| Python | 3.12.8（**不要用 3.13**，torch cu128 轮子只到 cp312） |
| 虚拟环境 | `.venv/`（不要用系统 Python，本机 `~/.cache/torch` 写不进去） |
| GPU | RTX 5070 Laptop 8 GB（sm_120 → 必须 cu128 轮子） |
| 分辨率 | 64×64（主线）；28px 仅附录 |
| batch / 步数预算 | 64 / 每 run 20000 步（`config.STEP_BUDGET`） |

## 1 环境（**本机 pip 不可用，必须手工装**）

本机 pip 会卡在 `Looking in indexes:` 不动（不是网络问题，见 `docs/NOTES.md` §7）。
正确做法：

```powershell
python -m venv .venv

# 1) torch 系：从已有环境 robocopy（或从 download.pytorch.org 直下 cu128 轮子后解包）
robocopy ..\datadiet-ehr\.venv\Lib\site-packages .venv\Lib\site-packages torch /E
robocopy ..\datadiet-ehr\.venv\Lib\site-packages .venv\Lib\site-packages torchgen /E
# …… 以及 filelock fsspec jinja2 sympy networkx numpy pandas dateutil tzdata tqdm 及各自 dist-info
# 注意：six.py 是单文件，必须用 Copy-Item，robocopy 会把它做成目录

# 2) torchvision（9.58 MB，唯一必须新下的 torch 系轮子）
.\ .venv\Scripts\python.exe build\get_wheels.py torchvision      # 从清华源下到 build\wh\
.\ .venv\Scripts\python.exe build\unpack_wheels.py               # zipfile 直接解包，绕开 pip

# 3) 其余小依赖同样走 get_wheels/unpack_wheels
.\ .venv\Scripts\python.exe build\get_wheels.py scipy matplotlib pillow medmnist

# 4) 验证
.\ .venv\Scripts\python.exe -c "import torch,torchvision;print(torch.__version__, torch.cuda.is_available())"
```

版本锁定见 `requirements-lock.txt`（实测可跑的那一套）。

## 2 数据（64px 是本地从 224px 源转出来的）

Zenodo 在本机不可达（`getaddrinfo failed` / 403），HF 官方站超时，所以：

```powershell
# 2.1 从 hf-mirror 拉 224px 原始 jpg（必须带 User-Agent，否则 302 后 403）
.\ .venv\Scripts\python.exe -m build.get_hf64 --name pathmnist --size 64
.\ .venv\Scripts\python.exe -m build.get_hf64 --name bloodmnist --size 64
.\ .venv\Scripts\python.exe -m build.get_hf64 --name pneumoniamnist --size 64
# → data/raw/{name}_64.npz（切分与官方一致：89996/10004/7180 等）
```

`data.ensure_dataset()` 找不到文件会直接抛 `FileNotFoundError`（**不会**回落到 Zenodo，
避免静默拿到错分辨率）。

## 3 编码器与 FID 权重（离线缓存，一次即可）

```powershell
# DINOv2 ViT-S/14：torch.hub 首次会拉 github zipball + 97.8 MB 权重
# 之后 metric._load_dinov2_offline() 只读 build/torch_home/hub，完全不联网
.\ .venv\Scripts\python.exe build\probe_encoder.py
# Inception v3（FID 用）权重在首次 --fid 时下载到 build/torch_home/hub/checkpoints
```

> 若 `TORCH_HOME` 没设到 `build/torch_home`，会去写 `~/.cache/torch` → `PermissionError`。
> `config.py` 已 `os.environ.setdefault("TORCH_HOME", ...)`，裸进程调用 metric 时要自己设。

## 4 数据侧审计（必须先跑，否则 τ 会被泄漏抬高）

```powershell
.\ .venv\Scripts\python.exe -m tools.check_leak --dataset pathmnist
.\ .venv\Scripts\python.exe -m tools.check_leak --dataset bloodmnist     # 已知 5 完全 + 4 近似重复
.\ .venv\Scripts\python.exe -m tools.check_leak --dataset pneumoniamnist
```

## 5 主实验（约 3.3 h / 5 个 run，seed 1）

```powershell
.\ .venv\Scripts\python.exe -m tools.night_queue --plan main --seed 1
```

每个 run：训练到 `epochs_for(N)`（等 20000 步）→ 每 `ckpt_every` 存 fp16 权重 + EMA 采样 512 张
→ 自动评分写 `results/metrics_long.csv`。

## 6 补两个维度（deep / 跨数据集 / 复现 / 闭环）

```powershell
.\ .venv\Scripts\python.exe -m tools.night_queue --plan deep        # 大 N 加步数（N≥1000）
.\ .venv\Scripts\python.exe -m tools.night_queue --plan cross       # bloodmnist / pneumoniamnist
.\ .venv\Scripts\python.exe -m tools.night_queue --plan dense       # N=1500/3000
.\ .venv\Scripts\python.exe -m tools.night_queue --plan seeds56     # seed 5/6
.\ .venv\Scripts\python.exe -m tools.night_queue --plan ctl_online  # 在线闭环（ε=0.02, seed 7）
```

## 7 一条龙汇总（13 步，约 2 min，不用 GPU）

```powershell
.\ .venv\Scripts\python.exe build\finalize.py --skip-score
# law → law_ci → law_perseed → fpr_sens → r_verify → transfer → controller
# → safety → grid_summary → plot → audit → report
```

产物：`results/law_fit.json`、`law_bootstrap.json`、`stop_policy_*.csv`、
`safe_early_stop.csv`、`grid_summary.csv`、`transfer.csv`、`figs/*.png`、`RESULTS.md`。

## 8 附录与核查（不用 GPU）

```powershell
.\ .venv\Scripts\python.exe -m build.typology          # 记忆化类型学（要 GPU 编码，约 3 min）
.\ .venv\Scripts\python.exe -m build.parity_by_run     # 逐 run 拟合优度
.\ .venv\Scripts\python.exe -m build.parity_decompose  # 误差分解：时间 vs 幅度
.\ .venv\Scripts\python.exe -m tools.pilot_appendix    # 28px vs 64px 对照
.\ .venv\Scripts\python.exe -m build.comparison_numbers# 与文献对照的数字
.\ .venv\Scripts\python.exe -m build.verify_claims     # PPT 逐格核查（17 PASS / 3 FAIL）
.\ .venv\Scripts\python.exe -m tools.report            # 重新生成 RESULTS.md
```

## 9 R 独立复算（可选但建议：它抓到过一个真 bug）

```powershell
.\ .venv\Scripts\python.exe -m tools.r_verify
# 需要 R ≥ 4.x，脚本自动在 C:\Program Files\R\R-*\bin\Rscript.exe 里找；
# 内部会把 TMP/TEMP/TMPDIR 指到 build/r_tmp（否则 R 报 cannot create 'R_TempDir'）
```

## 10 判定标准（别把"跑通"当"达标"）

| 检查 | 命令 | 期望 |
|---|---|---|
| 结果自洽 | `python -m tools.audit` | 打印"审计通过"，exit 0 |
| PPT 逐格 | `python -m build.verify_claims` | 17 PASS；3 FAIL 是**已知且已声明**的（R²<0.9、安全表 0 行） |
| 交叉语言 | `python -m tools.r_verify` | 0 行 CHECK |
| 复算一致性 | 重跑 `finalize.py` | `law_fit.json` 参数逐位不变（拟合是确定性的多起点） |

## 11 已知不满足预声明线的地方（不要试图"修好"）

1. **R²≥0.9 未达成**（R²_lin 0.774 / R²_log 0.709）。原因：可测范围只有 N≲1500，N≥2000 的
   平台低于检出限（0.6%）；且同 N 的种子方差达 8–63 倍。换口径能抬 R²，但那是作弊，
   `docs/PROTOCOL.md` §16/§17 已声明不改。
2. **严格口径下安全表 0 行全过**（`results/safe_early_stop.csv`）。这是结论的一部分：
   小数据上"隐私合规"与"生成质量"不可兼得。
3. 28px 与 64px 的绝对 M / FID 不可比（附录 8.1 只比趋势）。