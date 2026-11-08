"""build/ 目录索引：这 42 个脚本分别是干什么的、要不要复现、进不进论文。

为什么保留它们：每个"结论"背后至少有一次"先怀疑、再测量"的过程。删掉脚本，那些判断
（比如"必须用 EMA 采样""tau 不能跨编码器比""白化要截断"）就变成无从复核的口头经验。

标 **P** 的进论文（作为方法或附录的支撑），标 **D** 的只是调试过程，保留但不必复现。
本文件在 2026-11-07 项目整理时重写过一次：已应用的一次性补丁脚本（`patch_*.py`、
`append_r16.py`、`sync_docs_b11.py`）和纯调试打印脚本被删除（git 历史里仍可找回）。
"""

# build/ 脚本索引

| 脚本 | 用途 | 进论文 | 复现需要 |
|---|---|---|---|
| **环境装配** | | | |
| `get_wheels.py` | 从镜像解析并下载纯 python 轮子到 `build/wh/`（pip 在本机不可用） | D | 装环境时 |
| `unpack_wheels.py` | `zipfile.extractall` 直接解包轮子到 site-packages，绕开 pip | D | 装环境时 |
| `fetch.py` | 早期 urllib 下载实验（Range 续传原型） | D | 不必 |
| `strip_bom.py` | 剥掉 .py 的 UTF-8 BOM（PowerShell 写文件会加） | D | 按需 |
| **数据** | | | |
| `get_hf64.py` | 从 hf-mirror 拉 224px 原图 → LANCZOS 缩到目标分辨率 → 写 npz；含 416 判完成与 jsonl 去重 | **P** | 是（§2） |
| `check224.py` | 核对 224px 产物的切分尺寸与重复行（发现过 jsonl 整段重复） | **P** | 是（数据审计） |
| `bench_sample224.py` | 224px 的采样耗时与显存基准（决定 batch=16） | **P** | 是（预算表） |
| `test_law.py` | 合成数据自检：真值 γ=1 能否被复原（还原到 0.985） | **P** | 是（离线，10 s） |
| **度量探针（记忆率 M 的可信度）** | | | |
| `probe_metric.py` | raw/全维白化/截断白化的可分性对比（证明必须白化且必须截断） | **P** | 是（附录） |
| `probe_tau.py` | 逐 N 的 τ 与近复制品检出率 | **P** | 是（附录） |
| `probe_kdim.py` | 白化截断维 k 的不变性（τ 变 6 倍而 M 只动 6%，决定性） | **P** | 是（RESULTS §3.2） |
| `probe_encoder.py` | DINOv2 vs ResNet50 的 τ/dim 对照（混用会让 M 归零） | **P** | 是（附录） |
| `fpr_sens.py` | FPR 0.1%/0.5%/1% 下 τ 与 M 的敏感性（M 只动 4 个百分点） | **P** | 是（RESULTS §3.4） |
| `backfill_thr.py` | 给旧行补固定阈值列（M_050…M_095） | D | 不必 |
| **训练稳定性探针** | | | |
| `probe_blank.py` | 采样塌陷诊断（像素 std / flat 图计数） | **P** | 是（NOTES §1.5） |
| `probe_ema.py` | EMA/权重平均能否消除塌陷（M 0.66→0.93） | **P** | 是（NOTES §1.5） |
| **律的探针** | | | |
| `probe_law_variants.py` | 6 种律形式对比（证明瓶颈在数据覆盖不在函数形式） | **P** | 是（PROTOCOL §10） |
| `probe_uncensored_fit.py` | 只用已检出点重拟合（确认不是删失加权的问题） | **P** | 是（RESULTS §3.3） |
| `probe_threshold_law.py` | 换记忆率定义重拟合（校准 M 口径 R² 0.737，固定阈值 0.5/0.6 同向） | **P** | 是（RESULTS §3.1） |
| `probe_beta_n.py` | 让形状参数随 N 变（6 参数）——被 AICc/可辨识性否决，记录在案 | **P** | 是（NOTES §？） |
| `diag_law_by_n.py` | 逐档 bias（发现平台衰减不是单一幂律） | **P** | 是（NOTES §10） |
| `comparison_numbers.py` | 与文献对照要用的数字（首个 FID vs N、onset 斜率） | **P** | 是（COMPARISON.md） |
| `parity_by_run.py` | 逐 run 拟合优度（哪些 run 真的"拟合失败"） | **P** | 是（NOTES §10） |
| `parity_decompose.py` | 误差分解：只放开时间 vs 只放开幅度（79% 在幅度） | **P** | 是（NOTES §10） |
| `eq6.py` | 独立实现原文 Eq.(6) 的像素最近/次近比判据，与特征判据并列 | **P** | 是（PROTOCOL §21-24） |
| `poscontrol.py` | 阳性对照：N=64 跑 19k 步看 M 能不能动（能，0.406） | **P** | 是（PROTOCOL §24） |
| `seed_margin.py` | 免测量的 2σ 开环形态：E*_safe 与逐 seed 合规判定 | **P** | 是（RESULTS §5.2） |
| `typology.py` | 被复刻的是哪类训练样本（H1 少数派 / H2 离群 / H3 类别偏斜） | **P** | 是（RESULTS §8.2） |
| `verify_claims.py` | PPT 逐格核查表（从原始产物重算，17 PASS / 3 FAIL） | **P** | 是（RESULTS §8.3） |
| `export_for_R.py` | 导出给 base R 的表（拟合点、参数、区间） | **P** | 是（R 复算） |
| `encoder_compare.py` | 两编码器逐点对照表 + 三条稳健性判据 | **P** | 是（RESULTS §3.6） |
| `rescore_encoder.py` | 复用缓存样本、换编码器重算 M（写独立 CSV，绝不覆盖主表） | **P** | 是（RESULTS §3.6） |
| `license_audit.py` | 扫 `.venv` 的 dist-info 生成依赖许可表 | **P** | 是（THIRD_PARTY_NOTICES） |
| **流程与运维** | | | |
| `finalize.py` | 一条龙 14 步汇总（law → … → report → doc_numbers） | **P** | 是（§7） |
| `run_tonight.py` | 今晚跑批的顺序驱动（单实例锁，按分辨率分组避免抢显存） | **P** | 按需 |
| `night_queue.py`（在 `tools/`） | 队列：按 plan 跑训练+评分，断点续跑 | **P** | 是（§5/§6） |
| `wait_gpu.py` | 对称显存守卫（两边都等对方跑完再抢卡） | D | 按需 |
| `pilot224.py` | 224px pilot 的训练脚本（B9 用它跑出 inconclusive） | **P** | 是（PROTOCOL §20） |
| `run_tonight.py` | 顺序跑多档训练+评分（224px 与 64px 各自独立进程 env；单实例锁 `build/tonight.lock`） | **P** | 是（§11） |
| `run_campaign.py` | 第三轮跑批：224px accum4 / 128px accum2 / seed 10-12，轮间停机 12 min（PROTOCOL §24.8 R20） | **P** | 是 |
| `make_blood128.py` | 从 224px 源降采样出 `bloodmnist_128.npz`（分辨率阶梯的中间点） | **P** | 一次性 |
| `push_via_ip.py` | `github.com:443` 被丢包时，用可达 IP + Host 头 + gh token 推 master（不改 hosts） | D | 网络异常时 |
| `wait_then_extend224.py` | 等今晚链条释放锁后，把 224px 那档续训到 `E=6400` 并重评分 | D | 按需 |
| `wait_then_control_extend.py` | 链条结束后先补 64px **同 seed 同子集**对照（`--tag ..._s21_64`），再续训 224px | **P** | 是（NOTES §16.5） |
| `check224.py` | 核对 224 npz 的切分尺寸与重复行（抓出过"训练集翻倍"） | **P** | 是（§5） |
| `doc_numbers.py` | 从 `results/` 现算 `docs/NUMBERS.md`；`--check` 扫正文里与权威值不一致的 α/γ/β/A₀/R² | **P** | 是（CI 用） |
| `resample.py` | 对已有 ckpt 统一重采样（换样本数） | D | 按需 |
| `reset_runs.py` | 清某个 run 的全部产物与队列状态 | D | 按需 |

## 目录约定

- `build/` 里 `*.py` 与 `*.ps1` 进 git（复现脚本要留），其余（日志、wheel 缓存、torch_home、
  `r_tmp`、临时 json、`tonight.lock`）都在 `.gitignore` 里。
- 探针脚本一律从项目根以 `python -m build.<name>` 运行（`__file__` 里插了 `parent.parent` 到 sys.path）。
- 脚本之间**不要**互相 import（除 `config/data/metric/law` 四个主模块），避免改动一个影响一片。
