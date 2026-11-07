# results/ 索引：每个文件是谁生成的、里面是什么、文档里哪一节在引用它

整理原则：`results/` 里**只留能被脚本重新生成的东西**。任何一个文件都应该能回答三个问题：
① 它是哪条命令生成的；② 它承载哪个结论；③ 如果数字变了，谁要跟着改。

"谁生成"一栏都是**从项目根**执行的命令。`runs/` 与 `pilot28/` 是目录。

## 一、主数据（一切数字的源头）

| 文件 | 谁生成 | 里面是什么 | 被引用处 |
|---|---|---|---|
| `metrics_long.csv` | `python -m tools.score_run --all --fid --fid-stride 4` | **唯一真相表**：每个 (run, 检查点) 一行，M 及 M_050…M_095、sim_max/p95/mean、FID | RESULTS §2/§3；报告全部数字；`law` 与 `tools.*` 的输入 |
| `runs/<run>.json` | `python -m train ...` | 逐 run 曲线（每 epoch 的 loss/lr/梯范数）、`params_M`、是否早停、`stop_rule`、`epochs_done` | RESULTS §2；`tools.report` 完成度表 |
| `pilot28/metrics_long.csv` + `pilot28/runs/` | 早期 28px pilot（`config.IMAGE_SIZE=28`） | 28px 阶段的 182 行曲线，只用于附录的分辨率对照 | RESULTS §8.1；COMPARISON |

## 二、控制律

| 文件 | 谁生成 | 里面是什么 | 被引用处 |
|---|---|---|---|
| `law_fit.json` | `python -m law --dataset pathmnist --out results/law_fit.json` | 六候选律的 AICc 排名、L4 参数、R²、留一、删失合规；参数在 `fits[best]["params"]` | RESULTS §4；ABSTRACT；ALIGNMENT 04 行 |
| `law_fit_perseed.json` | `python -m law --dataset pathmnist --per-seed ...` | 逐 run 口径（不折叠种子）的参数与 R² | RESULTS §4 |
| `law_bootstrap.json` | `python -m tools.law_ci --n-boot 200` | 200 次分层 bootstrap 的 α/γ/E50/β 区间 + E*(ε) 区间（**与点估计同一张输入表**） | RESULTS §4；ALIGNMENT 04② |
| `r_fit_points.csv` | `python -m build.export_for_R` | 314/364 个拟合点（pathmnist + 种子均值 + 排除干预组），含 Python 预测值 `pred_py` | R 复算的输入 |
| `r_params.csv` | 同上 | Python 的 L4 参数，供 R 逐位比对 | R 复算 |
| `r_summary.csv` | 同上 | Python 的四个 R²、留一、α/γ 区间，供 R 自动对照 | R 复算 |
| `r_verify.csv` | `python -m tools.r_verify`（内部调 `Rscript R/verify_in_R.R`） | A–F 六组检查的逐项差异与状态（`exact`/`match`/`consistent`/`CHECK`） | RESULTS §3.5；CI 的 `r-recheck` job |

## 三、停训策略与安全表

| 文件 | 谁生成 | 里面是什么 | 被引用处 |
|---|---|---|---|
| `stop_policy_table.csv` | `python -m controller` | 每个 run × 四策略的停点、轮次、省轮%、是否合规 | RESULTS §6 |
| `stop_policy_summary.csv` | 同上 | 四策略汇总（平均轮次、省轮%、M_stop、违规率） | RESULTS §6；fig1(c)；OUTLINE C2 行 |
| `safe_early_stop.csv` | `python -m tools.safety` | 安全表：最差 seed + 假阳性地板 + 质量口径，逐 (N, ε, 质量上限) 一行，四条件全过才 `recommended=Y` | RESULTS §7；ALIGNMENT 04②；ABSTRACT |
| `seed_margin.csv` | `python -m build.seed_margin` | 免测量的 2σ 开环形态：σ_log、E*_law、E*_safe、省轮%、各 seed 是否合规 | RESULTS §5.2；OUTLINE C2′ 行 |

## 四、稳健性与交叉验证

| 文件 | 谁生成 | 里面是什么 | 被引用处 |
|---|---|---|---|
| `fpr_sensitivity.csv` | `python -m build.fpr_sens` | FPR 0.1/0.5/1% 下的 τ 与 M | RESULTS §3.4 |
| `transfer.csv` | `python -m tools.transfer` | bloodmnist / pneumoniamnist 的逐档对照与同族律 γ | RESULTS §3.3；ALIGNMENT 偏差 4 |
| `encoder_resnet50.csv` | `python -m build.rescore_encoder --encoder resnet50 --runs ...` | 用 ResNet50 重算的逐点 M（**独立文件，绝不覆盖主表**） | RESULTS §3.6 |
| `encoder_compare.csv` | `python -m build.encoder_compare` | 两编码器逐点差、Spearman、末点差 | RESULTS §3.6 |
| `encoder_summary.csv` | 同上 | 三条判据的通过/未通过汇总 | RESULTS §3.6；CI 不做（依赖缓存样本） |
| `probe_kdim.json` | `python -m build.probe_kdim` | 白化截断维 k 的 τ/M 不变性 | RESULTS §3.2 |
| `uncensored_fit.json` | `python -m build.probe_uncensored_fit` | 只用已检出点的重拟合（排除删失加权的影响） | RESULTS §3.3 |
| `threshold_law.csv` | `python -m build.probe_threshold_law` | 换记忆率定义后的重拟合 | RESULTS §3.1 |
| `fid_onset_vs_n.csv` | `python -m build.comparison_numbers` | 等步数首检查点 FID vs N、onset 步数斜率 | COMPARISON.md |
| `eq6.csv` | `python -m build.eq6` | 原文 Eq.(6) 像素判据与我们的特征判据的并列值 | PROTOCOL §21–24；RESULTS §3.1 |
| `pilot_vs_main.csv` | `python -m tools.pilot_appendix` | 28px vs 64px 同 N 同轮次对照 | RESULTS §8.1 |

## 五、诊断（写进 NOTES 的那些"为什么"）

| 文件 | 谁生成 | 里面是什么 | 被引用处 |
|---|---|---|---|
| `grid_summary.csv` | `python -m tools.grid_summary` | 逐 (dataset, N, seed) 的末点 M、平台、onset 轮次/步数、FID | fig1(a)(b)；NOTES §8/§10/§11 |
| `parity_by_run.csv` | `python -m build.parity_by_run` | 逐 run 的拟合优度与 bias（哪些 run 真的"拟合失败"） | NOTES §10 |
| `parity_decompose.csv` | `python -m build.parity_decompose` | 误差分解：只放时间轴 vs 只放幅度（79% 在幅度） | NOTES §10 |
| `typology.csv` / `typology_targets.csv` / `typology_classes.csv` | `python -m build.typology` | 三个假说的检验结果、被命中训练图清单、类别分布 | RESULTS §8.4 |
| `leak_report.json` | `python -m tools.check_leak` | 数据集内部近重复自查（bloodmnist 5 对完全重复等） | PRIVACY_AUDIT §2；PUBLISH 的 Ethics 段 |
| `poscontrol_64.csv` / `poscontrol_64_deep.csv` / `poscontrol_224.csv` | `python -m build.poscontrol` | 阳性对照的三次尝试（64px 短/长、224px） | PROTOCOL §22–24 |
| `pilot224_metrics.csv` | `python -m build.pilot224` | B9 的 224px pilot 结果（M=0，**判 inconclusive**） | PROTOCOL §20；RESULTS §8.2 |
| `pilot224_blood_n250.csv` / `pilot224_blood_n1000.csv` | `python -m build.run_tonight` 里的 `tools.score_run --out` | 224px 决定性档：bloodmnist N=250 / N=1000，与 64px 同 N 对照 | PROTOCOL §24.5 R17；RESULTS §8.3 |
| `verify_claims.csv` | `python -m build.verify_claims` | PPT 逐格核查：20 条断言的 PASS/FAIL 与证据行号 | RESULTS §8.3；CI 的 `gate` job |

## 六、元信息（发布与可复现）

| 文件 | 谁生成 | 里面是什么 | 被引用处 |
|---|---|---|---|
| `manifest.csv` | `python -m tools.manifest` | 每个进 git 文件的 sha256 与字节数（`MANIFEST.sha256` 的同源 CSV） | SUPPLEMENT §5；CI |
| `data_manifest.csv` | `python -m tools.manifest --data` | 数据集 npz 的 sha256（数据不进 git，但哈希留证） | REPRODUCE §4 |
| `env_snapshot.json` | `python -m tools.env_snapshot` | 机器与依赖版本快照（`docs/ENV.md` 的同源 JSON） | SUPPLEMENT §5 |

## 约定

- **不要手工改这里的任何文件**。数字不对就改脚本重跑，否则下一次 `finalize` 会覆盖，且审计链会断。
- 新增实验产物请同时更新本表（一行），否则半年后没人知道那个 csv 是哪来的。
- 目录型产物：`runs/`（训练曲线）、`pilot28/`（旧分辨率）、`../figs/`（图）、`../build/wh/`（wheel 缓存，不进 git）。