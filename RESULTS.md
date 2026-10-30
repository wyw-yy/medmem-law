# 实验结果总览（自动生成）

> 生成时间 2026-10-30 14:17:32 · 由 `python -m tools.report` 从 `results/` 汇总。
> 数字的出处见每节标注；协议与判据见 `docs/PROTOCOL.md`，踩坑记录见 `docs/NOTES.md`。

## 1. 主网格完成度

| N | seed | 测量点 | 最后 epoch / 预算 | τ | M_max | FID_min | 状态 |
|---|---|---|---|---|---|---|---|
| 250 | 1 | 33 | 5125 / 5125 | 0.6721 | 0.957 | 70.7 | ✅ 完成 |

- 总测量点 **33**，覆盖 **1** 个 run，数据集：pathmnist。
- 编码器：dinov2_s:384。

## 2. 控制律拟合

`results/law_fit.json` 还不存在 —— 主网格跑完 `python -m law` 生成。

## 3. 停训策略对比

`results/stop_policy_summary.csv` 还不存在 —— `python -m controller` 生成。

## 4. 数据审计（跨 split 泄漏）

| 数据集 | test→train 完全重复 | 近似重复(≥0.995) | val→train 完全重复 | p99 缩略图相似 | 判定 |
|---|---|---|---|---|---|
| pathmnist | 0 | 1 | 0 | 0.927 | LEAK |
| bloodmnist | 5 | 4 | 3 | 0.979 | LEAK |
| pneumoniamnist | 0 | 0 | 8 | 0.977 | OK |

- 校准 τ 时用 `LEAK_COS=0.99` 把参照集里这些重复图剔除，否则 τ 会被抬高、M 恒为 0。

## 5. 图

- `figs/fig_M_vs_E.png`
- `figs/fig_quality.png`
- `figs/probe_samples.png`

## 6. 复现命令

```powershell
.\ .venv\Scripts\python.exe -m tools.download_data        # 下数据
.\ .venv\Scripts\python.exe -m tools.check_leak --dataset pathmnist
.\ .venv\Scripts\python.exe -m tools.night_queue --plan main --seed 1
.\ .venv\Scripts\python.exe -m law                        # 拟合控制律
.\ .venv\Scripts\python.exe -m controller                # 停训策略对比
.\ .venv\Scripts\python.exe -m tools.plot                # 出图
.\ .venv\Scripts\python.exe -m tools.report              # 生成本文件
```
