# R 侧：独立复算（base R，不装任何包）

这个目录只有一个目的：**用第二个语言把关键数字重算一遍**，排除"Python pipeline 某处算错"。

| 文件 | 作用 |
|---|---|
| `verify_in_R.R` | 读 `results/r_*.csv` → 用 `optim` 重拟合 → 重写 R²/留一/bootstrap → 写 `results/r_verify.csv` |

## 怎么跑

```powershell
cd medmem-law
.\.venv\Scripts\python.exe -m tools.r_verify        # 导出 + 调 R + 打印对照表
```

`tools\r_verify.py` 会先在 Python 侧导出（`build\export_for_R.py`），再找 Rscript：
`MEDMEM_RSCRIPT` 环境变量 → `C:\Program Files\R\R-*\bin\Rscript.exe` → PATH。

本机实测的两条环境事实：
- `Rscript` **不在 PATH 上**（R 4.6.1 装在 `C:\Program Files\R\R-4.6.1`），所以封装脚本自己找。
- 从 venv 里 fork R 会报 `Fatal error: cannot create 'R_TempDir'`，封装脚本会把 `TMP/TEMP/TMPDIR` 指到 `build\r_tmp`。

## 它检查什么

| 检查 | 内容 | 判定 |
|---|---|---|
| A | 同一组参数下 R 的模型输出 vs Python 的预测（逐点） | 必须 `exact` |
| B | 用 Python 的预测复算 R²_log / R²_lin / 可测区间 R² / 删失合规率 | 必须 `exact` |
| C | R 独立重拟合（同目标、同边界、同多起点）→ 参数与残差平方和 | `match`（差异 <0.1%） |
| D | R 自己做留一 N（7 折）→ 池化 R² | `match` |
| E | R 自己做分层 bootstrap（200 次）→ α/γ 的 95% 分位 | `consistent`（MC 噪声内） |
| F | 用 `grid_summary.csv` 复算 onset 步数对 N 的 log-log 斜率 | 参考值 |

## 第一跑就抓到的东西

`E_ci` 一开始对不上（R 的 γ 区间比 Python 宽 0.12），顺着查下去发现是 **Python 侧的
`tools/law_ci.py` 口径错了**：它拿整张 `metrics_long.csv`（三个数据集 + 每个 seed 各算一个观测）
去 bootstrap，而主律 `law_fit.json` 是 pathmnist + 种子均值。等于报告的置信区间是另一个量的。

已修（`--dataset pathmnist` + `collapse_seeds`，与主拟合同口径），现在两边都在
γ≈−0.85 [−0.93, −0.74] 上。**这就是加 R 这一步的价值**：它不会替你算对，但它会逼你把口径写清楚。

## 从 R 的角度看这套分析

- 拟合是**非线性最小二乘 + 有界约束 + 多起点** → R 里就是 `optim(method="L-BFGS-B")`，不需要 `nls`
  （`nls` 对 5 参数、带删失罚项的目标不稳定）。
- 删失数据的处理（M=0 的点用单侧 softplus 罚）是手写的残差函数，R 与 Python 各写一遍，
  两份代码独立，但对同一个点给出同一个数 —— 这比"用同一个库"更有说服力。
- 结论：**这条律不是靠某个库的特性凑出来的**，换一个语言、换一个优化器，参数落在同一点。