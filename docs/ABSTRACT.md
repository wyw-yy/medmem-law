# 摘要 / 结论页草稿（数字全部来自 `results/`，可逐条核对）

## Abstract（英文版，约 250 词）

Medical image generators can reproduce training data. Whether they *will* — and when —
is currently answered case by case, after the fact. We ask a narrower, answerable question:
**for a diffusion model trained on a fixed dataset of size N, how does the memorization
rate M grow with training time, and can that growth be scheduled against?**

We train 32 DDPM models (13.0 M parameters) on three public medical imaging datasets
(PathMNIST, BloodMNIST, PneumoniaMNIST) at 64×64, sweeping N over seven values from 250
to 4,000 and epochs from 325 to 5,125 under an equal-*update*-budget protocol, and measure M
with a calibrated near-duplicate metric: DINOv2 features, truncated whitening, and a threshold
anchored to a 0.5% false-positive rate on a held-out non-member reference set. Each run yields
~30 checkpoints, giving 973 measurement points.

Three findings. (i) The **plateau** of M decays steeply with data: fitting a stretched-exponential
law with power-law amplitude decay, A(N) ∝ N^(−1.90) [95% CI 1.88–2.05 in the 3-seed fit], so that at
N≥2,000 the model never leaves the measurement floor (3/512 = 0.6%) within 80 k updates — four times
the equal-step budget. (ii) The **onset time** grows with N, but far slower than the update-normalized
prediction (γ_epoch = −0.83 ⟺ t₅₀ ∝ N^0.17), and per-seed onset varies by up to 8× at fixed N — so a
fitted schedule can describe the *expectation* but cannot *guarantee* compliance. (iii) Consequently, a
**closed-loop controller** that measures M online and stops before the threshold is crossed delivers 67%
savings of the training budget while violating the privacy budget in 0 of 28 runs, versus 75% violations
for FID-plateau stopping and 11% for the open-loop law. The fitted law reaches R²_lin 0.710 (in-sample)
but leave-one-N-out R² 0.46 — we report it as a descriptive law with an explicit validity boundary, not
as a predictive guarantee.

## 一句话结论（组会/答辩直接用）

> **"少训一点"这件事有规律，但没有保证；要保证就得在线测。**
> 规律：平台 ∝ N^(−1.9)，起跳时间 ∝ N^(0.17–0.65)（口径不同）；保证：闭环 0/28 违规、省 64%，
> 开环律 10.7% 违规，FID 早停 75% 违规。

## 三条主结论 + 各自的可核对出处

| # | 结论 | 数字 | 出处 |
|---|---|---|---|
| 1 | 平台高度随数据量快速衰减 | A(N) ∝ N^(−1.90)；7 档平台 0.863/0.469/0.092/0.021/0.007/0.011/0.014（N=250→4000，后三档贴地板） | `results/law_fit.json`、`figs/fig1_triptych.png (b)` |
| 2 | 起跳时间随 N 线性推迟（结构上复现文献），但常数小 30 倍、且 seed 方差极大 | `onset/N` = 11.07/10.63/10.40 步/张（N=250/500/1000），斜率 **+0.95**；文献 ≈300 步/张（Eq.6 像素判据）；γ_epoch=−0.831 ⟺ 名义 t₅₀∝N^0.169；同 N 不同 seed 的 onset 极差 8–60× | `results/grid_summary.csv`、`results/eq6.csv`、`docs/COMPARISON.md` §2 |
| 3 | 只有闭环能在"省时间"和"守隐私"之间同时成立 | D/D′：省 66.6%、违规 0/34；C 开环：省 64.8%、违规 11.8%；B(FID)：省 23.9%、违规 79.4%；A：违规 79.4% | `results/stop_policy_summary.csv`、`figs/fig1_triptych.png (c)` |

## 必须写进 limitation 的四件事

1. **R² 没到 0.9**：样本内 R²_lin 0.710 / R²_log 0.682；留一池化 0.464（逐 run 0.793/0.811）。预声明线未达成，如实报告。
   原因是可测范围只到 N≲1500（N≥2000 的 M 低于检出限 3/512），且**误差主要在幅度项**（只放开幅度可消 78.6% 残差，只放开时间轴只消 44%）。
2. **安全表 0/28 行全过**：严格四条件（最差 seed ≤ ε、经验停点不违约、FID ≤ 1.5×、ε > 2×FPR）没有任何一行同时成立 ——
   小数据上隐私与质量不可兼得，处方是加数据或闭环 + 质量二次判定。
3. **分辨率与数据来源**：64×64（由 MedMNIST+ 224 源下采样），不是原始 DICOM 临床分辨率；绝对 M/FID 不可跨分辨率比较（28px 附录给出对照）。
4. **时间单位与判据口径**：文献用优化步 + 像素最近/次近比判据，本文主坐标用 epoch + 白化特征余弦判据；
   两者**结构一致（onset ∝ N，斜率 0.95）但常数差约 30 倍**，不可直接比绝对值；换编码器会改变幅度指数（−2.46 vs −1.13）。

## 与预声明的关系（审稿人最会挑的地方，先说）

- 预声明（`docs/PROTOCOL.md` §10/§14）写明"R²_log ≥ 0.9 且留一 ≥ 0.8" → **未达成**，且**没有为了达标而换口径**：
  两套口径（种子均值 / 逐 run）都报，可测区间 R² 也报（0.727/0.649）。
- 中途新增候选律 L4/L5/L6 时记录了日期与理由（§10 R3），并在 cross/seed2 数据出来之前就固定；
  L6（指数幅度）与 L7（形状随 N 变）试过并明确记录为"被否决"——不是只报赢家。
- 干预组（开环/闭环早停的 run）**不进律拟合**（`law._stopped_runs`），否则结论自我循环。

## 如果只给一句话的"方法学贡献"

> 记忆化研究缺的不是更多案例，而是**可审计的测量协议**：阈值锚定假阳性率、编码器锁定、
> 种子方差与检出限显式报告、交付点语义（越界回退）、以及"口径不改"的预声明。
> 这套协议让"记忆率"从一次性观察变成一个可以写进 SOP 的数。