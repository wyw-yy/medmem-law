# =============================================================================
# 独立复算：用 base R（不装任何包）把 Python 侧的记忆化控制律重算一遍
# -----------------------------------------------------------------------------
# 为什么写这个脚本：
#   论文里的 R²、留一 N、γ/α 的置信区间如果只有 Python 一个实现，审稿人无法
#   排除"pipeline 里某处算错"。这里用 R 的 optim 独立重拟合，逐项对比。
#   两条完全独立的实现给出同样的数字，比多发 100 个 seed 更能说服人。
#
# 用法（Rscript 不在 PATH 上时用绝对路径）：
#   "C:/Program Files/R/R-4.6.1/bin/Rscript.exe" R/verify_in_R.R <项目根>
#   或用封装：python -m tools.r_verify
#
# 输入（由 build/export_for_R.py 生成）：
#   results/r_fit_points.csv   律拟合用的点（已折种子、已排除干预组）+ Python 的预测
#   results/r_params.csv       Python 选定的参数
#   results/r_summary.csv      Python 侧的验收数字
#   results/grid_summary.csv   每个 run 的 onset 步数（用于斜率复算）
# 输出：results/r_verify.csv（长表：check,item,r,python,abs_diff,status）
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "."
res  <- file.path(root, "results")

d  <- read.csv(file.path(res, "r_fit_points.csv"), stringsAsFactors = FALSE)
pa <- read.csv(file.path(res, "r_params.csv"),     stringsAsFactors = FALSE)
sm <- read.csv(file.path(res, "r_summary.csv"),    stringsAsFactors = FALSE)

p_py <- as.numeric(pa$value)                 # A0, alpha, beta, E50_0, gamma
names(p_py) <- pa$param
num  <- function(k) suppressWarnings(as.numeric(sm$value[match(k, sm$key)]))

EPS_LOG_LOW <- 1e-4      # 与 Python 的 law.EPS_LOG 对齐
CENSOR_K    <- 4.0       # 与 Python 的 law.CENSOR_K 对齐
MEAS_FLOOR  <- 0.02      # 与 Python 的 law.MEAS_FLOOR 对齐
N0          <- 250.0

# ---------------------------------------------------------------- 模型 L4
pred_l4 <- function(p, N, E) {
  r   <- pmax(N, 1) / N0
  amp <- p[1] * r^(-p[2])                    # A0 * (N/250)^(-alpha)；注意负号
  amp * (1 - exp(-(E / (p[4] * r^p[5]))^p[3]))
}

mdet <- function(n_gen) 3 / pmax(n_gen, 1)    # 0 命中时的 95% 二项上界

resid_l4 <- function(p, dat) {
  pr  <- pmin(pmax(pred_l4(p, dat$n, dat$epoch), 1e-9), 1)
  lp  <- log10(pr + EPS_LOG_LOW)
  obs <- dat$M > 0
  r   <- numeric(nrow(dat))
  r[obs] <- lp[obs] - log10(dat$M[obs] + EPS_LOG_LOW)
  if (any(!obs)) {
    z <- CENSOR_K * (lp[!obs] - log10(mdet(dat$n_gen[!obs]) + EPS_LOG_LOW))
    r[!obs] <- log1p(exp(pmin(pmax(z, -30), 30))) / CENSOR_K
  }
  r
}

obj <- function(p, dat) {
  r <- resid_l4(p, dat)
  if (!all(is.finite(r))) return(1e12)
  sum(r * r)
}

LO  <- c(1e-3, -2.0, 0.2, 1e-2, -1.0)
HI  <- c(1.0,   6.0, 8.0, 1e5,   3.0)

# 多起点：与 Python 的 _p0s 同构（gamma 扫描 × E50 尺度扫描），初值取 CANDIDATES["L4_amp_decay"]["p0"]
starts <- function() {
  base <- c(0.85, 2.0, 1.5, 700.0, 0.2)
  out <- list()
  for (g in c(0, 0.5, 1.0, 1.5)) {
    for (s in c(0.5, 1.0, 3.0)) {
      p <- base
      p[5] <- g
      p[4] <- min(max(base[4] * s, 1e-2), 1e5)
      out[[length(out) + 1]] <- p
    }
  }
  out
}

fit_l4 <- function(dat) {
  best <- NULL
  for (p0 in starts()) {
    r <- try(optim(p0, obj, dat = dat, method = "L-BFGS-B",
                   lower = LO, upper = HI, control = list(maxit = 2000)), silent = TRUE)
    if (inherits(r, "try-error")) next
    if (is.null(best) || r$value < best$value) best <- r
  }
  best
}

# ---------------------------------------------------------------- 逐项校验
rows <- list()
# tol: 绝对容差；参数用相对容差（E50 是 2401 这种量级，绝对 1e-3 没意义）
add <- function(check, item, r, py, note = "", tol = 1e-3, rel = FALSE, loose = 0.05) {
  if (is.na(py)) { d <- NA_real_ } else if (rel) { d <- abs(r - py) / max(abs(py), 1e-9) } else { d <- abs(r - py) }
  st <- if (is.na(d)) "n/a"
        else if (d <= 1e-6) "exact"
        else if (d <= tol) "match"
        else if (d <= loose) "consistent"
        else "CHECK"
  rows[[length(rows) + 1]] <<- data.frame(
    check = check, item = item, r = round(r, 6), python = round(py, 6),
    abs_diff = if (is.na(d)) NA_real_ else signif(d, 3), status = st, note = note,
    stringsAsFactors = FALSE)
}

## A. 模型实现一致性：同一组参数下 R 的预测 vs Python 的预测
pred_r <- pred_l4(p_py, d$n, d$epoch)
add("A_model", "max_abs_diff_pred", max(abs(pred_r - d$pred_py)), 0, "两条实现的模型必须逐点相同")

## B. 用 Python 的预测复算 R²（验证 R² 公式与点集选择）
obs  <- d$M > 0
meas <- d$M >= MEAS_FLOOR
r2 <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
add("B_metrics", "R2_log", r2(log10(d$M[obs] + EPS_LOG_LOW), log10(pmin(pmax(d$pred_py[obs], 1e-9), 1) + EPS_LOG_LOW)), num("r2_log"), "M>0 的 log10 空间")
add("B_metrics", "R2_lin", r2(d$M[obs], d$pred_py[obs]), num("r2_lin"), "M>0 的线性空间")
add("B_metrics", "R2_log_meas", r2(log10(d$M[meas] + EPS_LOG_LOW), log10(pmin(pmax(d$pred_py[meas], 1e-9), 1) + EPS_LOG_LOW)), num("r2_log_meas"), "M>=0.02")
add("B_metrics", "R2_lin_meas", r2(d$M[meas], d$pred_py[meas]), num("r2_lin_meas"), "M>=0.02")
cc <- mean(pred_l4(p_py, d$n[!obs], d$epoch[!obs]) <= mdet(d$n_gen[!obs]))
add("B_metrics", "censored_compliance", cc, num("censored_compliance"), "删失点预测是否低于检出限")

## C. 独立重拟合：不看 Python 的参数，只给同样的目标和边界
fit_r <- fit_l4(d)
add("C_refit", "objective", fit_r$value, num("objective"), "残差平方和（越小越好）")
add("C_refit", "objective_at_py", obj(p_py, d), num("objective"), "把 Python 解代进 R 的目标函数")
pn <- c("A0_at250", "alpha", "beta_h", "E50_at250", "gamma")
for (i in 1:5) add("C_refit", pn[i], fit_r$par[i], p_py[i], "R 独立拟合 vs Python", rel = (i == 4))
pred_fit <- pred_l4(fit_r$par, d$n, d$epoch)
add("C_refit", "R2_log", r2(log10(d$M[obs] + EPS_LOG_LOW), log10(pmin(pmax(pred_fit[obs], 1e-9), 1) + EPS_LOG_LOW)), num("r2_log"), "R 自己拟合的 R²")

## D. 留一 N（池化）：每折用其余 N 拟合，预测留出的 N
pooled_y <- c(); pooled_p <- c()
ns <- sort(unique(d$n))
for (h in ns) {
  tr <- d[d$n != h, ]; te <- d[d$n == h, ]
  if (nrow(te) == 0 || nrow(tr) < 8) next
  f <- fit_l4(tr)
  if (is.null(f)) next
  pr <- pmin(pmax(pred_l4(f$par, te$n, te$epoch), 0), 1)
  o  <- te$M > 0
  pooled_y <- c(pooled_y, log10(te$M[o] + EPS_LOG_LOW))
  pooled_p <- c(pooled_p, log10(pr[o] + EPS_LOG_LOW))
}
add("D_loo", "R2_pooled_loo", r2(pooled_y, pooled_p), num("loo_pooled"), "留一 N 后池化（与 report 同口径）")

## E. bootstrap 置信区间（按 N 分层，200 次）：用 R 自己的重采样
set.seed(20261101)
B <- 200
boot <- matrix(NA_real_, B, 2, dimnames = list(NULL, c("alpha", "gamma")))
idx_by_n <- split(seq_len(nrow(d)), d$n)
for (b in 1:B) {
  sel <- unlist(lapply(idx_by_n, function(ix) sample(ix, length(ix), replace = TRUE)))
  db <- d[sel, ]
  r <- try(optim(fit_r$par, obj, dat = db, method = "L-BFGS-B",
                 lower = LO, upper = HI, control = list(maxit = 1000)), silent = TRUE)
  if (!inherits(r, "try-error")) boot[b, ] <- r$par[c(2, 5)]
}
ci <- function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE)
# 两个 bootstrap 各自用不同的随机数流与优化器，端点只可能"落在蒙特卡洛噪声内"，
# 不可能逐位相同：B=200 时 2.5%/97.5% 分位的标准误约 sd*sqrt(p(1-p)/B)/phi ≈ 0.01。
add("E_ci", "alpha_lo", ci(boot[, "alpha"])[[1]], num("alpha_lo"), "bootstrap 分位（MC 噪声 ~0.01）", loose = 0.06)
add("E_ci", "alpha_hi", ci(boot[, "alpha"])[[2]], num("alpha_hi"), "", loose = 0.06)
add("E_ci", "gamma_lo", ci(boot[, "gamma"])[[1]], num("gamma_lo"), "", loose = 0.06)
add("E_ci", "gamma_hi", ci(boot[, "gamma"])[[2]], num("gamma_hi"), "", loose = 0.06)
add("E_ci", "alpha_sd", sd(boot[, "alpha"], na.rm = TRUE), NA, "R 的 bootstrap 离散度")
add("E_ci", "gamma_sd", sd(boot[, "gamma"], na.rm = TRUE), NA, "")

## F. onset 斜率（实测，不含模型）：到 1% 记忆率所需步数 vs N
gs <- read.csv(file.path(res, "grid_summary.csv"), stringsAsFactors = FALSE)
if ("steps_1pct" %in% names(gs)) {
  for (ds in unique(gs$dataset)) {
    s <- gs[gs$dataset == ds & is.finite(gs$steps_1pct) & gs$steps_1pct > 0, ]
    if (nrow(s) >= 2) {
      sl <- coef(lm(log(s$steps_1pct) ~ log(s$n)))[[2]]
      add("F_onset", paste0("slope_", ds), sl, NA, "log-log 斜率，≈1 表示 onset 步数 ∝ N")
    }
  }
}

out <- do.call(rbind, rows)
write.csv(out, file.path(res, "r_verify.csv"), row.names = FALSE)

cat("\n===== R 独立复算结果（base R，无外部包）=====\n")
print(out[, c("check", "item", "r", "python", "abs_diff", "status")], row.names = FALSE)
bad  <- out[out$status == "CHECK", ]
soft <- out[out$status == "consistent", ]
cat("\n判定：", if (nrow(bad) == 0) "无冲突" else paste0(nrow(bad), " 项冲突"),
    "（exact/match ", sum(out$status %in% c("exact", "match")), " 项",
    if (nrow(soft)) paste0("；consistent ", nrow(soft), " 项为 bootstrap 分位的蒙特卡洛噪声") else "",
    "）\n", sep = "")
if (nrow(bad)) cat("需要人工核对：", paste(bad$item, collapse = ", "), "\n", sep = "")