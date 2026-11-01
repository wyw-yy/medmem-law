"""把 Python 侧的拟合输入导出给 R 独立复算（base R，不装任何包）。

导出三样东西：
  results/r_fit_points.csv  律拟合真正用到的点（已折种子、已排除干预组）+ Python 的预测值
  results/r_params.csv      选定律的名字与 5 个参数（长表：param,value）
  results/r_summary.csv     Python 侧的验收数字（长表：key,value），供 R 逐项对照

为什么要这个：论文里的 R²、留一、置信区间如果只有 Python 一个实现，
审稿人（和导师）没法排除"pipeline 里某处算错"。R 用 base R 的 optim 独立
重算一遍，两条路对得上，等于给结论加了一道独立校验。
"""

from __future__ import annotations

import json

import pandas as pd

import config as C
import law as L


def main() -> None:
    fit = json.loads((C.RESULTS / "law_fit.json").read_text(encoding="utf-8"))
    best = fit["best"]
    fp = fit["fits"][best]
    params = [float(v) for v in fp["params"]]
    pnames = L.CANDIDATES[best]["pnames"]

    # 与 law.main(--dataset pathmnist) 完全同一条路径
    df = L.load_metrics()
    df = df[df["dataset"] == "pathmnist"].copy()
    df = L.collapse_seeds(df)
    df = df.sort_values(["n", "epoch"]).reset_index(drop=True)

    pred = L._pred(L.CANDIDATES[best]["f"], params,
                   df["n"].to_numpy(float), df["epoch"].to_numpy(float))
    out = pd.DataFrame({
        "dataset": df["dataset"].astype(str),
        "n": df["n"].astype(int),
        "epoch": df["epoch"].astype(int),
        "M": df["M"].astype(float),
        "n_gen": df["n_gen"].astype(float),
        "pred_py": pred.astype(float),
    })
    out.to_csv(C.RESULTS / "r_fit_points.csv", index=False, encoding="utf-8")

    pd.DataFrame({"param": pnames, "value": params}).to_csv(
        C.RESULTS / "r_params.csv", index=False, encoding="utf-8")

    summary = {
        "law": best,
        "n_points": len(df),
        "n_obs": int((df["M"] > 0).sum()),
        "n_meas": int((df["M"] >= L.MEAS_FLOOR).sum()),
        "r2_log": fp["r2_log"],
        "r2_lin": fp["r2_lin"],
        "r2_log_meas": fp["r2_log_meas"],
        "r2_lin_meas": fp["r2_lin_meas"],
        "aicc": fp["aicc"],
        "rmse_log": fp["rmse_log"],
        "censored_compliance": fp.get("censored_compliance"),
        "loo_pooled": fp.get("loo", {}).get("r2_pooled") if isinstance(fp.get("loo"), dict) else None,
        "objective": float(sum(L._resid(L.CANDIDATES[best]["f"], params,
                                        df["n"].to_numpy(float),
                                        df["epoch"].to_numpy(float),
                                        df["M"].to_numpy(float),
                                        L._mdet(df)) ** 2)),
    }
    # loo 可能不在 fits 里，退回顶层
    if summary["loo_pooled"] is None and "loo" in fit:
        summary["loo_pooled"] = (fit["loo"].get(best, {}) or {}).get("r2_pooled")

    # bootstrap 置信区间（与主拟合同口径：pathmnist + 种子均值），供 R 逐项对照
    bpath = C.RESULTS / "law_bootstrap.json"
    if bpath.exists():
        boot = json.loads(bpath.read_text(encoding="utf-8"))
        ci = boot.get("ci", {})
        for nm in ("alpha", "gamma"):
            if nm in ci:
                summary[f"{nm}_lo"] = ci[nm]["lo"]
                summary[f"{nm}_hi"] = ci[nm]["hi"]
        summary["boot_n"] = boot.get("n_boot")

    pd.DataFrame({"key": list(summary), "value": list(summary.values())}).to_csv(
        C.RESULTS / "r_summary.csv", index=False, encoding="utf-8")

    print(f"[export_for_R] {best}  {dict(zip(pnames, [round(p, 4) for p in params]))}")
    print(f"[export_for_R] {len(out)} 个点 → results/r_fit_points.csv；"
          f"objective={summary['objective']:.6f}")
    if summary["loo_pooled"] is None:
        print("[export_for_R] 警告：law_fit.json 里没找到留一池化 R²，R 会自己算")


if __name__ == "__main__":
    main()