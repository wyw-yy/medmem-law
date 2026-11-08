"""
四种停训策略的离线对比（不需要额外训练，全部在已测曲线上模拟）。

A 固定跑满      ：跑满 E_max（= 无早停控制的常规训练，即 PPT 里的基线）
B 生成质量平台  ：FID 在最近 W 个检查点上的改善 < δ 就停（最应该被我们打赢的对手）
C 开环律早停    ：E*(N,ε) 由 law.py 拟合的律直接给出，训练前就定好
D 闭环在线早停  ：每 CKPT_EVERY 轮实测一次 M，M ≥ ε 立刻停（含"预测+缓冲"版本）

评价指标（对每个 run）
  epochs       用了多少轮（成本）
  M_stop       停止时的记忆率（隐私风险）
  fid_stop     停止时的 FID（质量）
  compliant    是否满足 M_stop ≤ ε（合规）
  regret       相比 A 省下的训练轮次占比
论文里那张"成本—隐私"图就是从这里出来的。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def _by_epoch(df_run: pd.DataFrame) -> pd.DataFrame:
    return df_run.sort_values("epoch").reset_index(drop=True)


def stop_A(df_run, eps, **kw) -> int:
    return int(df_run["epoch"].max())


def stop_B(df_run, eps, window: int = 3, delta: float = 0.02) -> int:
    """FID 相对改善 < delta 连续 window 个点 → 停。没有 FID 列就退回跑满。"""
    if "fid" not in df_run or df_run["fid"].isna().all():
        return stop_A(df_run, eps)
    d = df_run.dropna(subset=["fid"]).reset_index(drop=True)
    if len(d) <= window:
        return stop_A(df_run, eps)
    for i in range(window, len(d)):
        prev = d.loc[i - window, "fid"]
        cur = d.loc[i, "fid"]
        if prev <= 0:
            continue
        if (prev - cur) / prev < delta:
            return int(d.loc[i - window, "epoch"])
    return int(d["epoch"].max())


def stop_C(df_run, eps, law: dict | None = None) -> int:
    """开环：直接用律给的 E*(N,ε)。"""
    if not law:
        return stop_A(df_run, eps)
    n = str(int(df_run["n"].iloc[0]))
    e = law.get("E_star", {}).get(str(eps), {}).get(n)
    if e is None or e != e:  # nan
        return stop_A(df_run, eps)
    return int(round(e))


def stop_D(df_run, eps, margin: float = 0.0) -> int:
    """闭环：每个检查点实测一次 M，一旦 M ≥ eps*(1-margin)，就交付**上一个检查点**
    （最后一个仍然合规的权重），而不是那个已经越界的权重。

    上一版直接返回越界点本身 → 停点 M 必然 > eps，违规率按定义就是 100%，
    这不是控制器的行为而是一个评测 bug（部署时丢弃越界 ckpt 是标准做法）。
    若第一个检查点就已经越界，说明没有安全点可交付，只好返回它并计为违规。
    """
    d = _by_epoch(df_run)
    thr = eps * (1 - margin)
    hit_idx = d.index[d["M"] >= thr]
    if len(hit_idx) == 0:
        return int(d["epoch"].max())
    i = int(hit_idx[0])
    return int(d["epoch"].iloc[max(i - 1, 0)])


def stop_D_predict(df_run, eps, slope_k: float = 1.0) -> int:
    """闭环+预测：用相邻两点的斜率外推，预测下一个检查点会越过 eps 就现在停。
    这是"控制器"应有的样子：不等越界，而是提前一个检查点刹住。"""
    d = _by_epoch(df_run)
    ep = d["epoch"].to_numpy(int)
    m = d["M"].to_numpy(float)
    for i in range(len(d) - 1):
        if m[i] >= eps:
            return int(ep[i])
        dz = ep[i + 1] - ep[i]
        slope = (m[i + 1] - m[i]) / max(dz, 1)
        if m[i] + slope_k * slope * dz >= eps:
            return int(ep[i])
    return int(ep[-1])


def evaluate(df: pd.DataFrame, law: dict | None, eps: float = C.EPS_MEM) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run"):
        g = _by_epoch(g)
        picks = {
            "A_fixed_max": stop_A(g, eps),
            "B_fid_plateau": stop_B(g, eps),
            "C_open_loop": stop_C(g, eps, law),
            "D_closed_loop": stop_D(g, eps),
            "Dp_closed_predict": stop_D_predict(g, eps),
        }
        e_max = picks["A_fixed_max"]
        # FID 只每隔 fid-stride 个点算过，停点处常常没有直接测量值。
        # 用该 run 的 FID 采样点做线性插值（外推用最近端点），并标记 fid_interp，
        # 免得把"没测"读成"没代价"。
        fid_pts = g[g["fid"].notna()][["epoch", "fid"]] if "fid" in g else pd.DataFrame()
        for name, ep in picks.items():
            at = g[g["epoch"] <= ep]
            if len(at) == 0:
                at = g.iloc[[0]]
            last = at.iloc[-1]
            fid_val, fid_interp = np.nan, False
            if pd.notna(last.get("fid")):
                fid_val = float(last["fid"])
            elif len(fid_pts) >= 2:
                fid_val = float(np.interp(float(ep), fid_pts["epoch"].to_numpy(), fid_pts["fid"].to_numpy()))
                fid_interp = True
            rows.append({
                "run": run, "dataset": g["dataset"].iloc[0], "n": int(g["n"].iloc[0]),
                "seed": int(g["seed"].iloc[0]) if "seed" in g else 0,
                "policy": name, "epochs": ep,
                "epoch_saving": 1.0 - ep / max(e_max, 1),
                "M_stop": float(last["M"]),
                "fid_stop": fid_val,
                "fid_interp": fid_interp,
                "compliant": bool(last["M"] <= eps),
            })
    return pd.DataFrame(rows)


def summarize(tbl: pd.DataFrame, eps: float = C.EPS_MEM) -> pd.DataFrame:
    s = (tbl.groupby("policy")
         .agg(n=("run", "size"),
              epochs=("epochs", "mean"),
              saving=("epoch_saving", "mean"),
              M_stop=("M_stop", "mean"),
              M_stop_p95=("M_stop", lambda x: float(np.quantile(x, 0.95))),
              fid=("fid_stop", "mean"),
              compliant=("compliant", "mean"))
         .reset_index())
    s["violation_rate"] = 1 - s["compliant"]
    return s.sort_values("epochs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(C.RESULTS / "metrics_long.csv"))
    ap.add_argument("--law", default=str(C.RESULTS / "law_fit.json"))
    ap.add_argument("--eps", type=float, default=C.EPS_MEM)
    # 默认只在主数据集上比较停训策略：跨数据集混在一起算"平均省了多少轮"没有意义
    # （bloodmnist N=1000 的 M 比 pathmnist 高 5 倍，两种 regime 会被平均掉）。
    ap.add_argument("--dataset", default="pathmnist", help="只用某个数据集；空字符串=全部")
    ap.add_argument("--seed", type=int, default=0, help="只用某个 seed；0=全部")
    a = ap.parse_args()

    df = pd.read_csv(a.metrics)
    # 与主拟合同一把尺子：先剔掉 N<FIT_MIN_N 的下界探针（它们首个检查点就已越线，
    # 任何策略都不可能"合规地停下"，混进来只会把违规率抬高、把省轮比例算错）。
    import law as L

    df = L.drop_probes(df)
    if a.dataset:
        df = df[df["dataset"] == a.dataset].copy()
    if a.seed:
        df = df[df["seed"] == a.seed].copy()
    if df.empty:
        raise SystemExit(f"没有数据（dataset={a.dataset!r} seed={a.seed}）")
    law = json.loads(Path(a.law).read_text(encoding="utf-8")) if Path(a.law).exists() else None
    if law is None:
        print("提示：没找到 law_fit.json，策略 C 会退化成跑满")

    tbl = evaluate(df, law, a.eps)
    tbl.to_csv(C.RESULTS / "stop_policy_table.csv", index=False)
    print(f"写入 {C.RESULTS / 'stop_policy_table.csv'}（{len(tbl)} 行）\n")

    s = summarize(tbl, a.eps)
    print(f"停训策略对比（ε={a.eps}）")
    print(f"{'策略':<20}{'run':>5}{'轮次':>8}{'省%':>7}{'M_停':>8}{'M_p95':>8}{'FID':>8}{'违规率':>8}")
    for _, r in s.iterrows():
        fid = f"{r['fid']:.1f}" if pd.notna(r["fid"]) else "  -"
        print(f"{r['policy']:<20}{r['n']:>5}{r['epochs']:>8.0f}{100*r['saving']:>7.0f}"
              f"{r['M_stop']:>8.4f}{r['M_stop_p95']:>8.4f}{fid:>8}"
              f"{100*r['violation_rate']:>7.0f}%")
    s.to_csv(C.RESULTS / "stop_policy_summary.csv", index=False)


if __name__ == "__main__":
    main()