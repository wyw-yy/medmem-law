"""对选定的控制律做分层 bootstrap 置信区间（按 N 分层重采样测量点）。

为什么要有：只报一个点估计（γ=0.22、R²=0.92）在评审里是站不住的 ——
必须给参数与 E* 的区间。分层重采样（在每个 N 内部有放回抽点）保持了
"每个数据规模若干条曲线"的原始结构，比整表重抽更合理。

用法：
    python -m tools.law_ci                 # 默认 200 次
    python -m tools.law_ci --n-boot 500
输出：results/law_bootstrap.json（+ 终端表格）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C   # noqa: E402
import law as L      # noqa: E402


def pick_model(df) -> str:
    """优先读 law_fit.json 里选出的模型，没有就现场按 AICc 选。"""
    p = C.RESULTS / "law_fit.json"
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            for key in ("chosen", "best", "model"):
                if isinstance(j.get(key), str) and j[key] in L.CANDIDATES:
                    return j[key]
            for c in j.get("candidates", j.get("fits", [])):
                if c.get("name") in L.CANDIDATES and c.get("chosen"):
                    return c["name"]
        except Exception:  # noqa: BLE001
            pass
    fits = [L.fit_one(nm, df) for nm in L.CANDIDATES]
    return min(fits, key=lambda d: d["aicc"])["name"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=C.EPS_MEM)
    # 默认必须与主拟合同口径：只主数据集 + 种子均值。
    # 踩过的坑：早期这里直接拿整张 CSV（三个数据集 + 每个 seed 各算观测），
    # 结果 bootstrap 的"点估计"跟 law_fit.json 对不上（γ −0.88 vs −0.85，
    # R²_log 0.737 vs 0.709）——给人的区间是另一个量的区间。
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--per-seed", action="store_true", help="不折种子（与 law --per-seed 对应）")
    ap.add_argument("--all-datasets", action="store_true", help="不按数据集过滤")
    a = ap.parse_args()

    df = L.load_metrics(Path(a.metrics) if a.metrics else None)
    if not a.all_datasets:
        df = df[df["dataset"] == a.dataset].copy()
    if not a.per_seed:
        df = L.collapse_seeds(df)
    print(f"[law_ci] 口径：dataset={a.dataset if not a.all_datasets else 'all'}，"
          f"{len(df)} 点，{'逐 run' if a.per_seed else '种子均值'}")
    name = pick_model(df)
    spec = L.CANDIDATES[name]
    base = L.fit_one(name, df)
    print(f"模型 {name}  参数 {np.round(base['params'], 4).tolist()}  "
          f"R²_log {base['r2_log']:.4f}  AICc {base['aicc']:.1f}")

    rng = np.random.default_rng(a.seed)
    groups = [g for _, g in df.groupby("n")]
    Ns = sorted(df["n"].unique())
    boot = {k: [] for k in ("params", "r2_log", "e_star")}
    ok = 0
    for _ in range(a.n_boot):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        d = __import__("pandas").concat(parts, ignore_index=True)
        try:
            r = L.fit_one(name, d)
        except Exception:  # noqa: BLE001
            continue
        boot["params"].append(r["params"])
        boot["r2_log"].append(r["r2_log"])
        e_star = []
        for n in Ns:
            try:
                e_star.append(float(L.solve_E_star(name, r["params"], n, a.eps)))
            except Exception:  # noqa: BLE001
                e_star.append(float("nan"))
        boot["e_star"].append(e_star)
        ok += 1
        if ok % 25 == 0:
            print(f"  已 {ok}/{a.n_boot}", flush=True)

    P = np.array(boot["params"], float)
    R2 = np.array(boot["r2_log"], float)
    ES = np.array(boot["e_star"], float)
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731

    names = spec.get("pnames") or [f"p{i}" for i in range(len(base["params"]))]
    out = {
        "model": name, "n_boot": ok, "eps": a.eps,
        "point": {"params": base["params"], "r2_log": base["r2_log"], "aicc": base["aicc"]},
        "ci": {nm: {"point": float(base["params"][i]), "lo": q(P[:, i])[0], "hi": q(P[:, i])[1]}
               for i, nm in enumerate(names)},
        "r2_log_ci": {"point": base["r2_log"], "lo": q(R2)[0], "hi": q(R2)[1]},
        "e_star": {str(n): {"point": float(L.solve_E_star(name, base["params"], n, a.eps)),
                            "lo": q(ES[:, i])[0], "hi": q(ES[:, i])[1], "p50": float(np.nanpercentile(ES[:, i], 50))}
                   for i, n in enumerate(Ns)},
    }
    outp = C.RESULTS / "law_bootstrap.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'参数':10s} {'点估计':>10s} {'95% CI':>24s}")
    for nm, d in out["ci"].items():
        print(f"{nm:10s} {d['point']:10.4f}   [{d['lo']:.4f}, {d['hi']:.4f}]")
    print(f"{'R²_log':10s} {base['r2_log']:10.4f}   [{out['r2_log_ci']['lo']:.4f}, {out['r2_log_ci']['hi']:.4f}]")
    print(f"\nE*(ε={a.eps}) 区间：")
    for n, d in out["e_star"].items():
        print(f"  N={n:>5s}  {d['point']:8.0f}   [{d['lo']:.0f}, {d['hi']:.0f}]")
    print(f"\n写出 {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())