"""清掉某个（些）run 的全部产物，让它可以干净重训。

为什么会需要：训练侧改了一件影响样本的事（比如改成 EMA 采样），
旧的样本和 metrics 行就不能再和新样本混在一条曲线里 —— 混了曲线就出假锯齿，
而 CSV 是唯一的真相来源，事后很难分辨哪一行是旧代码产的。

用法
  python build/reset_runs.py --match pathmnist_          # 所有 pathmnist run（默认只打印）
  python build/reset_runs.py --match pathmnist_ --yes    # 真删
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C
import pandas as pd

CSV = C.RESULTS / "metrics_long.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="pathmnist_", help="run 名前缀匹配")
    ap.add_argument("--run", action="append", default=None, help="精确指定某个 run（可多次）")
    ap.add_argument("--yes", action="store_true", help="不加就只打印计划")
    a = ap.parse_args()

    # 找到所有受影响的 run 名（从样本目录、ckpt 目录、CSV 三处取并集）
    names = set()
    for sub in ("samples", "ckpt"):
        d = C.CACHE / sub
        if d.exists():
            names |= {p.name for p in d.iterdir() if p.is_dir() and p.name.startswith(a.match)}
    if CSV.exists():
        df = pd.read_csv(CSV)
        names |= set(df.loc[df["run"].astype(str).str.startswith(a.match), "run"].unique())
    if a.run:
        names |= set(a.run)
    names = sorted(names)
    if not names:
        print(f"没有匹配 {a.match!r} 的 run")
        return

    print(f"将清除 {len(names)} 个 run 的产物：")
    for n in names:
        print("  -", n)
    if not a.yes:
        print("\n（只打印，没删；确认后加 --yes）")
        return

    for n in names:
        for sub in ("samples", "ckpt"):
            p = C.CACHE / sub / n
            if p.exists():
                shutil.rmtree(p)
        for p in (C.RESULTS / "runs" / f"{n}.json", C.BUILD / "logs" / f"{n}.log"):
            p.unlink(missing_ok=True)

    if CSV.exists():
        df = pd.read_csv(CSV)
        before = len(df)
        df = df[~df["run"].astype(str).isin(names)]
        df.to_csv(CSV, index=False)
        print(f"metrics_long.csv: {before} → {len(df)} 行")

    # 队列状态里也要摘掉这些 run，否则重启后会以为它们已经跑完
    st = C.BUILD / "queue_state.json"
    if st.exists():
        import json

        s = json.loads(st.read_text(encoding="utf-8"))
        for k in ("done", "scored"):
            s[k] = [r for r in s.get(k, []) if r not in names]
        s["failed"] = [f for f in s.get("failed", []) if f.get("run") not in names]
        st.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        print("queue_state.json 已同步清理")
    print(f"完成：清掉 {len(names)} 个 run")


if __name__ == "__main__":
    main()