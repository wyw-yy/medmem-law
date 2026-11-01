"""数据齐了之后一条龙重跑：评分 → 律 → bootstrap → 停训策略 → 安全表 → 图 → 报告。

用法：
    python build\finalize.py              # 全跑
    python build\finalize.py --skip-score  # 评分已完成时跳过（最常见）

每步失败不中断（后面把错误汇总），最后打印一张耗时/状态表。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

STEPS = [
    ("score_all", ["-m", "tools.score_run", "--all", "--fid", "--fid-stride", "4"], True),
    # 主律只用主数据集 pathmnist：cross 的 bloodmnist/pneumoniamnist 只有两档 N，
    # 混进主拟合会把"函数族"的估计带偏；它们的可迁移性由 transfer 步骤单独检验。
    # 输出仍写到 results/law_fit.json，因为 controller/safety/train --stop-law 都读这个文件名。
    ("law", ["-m", "law", "--dataset", "pathmnist", "--out", "results/law_fit.json"], False),
    ("law_ci", ["-m", "tools.law_ci", "--n-boot", "200"], False),
    # R 独立复算（base R 的 optim 重拟合 + 自写 bootstrap + 留一 N）：见 docs/NOTES.md §9。
    # 放在 law_ci 之后，因为它要读 law_fit.json 与 law_bootstrap.json 做对照。
    ("r_verify", ["-m", "tools.r_verify"], False),
    ("law_perseed", ["-m", "law", "--dataset", "pathmnist", "--per-seed",
                     "--out", "results/law_fit_perseed.json"], False),
    # τ 的唯一自由参数是 FPR：0.1%/0.5%/1% 各算一遍，看记忆率结论是否依赖它。
    ("fpr_sens", ["-m", "build.fpr_sens"], False),
    ("transfer", ["-m", "tools.transfer"], False),
    ("controller", ["-m", "controller"], False),
    ("safety", ["-m", "tools.safety"], False),
    ("grid_summary", ["-m", "tools.grid_summary"], False),
    ("plot", ["-m", "tools.plot", "--only", "all"], False),
    ("audit", ["-m", "tools.audit"], False),
    ("report", ["-m", "tools.report"], False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--only", default="", help="只跑名字里含该子串的步骤")
    a = ap.parse_args()

    results = []
    for name, args, is_score in STEPS:
        if is_score and a.skip_score:
            print(f"[skip] {name}")
            continue
        if a.only and a.only not in name:
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run([PY, "-u", *args], cwd=str(ROOT),
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = (p.stdout or "").strip().splitlines()[-12:]
            for line in tail:
                print("  " + line)
            if p.returncode != 0 and (p.stderr or "").strip():
                print("  [stderr] " + (p.stderr or "").strip().splitlines()[-1])
            results.append((name, p.returncode, time.time() - t0))
        except Exception as e:  # noqa: BLE001
            print(f"  异常：{type(e).__name__}: {e}")
            results.append((name, -1, time.time() - t0))

    print("\n==== 汇总 ====")
    for name, rc, dt in results:
        flag = "ok" if rc == 0 else f"rc={rc}"
        print(f"  {name:14s} {flag:8s} {dt:6.1f}s")


if __name__ == "__main__":
    main()