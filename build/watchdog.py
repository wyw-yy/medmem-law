"""值守脚本：盯住夜里的长链条，出现异常或跑完时打印一份体检报告。

判据（任一成立就退出并打印）：
  1. `build/phase3.log` 里出现 `汇总`（finalize 跑完了，整条链结束）；
  2. `build/queue_night.log` 超过 STALL_MIN 分钟没有新内容（训练卡死/进程被杀）；
  3. 超过 MAX_H 小时（兜底，避免永久挂住）。

为什么需要它：链条是我按阶段手动接力的（night 队列 → phase3 → finalize），
中间任何一环挂掉都不会有人告诉我；而 GPU 被功耗墙压到 202 MHz 时
单 run 要 40 分钟，人也不该一直盯着。
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "build" / "queue_night.log"
P3 = ROOT / "build" / "phase3.log"
POLL = 300
STALL_MIN = 25
MAX_H = 7.0


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def _tail(p: Path, n: int = 6) -> list[str]:
    if not p.exists():
        return []
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


def report(reason: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] 值守退出：{reason}")
    for name, p in (("queue_night.log", Q), ("phase3.log", P3)):
        print(f"\n--- {name} ---")
        for line in _tail(p, 8):
            print("  " + line)
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import pandas as pd,pathlib;"
             "p=pathlib.Path('results/metrics_long.csv');"
             "d=pd.read_csv(p) if p.exists() else None;"
             "print(d.groupby(['dataset','n','seed']).size().to_string() if d is not None else 'no csv')"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        print("\n--- metrics 行数 ---")
        print(out.stdout.strip() or out.stderr.strip())
    except Exception as e:  # noqa: BLE001
        print(f"  (读 CSV 失败：{e})")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print("\n--- GPU ---\n  " + (gpu.stdout.strip() or gpu.stderr.strip()))


def main() -> None:
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] 值守开始（每 {POLL}s 检查一次）")
    while True:
        if P3.exists() and "汇总" in P3.read_text(encoding="utf-8", errors="replace"):
            report("链条全部完成（finalize 已汇总）")
            return
        age = (time.time() - _mtime(Q)) / 60.0
        if age > STALL_MIN:
            report(f"queue_night.log 已 {age:.0f} 分钟没有更新（阈值 {STALL_MIN}）")
            return
        if (time.time() - t0) / 3600.0 > MAX_H:
            report(f"超过 {MAX_H:g} 小时兜底退出")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()