"""接力脚本：等当前夜里那一轮队列跑完，再依次做「控制器验证 → seed3 复现 → 一条龙汇总」。

为什么用"日志哨兵"而不是查进程：上一版 `build/night_chain.ps1` 用
`Get-CimInstance Win32_Process | Where CommandLine -like '*night_queue*'` 判断，
结果**匹配到了查询命令自己**，误判队列已退出并启动了第二个队列去抢同一个 run。
这里改成读 `build/queue_night.log` 里最后那行 "队列结束" —— 这行只有队列正常收尾才会出现。

用法（后台跑）：python build\\next_phases.py
日志：build\\phase3.log
"""
from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "build" / "phase3.log"
QUEUE_LOG = ROOT / "build" / "queue_night.log"
MAX_WAIT_H = 12.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def queue_finished() -> bool:
    if not QUEUE_LOG.exists():
        return False
    tail = QUEUE_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
    return "队列结束" in tail


def run(name: str, args: list[str], timeout_h: float = 0) -> int:
    log(f"=== 开始 {name} ===")
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-u", *args], cwd=str(ROOT), timeout=timeout_h * 3600 or None)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        log(f"{name} 超时（{timeout_h} h），继续下一步")
        return -9
    log(f"=== {name} 结束 rc={rc} 用时 {(time.time()-t0)/60:.1f} min ===")
    return rc


def main() -> None:
    log("接力脚本启动：等 night 队列收尾")
    t0 = time.time()
    while not queue_finished():
        if (time.time() - t0) / 3600 > MAX_WAIT_H:
            log(f"等了 {MAX_WAIT_H} h 还没看到队列收尾，直接继续（可能是被杀了）")
            break
        time.sleep(60)
    log("night 队列已收尾，开始接力")

    run("控制器验证（plan ctl）", ["-m", "tools.night_queue", "--plan", "ctl"])
    run("seed3 复现（plan replicate --seed 3）", ["-m", "tools.night_queue", "--plan", "replicate", "--seed", "3"])
    run("一条龙汇总（finalize --skip-score）", ["build/finalize.py", "--skip-score"], timeout_h=3)
    log("接力全部完成")


if __name__ == "__main__":
    main()