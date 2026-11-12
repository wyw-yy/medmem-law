"""计划任务用的"复活"探针：每 10 分钟跑一次，只在真卡死时重启队列。

为什么需要它：DSH 里的 cron 看门狗依赖 DSH 会话本身活着。11-11 22:54 机器被
合盖/手动睡眠 18.5 h，DSH 跟着挂起，那段时间没有任何东西在检查队列，醒来后
GPU 上下文已经死了、训练进程还挂在 e480 上白占着 7.5 GB 显存。

这个脚本由 Windows 计划任务（`schtasks /tn medmem_revive`）每 10 分钟调一次，
**和 DSH 是否活着无关**。它本身很轻（几秒），真正的判断与重启都在
`build.watch_campaign3 --fix` 里：只有"锁 pid 不在了"或"90 分钟没有新检查点"
才会动手，正常训练时它是纯粹的空转。

调用方式（计划任务里）：
    .venv\\Scripts\\python.exe -m build.revive_task
"""

from __future__ import annotations

import subprocess
import sys
import time

from config import BUILD, ROOT

PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = BUILD / "revive.log"


def _log(msg: str) -> None:
    line = time.strftime("[%m-%d %H:%M:%S] ") + msg
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    t0 = time.time()
    p = subprocess.run([str(PY), "-m", "build.watch_campaign3", "--fix"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=900,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    head = next((ln for ln in out.splitlines() if ln.startswith("VERDICT")), "VERDICT: ?")
    took = time.time() - t0
    # 只在非 OK 时写日志，避免每 10 分钟刷一行噪音
    if p.returncode != 0 or "--fix" in out:
        _log(head + ("  |  " + [ln for ln in out.splitlines() if "--fix" in ln][0]
                     if "--fix" in out else ""))
    elif took > 120:  # 探针本身变慢也要记一笔（GPU 可能被别的进程占着）
        _log(f"{head}  |  探针耗时 {took:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())