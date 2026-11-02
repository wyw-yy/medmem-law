"""等 GPU 空出来再开跑（与 datadiet-ehr/tools/wait_then_small.py 对称的守卫）。

背景（2026-11-02）：两个课题共用同一块 8 GB 卡。datadiet-ehr 有个脚本会"等显存空出来再跑"，
但它一旦开跑就会把我们挤进共享显存换页（GPU 100% 但只有 30 W / 180 MHz），反之亦然。
所以这边的长任务也应该先等，而不是直接抢。

用法（在 PowerShell 里）：
    .\\.venv\\Scripts\\python.exe -u build\\wait_gpu.py --need 2000
    .\\.venv\\Scripts\\python.exe -u build\\wait_gpu.py --need 3000 --interval 30 --stable 3 --max-hours 8

行为：每 interval 秒查一次 `nvidia-smi` 的 used 显存，连续 stable 次都满足
"可用 >= need" 才返回 0；超过 max-hours 返回 2（让上层决定是否放弃）。
只做只读查询，不动任何别人的进程。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time


def _query() -> tuple[int, int]:
    """返回 (used_mib, total_mib)。"""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return (0, 0)
    out = subprocess.run(
        [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return (0, 0)
    parts = [p.strip() for p in line[0].split(",")]
    return (int(float(parts[0])), int(float(parts[1])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=2000, help="需要多少 MiB 空闲显存（默认 2000）")
    ap.add_argument("--interval", type=int, default=60, help="轮询间隔秒（默认 60）")
    ap.add_argument("--stable", type=int, default=3, help="连续满足几次才算空闲（默认 3）")
    ap.add_argument("--max-hours", type=float, default=8.0, help="最长等多久（默认 8 h，超时返回 2）")
    a = ap.parse_args()

    t0 = time.time()
    ok = 0
    while True:
        used, total = _query()
        free = total - used if total else 0
        enough = total > 0 and free >= a.need
        ok = ok + 1 if enough else 0
        el = (time.time() - t0) / 60.0
        print(f"[wait_gpu] {time.strftime('%H:%M:%S')} used={used} free={free} "
              f"need={a.need} ok={ok}/{a.stable} waited={el:.1f}min", flush=True)
        if ok >= a.stable:
            print("[wait_gpu] 显存已连续稳定空闲，可以开跑", flush=True)
            return 0
        if el >= a.max_hours * 60:
            print(f"[wait_gpu] 等满 {a.max_hours} h 仍未空闲，放弃（返回 2）", flush=True)
            return 2
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())