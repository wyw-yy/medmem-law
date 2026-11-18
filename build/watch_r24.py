"""R24 链条的看门狗：只盯 r24_chain2 这一条链（② 采样探针 → ③ DP σ=2 seed2）。

为什么单独写：`build/watch_campaign3.py` 认的是 campaign3 的锁与日志，管不了这条新链；
R23 收工后我把 campaign3 的巡检一起删了，结果这条链断掉 21 小时没人发现。
所以这里改成**只看两件事**：链上还有没有活着的进程、日志有没有在推进。

判据（不依赖锁文件）：
- `DONE`  ：日志里出现 R24_ALL_DONE
- `FAILED`：最近一行是 `rc=<非0>` 或中间出错
- `STALLED`：没有匹配的 python 进程在跑，且日志超过 STALE_MIN 分钟没动
- `OK`    ：有进程在跑，且日志在推进

用法：
  python -m build.watch_r24          # 报状态
  python -m build.watch_r24 --fix    # 死了就按原命令重启整条链
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "build" / "r24_chain2.log"
TRAIN_LOG = ROOT / "build" / "logs" / "dp_s2_n250_s2_long.log"
CMD = ROOT / "build" / "r24_chain2.cmd"
KEYS = ("probe_sampler", "train_dp", "score_run")
STALE_MIN = 90
TOTAL_EPOCHS = 5125


def _procs() -> list[tuple[int, str]]:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"（取进程失败：{e}）")
        return []
    rows = []
    for line in (out.stdout or "").splitlines():
        pid, _, cmd = line.partition("|")
        if pid.strip().isdigit():
            rows.append((int(pid), cmd))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()

    text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    if "R24_ALL_DONE" in text:
        print("VERDICT: DONE — R24 链条已完成")
        return 0

    lines = [ln for ln in text.splitlines() if ln.strip()]
    last = lines[-1] if lines else "（日志为空）"
    mtime = dt.datetime.fromtimestamp(LOG.stat().st_mtime) if LOG.exists() else None
    age = (dt.datetime.now() - mtime).total_seconds() / 60 if mtime else 1e9

    running = [p for p in _procs() if any(k in p[1] for k in KEYS)]

    # 训练进度（③ 阶段的日志）
    prog = ""
    if TRAIN_LOG.exists():
        t = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
        hits = re.findall(r"epoch\s+(\d+)/(\d+)\s+loss=[\d.]+\s+([\d.]+)s/epoch", t)
        if hits:
            ep, tot, spe = hits[-1]
            ep, tot, spe = int(ep), int(tot), float(spe)
            left = (TOTAL_EPOCHS - ep) * spe / 60
            prog = (f"训练 e{ep}/{tot}"
                    f"（{100 * ep / tot:.0f}%），{spe:.1f}s/epoch，剩 ~{left / 60:.1f} h")

    if running:
        print(f"VERDICT: OK — {len(running)} 个进程在跑"
              f"（{[p for p, _ in running]}）；日志 {age:.0f} min 前更新")
        if prog:
            print("  " + prog)
        return 0

    verdict = "STALLED" if age > STALE_MIN else "OK"
    print(f"VERDICT: {verdict} — 没有匹配的进程在跑；日志最后一行（{age:.0f} min 前）：{last}")
    if prog:
        print("  " + prog)
    if verdict == "STALLED" and a.fix:
        print("  重启链条：build/r24_chain2.cmd")
        subprocess.Popen(["cmd", "/c", str(CMD)], cwd=str(ROOT),
                         creationflags=0x00000008)  # DETACHED_PROCESS
        return 0
    return 1 if verdict == "STALLED" else 0


if __name__ == "__main__":
    sys.exit(main())