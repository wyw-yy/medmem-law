"""等今晚链条（run_tonight.py）释放锁之后，把 224px 的 bloodmnist N=250 续训到 E=6400 并重新评分。

为什么需要它：224px 只能用 batch 16，25k 步只等于 1,600 轮；而 64px 的同一档跑到 5,125 轮、
平台 0.9707。要做"等 epoch"的比较，必须把 224px 也推到大 epoch 区间（见 docs/NOTES.md §16）。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOCK = ROOT / "build" / "tonight.lock"
LOG = ROOT / "build" / "extend224.log"

RUN = "bloodmnist_n250_s21"
EPOCHS = 6400
ENV224 = {"MEDMEM_IMAGE_SIZE": "224", "MEDMEM_BATCH": "16", "MEDMEM_SAMPLE_BATCH": "16"}


def _alive(pid: int) -> bool:
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def chain_running() -> bool:
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    return _alive(pid)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], env_extra: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONIOENCODING"] = "utf-8"
    log("  " + " ".join(cmd) + f"   env={env_extra}")
    p = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return p.returncode


def main() -> int:
    log("=== 等待今晚链条结束 ===")
    waited = 0
    while chain_running():
        if waited % 600 == 0:
            log(f"  链条仍在跑（已等 {waited // 60} 分钟）")
        time.sleep(60)
        waited += 60
    log(f"=== 链条已结束（等了 {waited // 60} 分钟），开始续训 224px {RUN} → E={EPOCHS} ===")

    rc = run([str(PY), "-u", "-m", "train", "--dataset", "bloodmnist", "--n", "250",
              "--seed", "21", "--epochs", str(EPOCHS), "--ckpt-every", "200", "--sample-n", "128"],
             ENV224)
    log(f"=== train 结束 rc={rc} ===")
    if rc != 0:
        return rc

    rc = run([str(PY), "-u", "-m", "tools.score_run", "--run", RUN, "--fid", "--fid-stride", "4",
              "--out", "pilot224_blood_n250.csv"], ENV224)
    log(f"=== score 结束 rc={rc} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())