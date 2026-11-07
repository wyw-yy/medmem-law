"""等今晚链条结束后，依次做两件事：

1. **64px 同子集对照**：bloodmnist N=250 seed=21（与 224px 那档同一批图）跑满 5,125 轮。
   为什么必须有它：224px 那档是 seed 21，而 64px 主网格的血样只有 seed 1；N=250 的 seed 间
   发作差异可达数十倍，拿 seed 1 当对照会把"子集差异"算到"分辨率"头上。
   用 `--tag bloodmnist_n250_s21_64` 与 224px 那档分开，避免 cache/samples 同名覆盖。

2. **224px 续训**：把已有的 bloodmnist N=250 seed=21（tag `bloodmnist_n250_s21`）从 E=1600
   续训到 E=6400，再重评分 —— 用来测出 224px 自己的平台（见 docs/NOTES.md §16）。
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
LOG = ROOT / "build" / "control_then_extend.log"

ENV224 = {"MEDMEM_IMAGE_SIZE": "224", "MEDMEM_BATCH": "16", "MEDMEM_SAMPLE_BATCH": "16"}
ENV64 = {"MEDMEM_IMAGE_SIZE": "64", "MEDMEM_BATCH": "64", "MEDMEM_SAMPLE_BATCH": "64"}


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
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def main() -> int:
    log("=== 等待今晚链条结束 ===")
    waited = 0
    while chain_running():
        if waited % 900 == 0:
            log(f"  链条仍在跑（已等 {waited // 60} 分钟）")
        time.sleep(60)
        waited += 60
    log(f"=== 链条已结束（等了 {waited // 60} 分钟）===")

    # 1) 64px 同子集对照
    log("=== 1/2 64px 同子集对照：bloodmnist N=250 seed=21 → 5125 轮 ===")
    rc = run([str(PY), "-u", "-m", "train", "--dataset", "bloodmnist", "--n", "250",
              "--seed", "21", "--epochs", "5125", "--ckpt-every", "160", "--sample-n", "128",
              "--tag", "bloodmnist_n250_s21_64"], ENV64)
    log(f"=== train(64px) 结束 rc={rc} ===")
    if rc != 0:
        return rc
    rc = run([str(PY), "-u", "-m", "tools.score_run", "--run", "bloodmnist_n250_s21_64",
              "--fid", "--fid-stride", "4", "--out", "pilot64_blood_n250_s21.csv"], ENV64)
    log(f"=== score(64px) 结束 rc={rc} ===")
    if rc != 0:
        return rc

    # 2) 224px 续训到 E=6400
    log("=== 2/2 224px 续训：bloodmnist N=250 seed=21 → E=6400 ===")
    rc = run([str(PY), "-u", "-m", "train", "--dataset", "bloodmnist", "--n", "250",
              "--seed", "21", "--epochs", "6400", "--ckpt-every", "200", "--sample-n", "128"],
             ENV224)
    log(f"=== train(224px) 结束 rc={rc} ===")
    if rc != 0:
        return rc
    rc = run([str(PY), "-u", "-m", "tools.score_run", "--run", "bloodmnist_n250_s21",
              "--fid", "--fid-stride", "4", "--out", "pilot224_blood_n250.csv"], ENV224)
    log(f"=== score(224px) 结束 rc={rc} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())