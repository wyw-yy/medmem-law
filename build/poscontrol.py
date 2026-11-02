"""阳性对照：在 N=64 的极小训练集上，模型必然背诵 —— 用来验证"度量本身在该分辨率下有效"。

动机：B9（224px pilot）跑出 `M=0`，有两种可能 ——（i）真的还没记忆化；（ii）度量在这个分辨率下失效。
阳性对照把 (i) 排掉：如果 N=64 时 `M → 1`，说明同一个度量管线在 224px 上能检出记忆化，
那么 pilot 的 `M=0` 就只能解释成"没背"。

为什么是 N=64 而不是更小：白化截断维 `k = max(32, min(d, N//2))`，
N=16 时 `k=32 > 秩(15)` → 又回到"零空间被放大"的退化情形（NOTES §1.2）。
N=64 给 `k=32 ≤ 秩(63)`，是这套度量能正常工作的最小规模附近。

两个分辨率各跑一遍：
- 64px：batch 64 → 1 步/epoch；2000 步 = 2000 epoch
- 224px：batch 16 → 4 步/epoch；2000 步 = 500 epoch

用法：python -m build.poscontrol
输出：`results/poscontrol_64.csv`、`results/poscontrol_224.csv`
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

CASES = [
    # (分辨率, batch, seed, epochs, ckpt_every)
    (64, 64, 31, 2000, 400),
    (224, 16, 32, 500, 100),
]


def call(args: list[str], env: dict, tag: str) -> int:
    t0 = time.time()
    p = subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-3:]
    print(f"[{tag}] rc={p.returncode} {time.time()-t0:.0f}s  " + " | ".join(tail), flush=True)
    if p.returncode != 0:
        print((p.stderr or "")[-1500:], flush=True)
    return p.returncode


def main() -> int:
    bad = 0
    for size, batch, seed, epochs, ckpt in CASES:
        run = f"pneumoniamnist_n64_s{seed}"
        env = dict(os.environ, MEDMEM_IMAGE_SIZE=str(size), MEDMEM_BATCH=str(batch),
                   MEDMEM_SAMPLE_BATCH="16" if size >= 224 else "64",
                   PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONPATH=".")
        print(f"=== {size}px  run={run}  batch={batch}  epochs={epochs}  ckpt={ckpt} ===", flush=True)
        rc = call([PY, "-u", "-m", "train", "--dataset", "pneumoniamnist", "--n", "64",
                   "--seed", str(seed), "--epochs", str(epochs), "--ckpt-every", str(ckpt),
                   "--sample-n", "128"], env, f"train{size}")
        bad |= rc
        rc = call([PY, "-u", "-m", "tools.score_run", "--run", run,
                   "--out", f"poscontrol_{size}.csv"], env, f"score{size}")
        bad |= rc
    print("完成" if not bad else "有非零返回", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())