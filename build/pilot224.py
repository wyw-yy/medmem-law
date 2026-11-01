"""B9：224px 分辨率 pilot（一个配置，用来回答"结论是不是 64px 特有的"）。

设计（预声明，写进 PROTOCOL R12）
- 数据集：pneumoniamnist（224px 源已在本地，转好的 npz 250 MB）
- N=1000，seed=1，batch=16（8 GB 显存跑 224px 只能这么小）
- **等步数**：8000 优化步（64px 主网格是 20000 步；这里为了 2 h 内跑完降到 40%，
  代价是记忆化可能还没完全起来 —— 所以只看"有没有记忆化"这个定性问题）
- 测量点 8 个（8000/8=1000 步一个），每点采样 256 张（1/256=0.4% 量化步长）
- 输出隔离：`results/pilot224_metrics.csv`，绝不写进主 CSV

用法：python -m build.pilot224
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

ENV = dict(os.environ, MEDMEM_IMAGE_SIZE="224", MEDMEM_BATCH="16",
           PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONPATH=".")

SEED = 21               # 独立 seed：绝不复用 64px 主网格已有的 run tag（否则会"续跑"到 2400 轮直接结束）
RUN = f"pneumoniamnist_n1000_s{SEED}"
EPOCHS = 128          # 128 ep × 1000 张 / 16 = 8000 步
CKPT = 16             # 8 个测量点
SAMPLE_N = 256


def call(args: list[str], log):
    t0 = time.time()
    p = subprocess.run(args, cwd=str(ROOT), env=ENV, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log.write(p.stdout or "")
    log.write(p.stderr or "")
    log.flush()
    print(f"  rc={p.returncode} 用时 {time.time() - t0:.0f}s", flush=True)
    return p.returncode


def main() -> int:
    log_path = ROOT / "build" / "pilot224.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== B9 224px pilot 开始 {time.ctime()} ===\n")
        log.write(f"N=1000 seed={SEED} epochs={EPOCHS} batch=16 sample_n={SAMPLE_N}"
                  f" → {EPOCHS * 1000 // 16} 步\n")
        rc1 = call([PY, "-u", "-m", "train", "--dataset", "pneumoniamnist", "--n", "1000",
                    "--seed", str(SEED), "--epochs", str(EPOCHS), "--ckpt-every", str(CKPT),
                    "--sample-n", str(SAMPLE_N)], log)
        rc2 = call([PY, "-u", "-m", "tools.score_run", "--run", RUN,
                    "--out", "pilot224_metrics.csv", "--fid", "--fid-stride", "2"], log)
        log.write(f"=== 结束 train rc={rc1} score rc={rc2} {time.ctime()} ===\n")
    print(f"日志 {log_path}")
    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    sys.exit(main())