"""第三轮跑批：分辨率阶梯（224 / 128，均用梯度累积对齐到等效 batch 64）+ seed 补到 12/档。

为什么要这三组（预声明见 docs/PROTOCOL.md §24.8 R20）：
  A  224px、batch 16 × accum 4 → 等效 batch 64。这是唯一能解耦"分辨率 vs batch"的实验：
     上一轮 224px 只能 batch 16，比 64px 主网格少 4 倍重复次数，M=0 有两种解释。
  B  128px、batch 32 × accum 2 → 等效 batch 64。让分辨率阶梯成为 64/128/224 三点，
     否则只能说"至少 8 倍"，说不出趋势。
  C  pathmnist N=250/500/1000 各补 seed 10/11/12。最差 seed 安全表与 σ_log 现在靠 7–9 个 seed。

每轮之间停机休息（默认 12 min），跑完在 build/campaign.log 里写总结。
单实例锁 build/campaign.lock，防止两个队列抢同一张卡。

用法：
    python -m build.run_campaign                # 全部三轮
    python -m build.run_campaign --rounds A     # 只跑某几轮
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import config as C

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = C.BUILD / "campaign.log"
LOCK = C.BUILD / "campaign.lock"

# 主网格最后采用的 epoch 预算（与 results/runs/*.json 里 pathmnist 的 epochs_target 一致）
FINAL = {250: 5125, 500: 2550, 1000: 2400}
CKPT = {250: 160, 500: 80, 1000: 75}


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def acquire_lock() -> None:
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip() or 0)
        except ValueError:
            pid = 0
        if _pid_alive(pid) and pid != os.getpid():
            log(f"[abort] {LOCK.name} 里是活进程 {pid}，已有队列在跑。要强制启动先删锁。")
            sys.exit(3)
    LOCK.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK.unlink(missing_ok=True))


def run(cmd: list[str], env_extra: dict[str, str], what: str) -> None:
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    log(f"▶ {what}")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        log(f"  ✗ rc={r.returncode} 用时 {dt:.1f} min；stderr 尾部：\n{r.stderr[-1500:]}")
        raise SystemExit(1)
    log(f"  ✓ 用时 {dt:.1f} min")


def score_round(env_extra: dict[str, str], run_name: str, out: str | None) -> None:
    cmd = [PY, "-m", "tools.score_run", "--run", run_name, "--fid", "--fid-stride", "4"]
    if out:
        cmd += ["--out", out]
    run(cmd, env_extra, f"score {run_name}" + (f" -> {out}" if out else ""))


def round_a() -> None:
    """224px × accum 4（等效 batch 64），E=6400。"""
    env = {"MEDMEM_IMAGE_SIZE": "224", "MEDMEM_BATCH": "16", "MEDMEM_ACCUM": "4",
           "MEDMEM_SAMPLE_BATCH": "16"}
    tag = "bloodmnist_n250_s21_a4"
    run([PY, "-m", "train", "--dataset", "bloodmnist", "--n", "250", "--seed", "21",
         "--epochs", "6400", "--ckpt-every", "200", "--sample-n", "128", "--tag", tag],
        env, f"224px accum4 train {tag} (6400 epochs, effective batch 64)")
    score_round(env, tag, "results/pilot224_accum_blood_n250.csv")


def round_b() -> None:
    """128px × accum 2（等效 batch 64），E=6400。"""
    env = {"MEDMEM_IMAGE_SIZE": "128", "MEDMEM_BATCH": "32", "MEDMEM_ACCUM": "2",
           "MEDMEM_SAMPLE_BATCH": "32"}
    tag = "bloodmnist_n250_s21_128"
    run([PY, "-m", "train", "--dataset", "bloodmnist", "--n", "250", "--seed", "21",
         "--epochs", "6400", "--ckpt-every", "200", "--sample-n", "256", "--tag", tag],
        env, f"128px accum2 train {tag} (6400 epochs, effective batch 64)")
    score_round(env, tag, "results/pilot128_blood_n250.csv")


def round_c() -> None:
    """pathmnist 三个可测档各补 seed 10/11/12（64px 主网格配置，产物进主 CSV）。"""
    env = {"MEDMEM_IMAGE_SIZE": "64", "MEDMEM_BATCH": "64", "MEDMEM_ACCUM": "1",
           "MEDMEM_SAMPLE_BATCH": "64"}
    for n in (250, 500, 1000):
        for seed in (10, 11, 12):
            tag = f"pathmnist_n{n}_s{seed}"
            if (C.RESULTS / "runs" / f"{tag}.json").exists():
                log(f"  · 跳过 {tag}（已有 run json）")
                continue
            run([PY, "-m", "train", "--dataset", "pathmnist", "--n", str(n),
                 "--seed", str(seed), "--epochs", str(FINAL[n]),
                 "--ckpt-every", str(CKPT[n])], env,
                f"64px {tag} ({FINAL[n]} epochs)")
            score_round(env, tag, None)


ROUNDS = {"A": ("224px accum4", round_a), "B": ("128px accum2", round_b),
          "C": ("seed 10-12", round_c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", default="ABC")
    ap.add_argument("--rest", type=int, default=720, help="轮间休息秒数")
    a = ap.parse_args()

    acquire_lock()
    names = [c for c in a.rounds.upper() if c in ROUNDS]
    log("=" * 30)
    log(f"第三轮跑批开始：{names}，轮间休息 {a.rest}s")
    t0 = time.time()
    for i, key in enumerate(names):
        title, fn = ROUNDS[key]
        log(f"===== 轮 {key}：{title} =====")
        fn()
        if i + 1 < len(names):
            log(f"休息 {a.rest}s（让显卡降温 / 让 Windows 回收显存）")
            time.sleep(a.rest)
    dt = (time.time() - t0) / 3600
    log(f"########## 第三轮跑批结束：{dt:.1f} h ##########")


if __name__ == "__main__":
    main()