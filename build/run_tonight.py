"""今晚的跑批驱动（顺序执行，避开 8 GB 显存争抢）。

为什么需要这个脚本而不是几条手敲命令：
  1. `IMAGE_SIZE` / `BATCH` / `SAMPLE_BATCH` 是**进程级**环境变量（config.py 在 import 时读），
     224px 与 64px 的作业不能塞进同一个进程；
  2. 224px 的产物必须与主 CSV 隔离（`--out pilot224_*.csv`），否则同一 (run, epoch) 会撞车；
  3. 每一步失败都要留下明确日志，不能静默跳过。

顺序（按"能改变结论"排序）：
  1. 224px · bloodmnist · N=250  · seed 21 · 25k 步 → 与 64px 同 N 对照（64px 时 M_max≈0.97）
  2. 224px · bloodmnist · N=1000 · seed 21 · 25k 步 → 与 64px 同 N 对照（64px 时 M_max≈0.34）
  3. 64px · pathmnist · N=128 · seed 1 · 20k 步     → 把 onset/N 常数的成立下界从 250 压到 128
  4. 64px · pathmnist · N=192 · seed 1 · 20k 步     → 同上，中间档
  5. ResNet50 复算 N=2000/3000/4000（复用已缓存样本，不重训）

为什么不用 pathmnist 做 224px：它的 224 源转成 npz 有 ~14 GB，`load_split` 要整块读进内存，
31 GB 机器上有风险；bloodmnist 在 64px 上信号最强（N=250 时 M≈0.97），且 224 版只有 1.5 GB。

用 `python -m build.run_tonight` 启动；日志 `build/tonight.log`。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "build" / "tonight.log"
LOCK = ROOT / "build" / "tonight.lock"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def _pid_alive(pid: int) -> bool:
    """Windows 下判断 pid 是否还活着（OpenProcess + GetExitCodeProcess == STILL_ACTIVE）。"""
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok) and code.value == 259


def acquire_lock() -> bool:
    """单实例锁：两个 run_tonight 同时跑会抢 8 GB 显存（会退化成共享内存换页，慢十倍）。"""
    import atexit

    if LOCK.exists():
        try:
            other = int(LOCK.read_text(encoding="utf-8").strip())
        except Exception:  # noqa: BLE001
            other = -1
        if other > 0 and _pid_alive(other):
            print(f"[拒绝启动] 另一个 run_tonight 正在跑（pid={other}）。"
                  f"先确认它是不是要保留，再删 {LOCK} 或杀进程。", flush=True)
            return False
        print(f"[锁] 发现陈旧锁 pid={other}（已退出），覆盖", flush=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: LOCK.unlink(missing_ok=True))
    return True

ENV_224 = {"MEDMEM_IMAGE_SIZE": "224", "MEDMEM_BATCH": "16", "MEDMEM_SAMPLE_BATCH": "16"}
ENV_64 = {"MEDMEM_IMAGE_SIZE": "64", "MEDMEM_BATCH": "64", "MEDMEM_SAMPLE_BATCH": "64"}


def log(line: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    text = f"[{stamp}] {line}"
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def run(name: str, cmd: list[str], env_extra: dict[str, str]) -> bool:
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT)
    log(f"=== 开始 {name} ===")
    log("  " + " ".join(cmd) + "   env=" + str(env_extra))
    t0 = time.time()
    with LOG.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    ok = proc.returncode == 0
    log(f"=== {name} 结束 rc={proc.returncode} 用时 {dt:.1f} min ===")
    return ok


def ensure_blood224() -> bool:
    npz = ROOT / "data" / "raw" / "bloodmnist_224.npz"
    if npz.is_file() and npz.stat().st_size > 1e9:
        log(f"bloodmnist_224.npz 已就绪（{npz.stat().st_size / 1e6:.0f} MB）")
        return True
    log("bloodmnist_224.npz 缺失 → 从本地 zip 转换")
    return run("转换 bloodmnist 224px",
               [PY, "-u", "-m", "build.get_hf64", "--name", "bloodmnist", "--size", "224"],
               {})


def clean(run_tag: str) -> None:
    for p in (ROOT / "cache" / "samples" / run_tag, ROOT / "cache" / "ckpt" / run_tag,
              ROOT / "results" / "runs" / f"{run_tag}.json"):
        if p.exists():
            if p.is_dir():
                import shutil

                shutil.rmtree(p)
            else:
                p.unlink()
            log(f"  清理残留 {p.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-224", action="store_true")
    ap.add_argument("--skip-64", action="store_true")
    ap.add_argument("--skip-rescore", action="store_true")
    a = ap.parse_args()

    log("########## 今晚跑批开始 ##########")
    if not acquire_lock():
        return 3
    failures: list[str] = []

    if not a.skip_224:
        have = ensure_blood224()
        # 1) 224px bloodmnist N=250 · 25k 步
        if have:
            clean("bloodmnist_n250_s21")
            if run("train 224 bloodmnist N=250 s21",
                   [PY, "-u", "-m", "train", "--dataset", "bloodmnist", "--n", "250",
                    "--seed", "21", "--epochs", "1600", "--ckpt-every", "200", "--sample-n", "128"],
                   ENV_224):
                run("score 224 bloodmnist N=250",
                    [PY, "-u", "-m", "tools.score_run", "--run", "bloodmnist_n250_s21",
                     "--out", "pilot224_blood_n250.csv", "--fid", "--fid-stride", "2"],
                    ENV_224)
            else:
                failures.append("train224 blood N=250")

            # 2) 224px bloodmnist N=1000 · 25k 步（400 ep × 62.5 步）
            clean("bloodmnist_n1000_s21")
            if run("train 224 bloodmnist N=1000 s21",
                   [PY, "-u", "-m", "train", "--dataset", "bloodmnist", "--n", "1000",
                    "--seed", "21", "--epochs", "400", "--ckpt-every", "50", "--sample-n", "128"],
                   ENV_224):
                run("score 224 bloodmnist N=1000",
                    [PY, "-u", "-m", "tools.score_run", "--run", "bloodmnist_n1000_s21",
                     "--out", "pilot224_blood_n1000.csv", "--fid", "--fid-stride", "2"],
                    ENV_224)
            else:
                failures.append("train224 blood N=1000")
        else:
            failures.append("bloodmnist 224 数据未就绪")

    if not a.skip_64:
        clean("pathmnist_n128_s1")
        if run("train 64 pathmnist N=128 s1",
               [PY, "-u", "-m", "train", "--dataset", "pathmnist", "--n", "128",
                "--seed", "1", "--epochs", "10000", "--ckpt-every", "1250"],
               ENV_64):
            run("score 64 pathmnist N=128",
                [PY, "-u", "-m", "tools.score_run", "--run", "pathmnist_n128_s1",
                 "--fid", "--fid-stride", "4"], ENV_64)
        else:
            failures.append("train64 N=128")

        clean("pathmnist_n192_s1")
        if run("train 64 pathmnist N=192 s1",
               [PY, "-u", "-m", "train", "--dataset", "pathmnist", "--n", "192",
                "--seed", "1", "--epochs", "6675", "--ckpt-every", "825"],
               ENV_64):
            run("score 64 pathmnist N=192",
                [PY, "-u", "-m", "tools.score_run", "--run", "pathmnist_n192_s1",
                 "--fid", "--fid-stride", "4"], ENV_64)
        else:
            failures.append("train64 N=192")

    if not a.skip_rescore:
        run("resnet50 复算 N=2000/3000/4000",
            [PY, "-u", "-m", "build.rescore_encoder", "--encoder", "resnet50",
             "--runs", "pathmnist_n2000_s1", "pathmnist_n3000_s1", "pathmnist_n4000_s1"],
            ENV_64)

    if failures:
        log("有步骤失败: " + ", ".join(failures))
    log("########## 今晚跑批结束 ##########")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())