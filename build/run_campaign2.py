"""第四轮跑批（R22）：DP-SGD 基线、容量（base=64）、224px 平台。

三件事按"便宜且信息量大"到"贵"排序，顺序执行，每轮之间停机降温：

A. DP-SGD σ=1.0 · pathmnist N=250 · 1200 轮
B. DP-SGD σ=8.0 · pathmnist N=250 · 1200 轮
C. base=64（≈52M 参数）· pathmnist N=250 · 5125 轮
D. base=64 · pathmnist N=1000 · 2400 轮
E. 224px · bloodmnist N=250 · 续训 6400 → 16000 轮（测自身平台）

产物全部写独立 CSV（`--out`），**不污染主 CSV**：DP 与 base=64 都是"另一种训练方式"，
不属于主网格（主网格是 base=32 / 64px / 非隐私训练）。

用法：
    python -m build.run_campaign2                 # 全跑
    python -m build.run_campaign2 --only A,B      # 只跑指定几组
    python -m build.run_campaign2 --hours 12      # 时间到就停（不打断当前 run）
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "build" / "campaign2.log"
LOCK = ROOT / "build" / "campaign2.lock"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
COOL = 600  # 每轮之隔 10 min，让 GPU 降温


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pid_alive(pid: int) -> bool:
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
        if pid and _pid_alive(pid):
            raise SystemExit(f"已有跑批在跑（pid={pid}）；真要重跑先删 {LOCK}")
        log(f"覆盖陈旧锁（pid={pid} 已不在）")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")


def run(args: list[str], env_extra: dict[str, str] | None = None, tag: str = "") -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    log(f"▶ {tag}  {' '.join(args)}")
    t0 = time.time()
    p = subprocess.run([PY, *args], cwd=str(ROOT), env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    log(f"✔ {tag}  rc={p.returncode}  {(time.time()-t0)/60:.1f} min")
    return p.returncode


def score(run_name: str, out: str, tag: str, eps_note: str = "") -> None:
    run(["-m", "tools.score_run", "--run", run_name, "--fid", "--fid-stride", "4",
         "--out", out], tag=f"{tag} 评分{eps_note}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--hours", type=float, default=0, help="超过这个时长就不再开新 job")
    a = ap.parse_args()
    only = {s.strip().upper() for s in a.only.split(",") if s.strip()}
    t_start = time.time()

    def want(k: str) -> bool:
        return (not only or k in only)

    acquire_lock()
    log(f"===== R22 开始（only={sorted(only) or 'ALL'}）=====")
    failed: list[str] = []

    # ---------------- A/B: DP-SGD 两个噪声水平
    for key, sigma, tag in (("A", "1.0", "dp_s1_n250_s1"), ("B", "8.0", "dp_s8_n250_s1")):
        if not want(key):
            continue
        if a.hours and (time.time() - t_start) / 3600 > a.hours:
            log("时间到，跳过剩余 job"); break
        rc = run(["-m", "train_dp", "--dataset", "pathmnist", "--n", "250", "--seed", "1",
                  "--epochs", "1200", "--ckpt-every", "80", "--sample-n", "512",
                  "--micro", "8", "--noise", sigma, "--clip", "1.0", "--tag", tag],
                 tag=f"{key} DP σ={sigma}")
        if rc != 0:
            failed.append(key)
        else:
            score(tag, f"results/{tag}.csv", key, f"（σ={sigma}）")
        time.sleep(COOL)

    # ---------------- C/D: 容量 base=64
    for key, n, ep, ck in (("C", 250, 5125, 160), ("D", 1000, 2400, 80)):
        if not want(key):
            continue
        if a.hours and (time.time() - t_start) / 3600 > a.hours:
            log("时间到，跳过剩余 job"); break
        tag = f"b64_pathmnist_n{n}_s1"
        rc = run(["-m", "train", "--dataset", "pathmnist", "--n", str(n), "--seed", "1",
                  "--epochs", str(ep), "--ckpt-every", str(ck), "--sample-n", "512", "--tag", tag],
                 env_extra={"MEDMEM_BASE_CH": "64", "MEDMEM_BATCH": "32"},
                 tag=f"{key} base=64 N={n}")
        if rc != 0:
            failed.append(key)
        else:
            score(tag, f"results/{tag}.csv", key)
        time.sleep(COOL)

    # ---------------- E: 224px 自身平台（续训到 16000 轮）
    if want("E"):
        if not (a.hours and (time.time() - t_start) / 3600 > a.hours):
            tag = "bloodmnist_n250_s21_a4"
            rc = run(["-m", "train", "--dataset", "bloodmnist", "--n", "250", "--seed", "21",
                      "--epochs", "16000", "--ckpt-every", "400", "--sample-n", "128", "--tag", tag],
                     env_extra={"MEDMEM_IMAGE_SIZE": "224", "MEDMEM_BATCH": "16", "MEDMEM_ACCUM": "4",
                                "MEDMEM_SAMPLE_BATCH": "16"},
                     tag="E 224px 续训到 16000")
            if rc != 0:
                failed.append("E")
            else:
                score(tag, "results/pilot224_accum_blood_n250.csv", "E")
        else:
            log("时间到，跳过 E")

    log(f"===== R22 结束：失败 {failed or '无'}；总 {(time.time()-t_start)/3600:.1f} h =====")
    if LOCK.exists():
        LOCK.unlink()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
