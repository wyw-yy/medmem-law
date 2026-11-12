"""第五轮跑批（R23）：把 R22 留下的四个口子跑到自然收敛。

R22 只跑到 1,200 轮/固定 C=1.0/两个 σ，只能说"没看到"，不能说"不会"。这一轮补齐：

F. DP σ=8.0 · 5,125 轮        —— DP 是推迟 onset 还是永久消除？
G. DP σ=2.0 · 1,200 轮        —— 中间强度有没有可用窗口？
H. DP σ=4.0 · 1,200 轮        —— 同上
I. 裁剪范数 C=0.1 / 0.5 / 2.0（σ=8 固定）—— M=0 是不是裁剪太狠的假象？
J. base=48（≈30M）N=250 / N=1000 —— 容量-平台从两点变三点
K. 224px · N=1000 · 9,600 轮  —— 分辨率延迟是否随 N 放大？

产物全部独立 CSV，不进主律拟合。用法：
    python -m build.run_campaign3              # 全跑（约 39 h）
    python -m build.run_campaign3 --only F,G   # 只跑指定几组
    python -m build.run_campaign3 --hours 12   # 时间到就停（不打断当前 run）
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
LOG = ROOT / "build" / "campaign3.log"
LOCK = ROOT / "build" / "campaign3.lock"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
COOL = 1800  # 每轮之间停机 30 min（用户要求：每个任务之间让电脑歇半小时）


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
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
    dt = (time.time() - t0) / 60
    log(f"✔ {tag}  rc={p.returncode}  {dt:.1f} min")
    if p.returncode != 0:
        # 非零退出必须立刻停：R22 的教训是评分脚本失败被吞掉，四份 CSV 根本没生成。
        raise SystemExit(f"✘ {tag} 失败（rc={p.returncode}），整条链停止")
    return p.returncode


def score(run_name: str, out: str, tag: str,
          dataset: str | None = None, n: int | None = None, seed: int | None = None) -> None:
    """评分。

    DP/容量档的 tag 里没有数据集名（`dp_s8_n250_s1_long`），`score_run` 的
    `parse_tag` 猜不出来 → 必须显式传 `--dataset/--n/--seed`。
    （R23 的 F 组就是因为这个在训练完成后评分 rc=1，链条停了一次。）
    """
    args = ["-m", "tools.score_run", "--run", run_name, "--fid", "--fid-stride", "4",
            "--out", out]
    if dataset:
        args += ["--dataset", dataset, "--n", str(n), "--seed", str(seed)]
    run(args, tag=f"{tag} 评分")


def dp(key: str, sigma: str, clip: str, epochs: int, tag: str) -> None:
    run(["-m", "train_dp", "--dataset", "pathmnist", "--n", "250", "--seed", "1",
         "--epochs", str(epochs), "--ckpt-every", str(320 if epochs > 2000 else 80),
         "--sample-n", "512", "--micro", "8", "--noise", sigma, "--clip", clip, "--tag", tag],
        tag=f"{key} DP σ={sigma} C={clip} {epochs} 轮")
    score(tag, f"results/{tag}.csv", key, dataset="pathmnist", n=250, seed=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--hours", type=float, default=0, help="超过这个时长就不再开新 job")
    a = ap.parse_args()
    only = {s.strip().upper() for s in a.only.split(",") if s.strip()}
    t_start = time.time()

    seen: set[str] = set()

    def want(k: str) -> bool:
        """--only 支持组字母（I）与子键（I2），两边互相前缀匹配都算。

        11-12 踩过：--only I2,I3,... 时块守卫写的是 want("I")，"I2" != "I"
        → 整个 I 块被静默跳过、直接从 K 开始跑，而且日志里看不出少跑了东西。
        """
        if not only:
            return True
        hit = any(o == k or o.startswith(k) or k.startswith(o) for o in only)
        if hit:
            seen.add(k)
        return hit

    def out_of_time() -> bool:
        return bool(a.hours) and (time.time() - t_start) / 3600 > a.hours

    acquire_lock()
    log(f"===== R23 开始（only={sorted(only) or 'ALL'}）=====")

    # ---------------- F: DP 跑满整条预算（σ=8）
    if want("F") and not out_of_time():
        dp("F", "8.0", "1.0", 5125, "dp_s8_n250_s1_long")
        time.sleep(COOL)

    # ---------------- G/H: 中间噪声强度
    for key, sigma, tag in (("G", "2.0", "dp_s2_n250_s1"), ("H", "4.0", "dp_s4_n250_s1")):
        if want(key) and not out_of_time():
            dp(key, sigma, "1.0", 1200, tag)
            time.sleep(COOL)

    # ---------------- I: 裁剪范数（σ=8 固定，只动 C）
    for key, clip, tag in (("I1", "0.1", "dp_s8_c01_n250_s1"),
                           ("I2", "0.5", "dp_s8_c05_n250_s1"),
                           ("I3", "2.0", "dp_s8_c20_n250_s1")):
        if want(key) and not out_of_time():
            dp(key, "8.0", clip, 1200, tag)
            time.sleep(COOL)

    # ---------------- J: 容量三点（base=48）的中间那一档
    for key, n, ep, ck in (("J1", 250, 5125, 160), ("J2", 1000, 2400, 80)):
        if want("J") and not out_of_time():
            tag = f"b48_pathmnist_n{n}_s1"
            run(["-m", "train", "--dataset", "pathmnist", "--n", str(n), "--seed", "1",
                 "--epochs", str(ep), "--ckpt-every", str(ck), "--sample-n", "512", "--tag", tag],
                env_extra={"MEDMEM_BASE_CH": "48", "MEDMEM_BATCH": "32"},
                tag=f"{key} base=48 N={n}")
            score(tag, f"results/{tag}.csv", key)
            time.sleep(COOL)

    # ---------------- K: 224px × N=1000（分辨率延迟是否随 N 放大）
    if want("K") and not out_of_time():
        tag = "bloodmnist_n1000_s21_a4"
        run(["-m", "train", "--dataset", "bloodmnist", "--n", "1000", "--seed", "21",
             "--epochs", "9600", "--ckpt-every", "800", "--sample-n", "128", "--tag", tag],
            env_extra={"MEDMEM_IMAGE_SIZE": "224", "MEDMEM_BATCH": "16", "MEDMEM_ACCUM": "4",
                       "MEDMEM_SAMPLE_BATCH": "16"},
            tag="K 224px N=1000 到 9600 轮")
        score(tag, "results/pilot224_blood_n1000_a4.csv", "K")

    # ---------------- L/M/N: DP 的剂量-反应（σ=1/2/4 全部跑满 5,125 轮）
    # 只有 σ=8 跑满过；1,200 轮时无隐私基线本身 M 才 0.06，短预算下"DP 的 M=0"没有说服力。
    # 四档同预算（σ=1/2/4/8 × 5,125 轮）才能画出一条真正的"隐私强度-记忆化-质量"曲线。
    for key, sigma, tag in (("L", "2.0", "dp_s2_n250_s1_long"),
                            ("M", "4.0", "dp_s4_n250_s1_long"),
                            ("N", "1.0", "dp_s1_n250_s1_long")):
        if want(key) and not out_of_time():
            dp(key, sigma, "1.0", 5125, tag)
            time.sleep(COOL)

    # ---------------- O: 128px × N=1000（补全"分辨率 × N"的交互）
    # 现在只有 128px 的 N=250 与 224px 的 N=250/N=1000；缺 128px 的 N=1000 →
    # 无法判断"分辨率把 onset 推后"是固定倍数，还是随 N 变化（即与 γ 交互）。
    if want("O") and not out_of_time():
        tag = "bloodmnist_n1000_s21_128"
        run(["-m", "train", "--dataset", "bloodmnist", "--n", "1000", "--seed", "21",
             "--epochs", "9600", "--ckpt-every", "800", "--sample-n", "128", "--tag", tag],
            env_extra={"MEDMEM_IMAGE_SIZE": "128", "MEDMEM_BATCH": "32", "MEDMEM_ACCUM": "2",
                       "MEDMEM_SAMPLE_BATCH": "32"},
            tag="O 128px N=1000 到 9600 轮")
        score(tag, "results/pilot128_blood_n1000.csv", "O", dataset="bloodmnist", n=1000, seed=21)
        time.sleep(COOL)

    # ---------------- P: 容量效应的种子稳健性（base=64, N=250, seed 2）
    # C 组（seed 1）显示 base=64 把平台从 0.82 抬到 0.89、把 e640 的 M 放大 78 倍；
    # 单 seed 无法排除"这个 seed 的子集恰好更好记"（同 N 的 M_max 极差可达 33 倍）。
    if want("P") and not out_of_time():
        tag = "b64_pathmnist_n250_s2"
        run(["-m", "train", "--dataset", "pathmnist", "--n", "250", "--seed", "2",
             "--epochs", "5125", "--ckpt-every", "160", "--sample-n", "512", "--tag", tag],
            env_extra={"MEDMEM_BASE_CH": "64", "MEDMEM_BATCH": "32"},
            tag="P base=64 N=250 seed=2")
        score(tag, f"results/{tag}.csv", "P", dataset="pathmnist", n=250, seed=2)
        time.sleep(COOL)

    # ---------------- Q: 64px × N=1000（补齐"分辨率 × N"的 2×3 表）
    # 分辨率阶梯现在是 64/128/224 × (N=250, N=1000)；224 与 128 的 N=1000 在上面两组，
    # 64px 只有 N=250 的 pilot64_blood_n250_s21 → 缺 64px 的 N=1000 这一格，
    # 补上才能说"分辨率把 onset 推后 5.75×"是固定倍数还是与 N 交互。
    if want("Q") and not out_of_time():
        tag = "bloodmnist_n1000_s21_64"
        run(["-m", "train", "--dataset", "bloodmnist", "--n", "1000", "--seed", "21",
             "--epochs", "4800", "--ckpt-every", "100", "--sample-n", "128", "--tag", tag],
            tag="Q 64px N=1000 到 4800 轮")
        score(tag, "results/pilot64_blood_n1000.csv", "Q", dataset="bloodmnist", n=1000, seed=21)
        time.sleep(COOL)

    # ---------------- R/S: 分辨率阶梯的第二个 seed（成对的 64 vs 128）
    # 阶梯现在是"同一个 seed 21 的 64/128/224 配对"，倍数（5.75×）是对的，但只有
    # 一个 seed。N=250 的 M_max 在种子间极差可达 33×，审稿人会问"倍数是不是这个
    # seed 特有"。补 seed 22 的 64px 与 128px → 两对配对，倍数就有了重复。
    if want("R") and not out_of_time():
        tag = "bloodmnist_n250_s22_128"
        run(["-m", "train", "--dataset", "bloodmnist", "--n", "250", "--seed", "22",
             "--epochs", "6400", "--ckpt-every", "800", "--sample-n", "128", "--tag", tag],
            env_extra={"MEDMEM_IMAGE_SIZE": "128", "MEDMEM_BATCH": "32", "MEDMEM_ACCUM": "2",
                       "MEDMEM_SAMPLE_BATCH": "32"},
            tag="R 128px N=250 seed=22 到 6400 轮")
        score(tag, "results/pilot128_blood_n250_s22.csv", "R",
              dataset="bloodmnist", n=250, seed=22)
        time.sleep(COOL)

    if want("S") and not out_of_time():
        tag = "bloodmnist_n250_s22_64"
        run(["-m", "train", "--dataset", "bloodmnist", "--n", "250", "--seed", "22",
             "--epochs", "5125", "--ckpt-every", "160", "--sample-n", "128", "--tag", tag],
            env_extra={"MEDMEM_SAMPLE_BATCH": "32"},
            tag="S 64px N=250 seed=22 到 5125 轮")
        score(tag, "results/pilot64_blood_n250_s22.csv", "S",
              dataset="bloodmnist", n=250, seed=22)
        time.sleep(COOL)

    if only:
        missing = {o for o in only if not any(o == k or o.startswith(k) or k.startswith(o)
                                              for k in seen)}
        if missing:
            log(f"✘ 这些组被请求但一次都没跑：{sorted(missing)}（--only 写法或守卫有 bug）")
            raise SystemExit(f"请求的组未执行：{sorted(missing)}")
    log(f"===== R23 结束；总 {(time.time()-t_start)/3600:.1f} h =====")
    if LOCK.exists():
        LOCK.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())