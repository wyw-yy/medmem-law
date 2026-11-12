"""跑批看门狗：一条命令判定"训练队列现在是否正常"。

设计原则：**只输出判定，不做动作**。判定由谁执行由调用方决定（本项目的用法是
定时提醒里读 VERDICT，异常才修、正常只回一行）。

检查项：
1. 锁与进程：`build/campaign3.lock` 的 pid 是否还活着；没有 python 进程但锁还在 = 僵尸锁。
2. 日志错误：`build/campaign3.log` 里是否有 `✘` / `rc=1` / Traceback。
3. 检查点新鲜度：当前 run 的 `cache/ckpt/<tag>/*.pt` 最新修改时间是否超过"两倍预期间隔"。
   预期间隔 = ckpt_every × s/per_epoch（按 run 类型查表），再加每次测量约 6 min。
   这一步能抓到"进程还在但已卡死"。
4. GPU：利用率 / 显存 / 温度（拿不到就跳过，不算失败）。
5. 队列进度：从日志里解析"▶ 已开始"与"✔ 已完成"，算出还剩哪几组、当前组进度与 ETA。

退出码：0 = OK 或 DONE；1 = STALLED / FAILED（需要人处理）。

用法：
    python -m build.watch_campaign3
    python -m build.watch_campaign3 --json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "build" / "campaign3.log"
LOCK = ROOT / "build" / "campaign3.lock"
CKPT = ROOT / "cache" / "ckpt"

# 每组：(组名, tag, ckpt 间隔, 每 epoch 秒数)。数字都来自实测（results/runs/*.json）。
GROUPS: list[tuple[str, str, int, float]] = [
    ("I1", "dp_s8_c01_n250_s1", 80, 13.0),
    ("I2", "dp_s8_c05_n250_s1", 80, 13.0),
    ("I3", "dp_s8_c20_n250_s1", 80, 13.0),
    ("J1", "b48_pathmnist_n250_s1", 160, 2.6),
    ("J2", "b48_pathmnist_n1000_s1", 80, 2.6),
    ("K", "bloodmnist_n1000_s21_a4", 800, 4.1),
    ("L", "dp_s2_n250_s1_long", 320, 8.5),
    ("M", "dp_s4_n250_s1_long", 320, 8.5),
    ("N", "dp_s1_n250_s1_long", 320, 8.5),
    ("O", "bloodmnist_n1000_s21_128", 800, 1.5),
    ("P", "b64_pathmnist_n250_s2", 160, 2.6),
    ("Q", "bloodmnist_n1000_s21_64", 100, 0.6),
    ("R", "bloodmnist_n250_s22_128", 800, 1.5),
    ("S", "bloodmnist_n250_s22_64", 160, 0.6),
]


def _pid_alive(pid: int) -> bool:
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


KEYWORDS = ("train", "train_dp", "score_run", "run_campaign3", "typology", "finalize")


def _python_pids() -> list[int]:
    """只统计本队列的 python 进程。

    用 tasklist 会把编辑器/别的脚本也算进来 → 锁 pid 已死但队列没跑完时会误判成 OK
    （11-12 合盖休眠后就是这样：GPU 空转、锁是僵尸，看门狗却报 OK）。
    """
    pids: list[int] = []
    try:
        out = subprocess.run(["wmic", "process", "where", "name='python.exe'",
                              "get", "ProcessId,CommandLine", "/format:csv"],
                             capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            low = line.lower()
            if any(k in low for k in KEYWORDS):
                tail = line.rsplit(",", 2)
                if len(tail) >= 2 and tail[1].strip().isdigit():
                    pids.append(int(tail[1].strip()))
    except Exception:
        pass
    if pids:
        return pids
    # wmic 不可用时的退路：只看进程数，至少不会因为"别的 python"把 STALLED 掩盖成 OK
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        if not out:
            return "n/a"
        u, m, t = [x.strip() for x in out[0].split(",")[:3]]
        return f"{u}% / {m} MiB / {t}°C"
    except Exception:  # noqa: BLE001
        return "n/a"


def _log_lines() -> list[str]:
    if not LOG.exists():
        return []
    return LOG.read_text(encoding="utf-8", errors="replace").splitlines()


def _epochs_in(tag: str) -> tuple[int, int, float]:
    """返回 (最新 epoch, ckpt 个数, 最新 ckpt 的分钟龄)。"""
    d = CKPT / tag
    if not d.exists():
        return 0, 0, 1e9
    files = sorted(d.glob("e*.pt"))
    if not files:
        return 0, 0, 1e9
    last = max(files, key=lambda p: p.stat().st_mtime)
    ep = 0
    m = re.match(r"e(\d+)", last.name)
    if m:
        ep = int(m.group(1))
    age = (time.time() - last.stat().st_mtime) / 60
    return ep, len(files), age


def _s_per_epoch_meas(tag: str) -> float | None:
    """从最近两个检查点的 (mtime 差)/(epoch 差) 反推实测 s/epoch。

    比 `GROUPS` 表里的基准值可靠：表里的数字是 3-epoch 冒烟外推，实测偏慢约 30%
    （I1 的 σ=8/C=0.1 配置实测 9.75 s/epoch，表里写 13.0 → ETA 虚高 1.5 倍）。
    """
    d = CKPT / tag
    if not d.exists():
        return None
    rows: list[tuple[int, float]] = []
    for p in d.glob("e*.pt"):
        m = re.match(r"e(\d+)", p.name)
        if m:
            rows.append((int(m.group(1)), p.stat().st_mtime))
    if len(rows) < 2:
        return None
    rows.sort()
    # 用最近 4 个检查点、取相邻间隔的中位数：每个间隔里可能夹着一次采样+评分
    # （512 张 × DDIM 50 ≈ 6 min），单间隔会被这次采样拉高到 20+ s/epoch。
    deltas = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(rows[-4:-1], rows[-3:])
              if b[0] > a[0] and b[1] > a[1]]
    if not deltas:
        return None
    deltas.sort()
    return deltas[len(deltas) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fix", action="store_true",
                    help="卡死时自动杀进程、清锁、重启队列（合盖休眠后 GPU 掉上下文就靠这个）")
    a = ap.parse_args()

    lines = _log_lines()
    # 只看**当前这轮队列**（最后一次 "===== R23 开始" 之后）的日志：
    # 上一轮 06:55 那次"F 评分 rc=1"已经修好了，不能让它永远把判定压成 FAILED。
    start_idx = 0
    for i, l in enumerate(lines):
        if "R23 开始" in l:
            start_idx = i
    lines = lines[start_idx:]
    started = [l for l in lines if "▶" in l]
    done = [l for l in lines if "✔" in l]
    errors = [l for l in lines if ("✘" in l or "rc=1" in l or "Traceback" in l)]

    pid_lock = None
    if LOCK.exists():
        try:
            pid_lock = int(LOCK.read_text().strip() or 0)
        except ValueError:
            pid_lock = 0
    alive = bool(pid_lock) and _pid_alive(pid_lock)
    pys = _python_pids()

    # 当前组：日志里最后一次 ▶ 的 tag
    cur_tag, cur_key = "", ""
    if started:
        m = re.search(r"--tag\s+(\S+)", started[-1])
        cur_tag = m.group(1) if m else ""
        for key, tag, _, _ in GROUPS:
            if tag == cur_tag:
                cur_key = key

    # 队列状态：✔ 行（"✔ I1 DP ..."）或产物 CSV 已存在，都算完成
    done_keys = set()
    for l in done:
        m = re.match(r"\[[^\]]+\]\s+✔\s+([A-Z]\d?)", l)
        if m:
            done_keys.add(m.group(1))
    RESULT_CSV = {
        "I1": "dp_s8_c01_n250_s1.csv", "I2": "dp_s8_c05_n250_s1.csv",
        "I3": "dp_s8_c20_n250_s1.csv", "J1": "b48_pathmnist_n250_s1.csv",
        "J2": "b48_pathmnist_n1000_s1.csv", "K": "pilot224_blood_n1000_a4.csv",
    "O": "pilot128_blood_n1000.csv",
    "P": "b64_pathmnist_n250_s2.csv",
    "Q": "pilot64_blood_n1000.csv",
    "R": "pilot128_blood_n250_s22.csv",
    "S": "pilot64_blood_n250_s22.csv",
        "L": "dp_s2_n250_s1_long.csv", "M": "dp_s4_n250_s1_long.csv",
        "N": "dp_s1_n250_s1_long.csv",
    }
    for key, name in RESULT_CSV.items():
        p = ROOT / "results" / name
        if p.exists() and p.stat().st_size > 0:
            done_keys.add(key)
    # 当前正在跑的那一组单独列出，不算"待跑"
    pending = [k for k, _, _, _ in GROUPS if k not in done_keys and k != cur_key]

    # 当前组进度：用"▶ 行的时间戳 + 实测 s/epoch"估，比检查点 eNNNN 可靠
    # （重启后旧检查点还在目录里，按 eNNNN 会直接读成上一次的进度）
    progress = ""
    stale_note = ""
    if cur_tag:
        rule = next((g for g in GROUPS if g[1] == cur_tag), None)
        ep, n_ck, age = _epochs_in(cur_tag)
        if rule:
            key0, _, ck_int, s_ep = rule
            total = {"I1": 1200, "I2": 1200, "I3": 1200, "J1": 5125, "J2": 2400, "O": 9600, "P": 5125, "Q": 4800,
                     "K": 9600, "L": 5125, "M": 5125, "N": 5125, "R": 6400, "S": 5125}.get(key0, 0)
            started_at = None
            m_t = re.match(r"\[(\d\d)-(\d\d) (\d\d:\d\d:\d\d)\]", started[-1])
            if m_t:
                mm, dd, hh = m_t.group(1), m_t.group(2), m_t.group(3)
                try:
                    t = time.strptime(f"{time.localtime().tm_year}-{mm}-{dd} {hh}", "%Y-%m-%d %H:%M:%S")
                    started_at = time.mktime(t)
                except ValueError:
                    started_at = None
            if total and started_at:
                elapsed_min = max(0.0, (time.time() - started_at) / 60)
                ep_time = int(elapsed_min * 60 / s_ep)
                # 检查点文件名里的 eNNNN 是精确进度；实测 s/epoch 从最近两个检查点的
                # mtime 差反推（表里的 s_ep 是 3-epoch 基准外推，会系统性偏慢 30%）。
                s_ep_meas = _s_per_epoch_meas(cur_tag)
                if s_ep_meas:
                    s_ep = s_ep_meas
                ep_est = min(total, max(ep, ep_time))
                pct = 100.0 * ep_est / total
                eta_min = (total - ep_est) * s_ep / 60  # s_ep 已含测量开销
                progress = (f"约 e{ep_est}/{total} ({pct:.0f}%)，已跑 {elapsed_min/60:.1f} h，"
                            f"剩 ~{eta_min/60:.1f} h（检查点最新 e{ep}，{age:.0f} min 前，"
                            f"实测 {s_ep:.1f}s/epoch）")
            expect_min = ck_int * s_ep / 60 + 6
            if age > 2 * expect_min + 15 and (started_at is None or time.time() - started_at > 2 * expect_min * 60):
                stale_note = (f"检查点 {age:.0f} min 未更新（预期 ≤{2*expect_min:.0f} min）"
                              f" → 可能卡死")

    verdict, detail = "OK", ""
    if errors:
        verdict, detail = "FAILED", f"日志里有错误行：{errors[-1][:120]}"
    elif (pending or cur_key) and not alive:
        verdict, detail = "STALLED", (f"锁里的 pid={pid_lock} 不在了，但队列没跑完"
                                       f"（剩余 {pending or [cur_key]}）→ 需要重启")
    elif (pending or cur_key) and alive and cur_tag and stale_note:
        verdict, detail = "STALLED", stale_note
    elif not pending and not cur_key and not alive:
        verdict, detail = "DONE", "队列已全部完成"

    out = {
        "verdict": verdict,
        "detail": detail,
        "current": {"key": cur_key, "tag": cur_tag, "progress": progress},
        "remaining": pending,
        "running": cur_key,
        "done": sorted(done_keys),
        "gpu": _gpu(),
        "lock_pid": pid_lock,
        "lock_alive": alive,
        "python_pids": pys,
        "log": str(LOG),
    }
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"VERDICT: {verdict}" + (f" — {detail}" if detail else ""))
        if cur_key:
            print(f"当前组: {cur_key} ({cur_tag})  {progress}")
        print(f"队列: 已完成 {sorted(done_keys) or '无'}；待跑 {pending or '无'}")
        print(f"GPU: {out['gpu']}   锁 pid={pid_lock} alive={alive}   python={pys}")
        if verdict in ("STALLED", "FAILED") and pending:
            resume = ([cur_key] if cur_key else []) + pending
            print(f"恢复命令: .\\.venv\\Scripts\\python.exe -m build.run_campaign3 --only {','.join(resume)}")

    # --fix：卡死时自己动手（合盖休眠 / GPU 掉上下文 / 僵尸锁都靠它恢复）。
    # 守卫：必须有"检查点确实太久没动"的证据，否则宁可不杀（长时间采样会误伤）。
    if a.fix and verdict in ("STALLED", "FAILED") and (pending or cur_key):
        if verdict == "STALLED" and detail.startswith("锁里的 pid 不在了"):
            ok = True
        else:
            ep, n_ck, age = _epochs_in(cur_tag) if cur_tag else (0, 0, 1e9)
            ok = age > 90  # 90 分钟没有任何新检查点才算真的死了
        if not ok:
            print("（--fix：证据不足，暂不重启）")
        else:
            print("--fix：判定为真卡死，开始清理")
            for pid in {pid_lock, *pys}:
                if pid and pid != os.getpid():
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                   capture_output=True, text=True)
            time.sleep(5)
            if LOCK.exists():
                LOCK.unlink()
            only = ([cur_key] if cur_key else []) + pending
            auto_log = ROOT / "build" / "campaign3_revive.log"
            subprocess.Popen(
                [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "build.run_campaign3",
                 "--only", ",".join(only)],
                cwd=str(ROOT),
                stdout=open(auto_log, "a", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                creationflags=0x00000008,  # DETACHED_PROCESS
            )
            print(f"--fix：已重启队列（--only {','.join(only)}），日志 build/campaign3_revive.log")
            verdict = "RESTARTED"
    return 0 if verdict in ("OK", "DONE", "RESTARTED") else 1


if __name__ == "__main__":
    sys.exit(main())
