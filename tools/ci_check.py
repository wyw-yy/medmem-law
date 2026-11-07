"""CI 守门（不碰 GPU、不碰数据集，只吃仓库里已提交的 CSV/JSON/图）。

它回答一个具体问题：**有人改了代码或结果，数字会不会悄悄变？**

检查项：
  1. MANIFEST.sha256   受控文件哈希一致（跨机器可复现）
  2. tools.audit       测量表自身的 7 条一致性规则
  3. verify_claims     22 条论文声明；PASS 数不得下降，FAIL 只能是那 3 条已知的
  4. 律重拟合          用同一张表重跑 law.main，5 个参数必须与 law_fit.json 一致（1% 内）
  5. R 复算            若环境里有 Rscript，则 r_verify.csv 不得出现 CHECK 行

用法：
    python -m tools.ci_check            # 全跑
    python -m tools.ci_check --skip-r   # 没装 R 时

退出码：0 全过；1 有失败；2 环境缺失（用作 CI 的"跳过"信号，不算失败）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# 这三条 FAIL 是论文里**刻意保留**的未达标项（见 docs/SUPPLEMENT.md §2）。
# 它们若变成 PASS，说明有人偷偷放宽了口径；若出现新的 FAIL，说明数字被改坏。
EXPECTED_FAILS = {"01④", "03④", "04④"}
MIN_PASS = 17


def _run(module: str, *args: str) -> tuple[int, str]:
    p = subprocess.run([PY, "-m", module, *args], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def check_manifest() -> tuple[str, bool, str]:
    rc, out = _run("tools.manifest", "--verify")
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
    return "MANIFEST", rc == 0, tail[0]


def check_audit() -> tuple[str, bool, str]:
    rc, out = _run("tools.audit")
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or [""]
    return "audit", rc == 0, tail[0]


def check_claims() -> tuple[str, bool, str]:
    _run("build.verify_claims")  # 它自己会写 results/verify_claims.csv，rc 不可信
    csv = ROOT / "results" / "verify_claims.csv"
    if not csv.is_file():
        return "verify_claims", False, "没有生成 results/verify_claims.csv"
    d = pd.read_csv(csv)
    n_pass = int((d["status"] == "PASS").sum())
    fails = set(d.loc[d["status"] == "FAIL", "ppt"].astype(str))
    ok = n_pass >= MIN_PASS and fails <= EXPECTED_FAILS
    detail = f"PASS {n_pass}（要求≥{MIN_PASS}）"
    if fails - EXPECTED_FAILS:
        detail += f"；出现未预期的 FAIL：{sorted(fails - EXPECTED_FAILS)}"
    if EXPECTED_FAILS - fails:
        detail += f"；已知未达标项变成了 PASS（口径被放宽？）：{sorted(EXPECTED_FAILS - fails)}"
    return "verify_claims", ok, detail


def check_numbers() -> tuple[str, bool, str]:
    """文档里的数字必须与 results/ 现算值一致（防手抄漂移）。"""
    rc, out = _run("build.doc_numbers", "--check")
    rc2, out2 = _run("build.doc_numbers", "--verify")
    tail = [ln for ln in (out + "\n" + out2).splitlines() if ln.strip()]
    return "numbers", rc == 0 and rc2 == 0, (tail[-1] if tail else "")


def check_law() -> tuple[str, bool, str]:
    import law

    ref_path = ROOT / "results" / "law_fit.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    best = ref["best"]
    ref_p = [float(x) for x in ref["fits"][best]["params"]]

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "law_fit_ci.json"
        law.main(metrics_csv=str(ROOT / "results" / "metrics_long.csv"),
                 out=str(out), datasets=["pathmnist"])
        got = json.loads(out.read_text(encoding="utf-8"))

    if got["best"] != best:
        return "law_refit", False, f"选中的律变了：{best} → {got['best']}"
    got_p = [float(x) for x in got["fits"][best]["params"]]
    worst = max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(got_p, ref_p))
    r2a = float(got["fits"][best]["r2_log"])
    r2b = float(ref["fits"][best]["r2_log"])
    ok = worst <= 0.01 and abs(r2a - r2b) <= 0.01
    return "law_refit", ok, f"{best}：参数最大相对偏差 {worst:.2e}，R²_log {r2a:.4f} vs {r2b:.4f}"


def check_r() -> tuple[str, bool, str]:
    import shutil

    if not shutil.which("Rscript"):
        return "R_verify", True, "跳过（环境无 Rscript）"
    rc, out = _run("tools.r_verify")
    csv = ROOT / "results" / "r_verify.csv"
    if not csv.is_file():
        return "R_verify", False, f"R 未产出 r_verify.csv（rc={rc}）：{out.strip().splitlines()[-1:] or ''}"
    d = pd.read_csv(csv)
    bad = d[d["status"].astype(str).str.upper() == "CHECK"]
    return "R_verify", len(bad) == 0, f"{len(d)} 条检查，CHECK {len(bad)} 条"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-r", action="store_true")
    a = ap.parse_args()

    # 控制台编码兜底：Windows 的 GBK 代码页打不出 ✅/❌ 以外的字符时不至于整轮崩掉
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    checks = [check_manifest, check_audit, check_claims, check_numbers, check_law]
    if not a.skip_r:
        checks.append(check_r)

    rows, ok_all = [], True
    for fn in checks:
        try:
            name, ok, detail = fn()
        except Exception as exc:  # 单个检查崩了不该拖垮整轮报告
            name, ok, detail = fn.__name__, False, f"{type(exc).__name__}: {exc}"
        rows.append((name, ok, detail))
        ok_all &= ok

    width = max(len(r[0]) for r in rows)
    print("\n" + "=" * 74)
    for name, ok, detail in rows:
        # 纯 ASCII 标记：GBK 控制台 / 精简容器里都不会炸
        print(("[PASS] " if ok else "[FAIL] ") + name.ljust(width) + "  " + detail)
    print("=" * 74)
    print("结论：" + ("全部通过" if ok_all else "有检查未通过 —— 数字或口径被动过，先查再提交"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())