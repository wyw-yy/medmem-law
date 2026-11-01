"""用 R（base R）独立复算律拟合的关键数字，并和 Python 侧逐项对照。

为什么要有这一步：论文里的 R²、留一 N、γ/α 的置信区间若只有 Python 一个实现，
审稿人无法排除"pipeline 某处算错"。R 用 optim 独立重拟合，两条路对得上，
等于给结论加了一道独立校验。发现过分歧时优先怀疑自己——
第一次跑就抓到了 `tools/law_ci.py` 口径不一致的真 bug（见 docs/NOTES.md §9）。

用法：
    python -m tools.r_verify              # 导出 → 调 R → 打印对照表
    python -m tools.r_verify --no-export  # 只调 R（改过 r_*.csv 后）

R 不在 PATH 上（本机实测：`Rscript` 找不到），按下面顺序找：
    1) 环境变量 MEDMEM_RSCRIPT
    2) C:/Program Files/R/R-*/bin/Rscript.exe（取版本号最大的）
    3) PATH 里的 Rscript
输出：results/r_verify.csv（由 R 写）
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402


def find_rscript() -> Path | None:
    env = os.environ.get("MEDMEM_RSCRIPT")
    if env and Path(env).exists():
        return Path(env)
    cands = glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
    cands += glob.glob(os.path.expanduser(r"~\AppData\Local\Programs\R\R-*\bin\Rscript.exe"))

    def ver(p: str):
        try:
            return tuple(int(x) for x in Path(p).parents[2].name.replace("R-", "").split("."))
        except Exception:  # noqa: BLE001
            return (0,)

    cands = [c for c in cands if Path(c).exists()]
    if cands:
        return Path(max(cands, key=ver))
    from shutil import which
    w = which("Rscript")
    return Path(w) if w else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true", help="跳过导出（只用现有 r_*.csv）")
    a = ap.parse_args()

    if not a.no_export:
        r = subprocess.run([sys.executable, "-m", "build.export_for_R"], cwd=ROOT)
        if r.returncode != 0:
            print("[r_verify] 导出失败", file=sys.stderr)
            return r.returncode

    rs = find_rscript()
    if rs is None:
        print("[r_verify] 找不到 Rscript。装 R 或设 MEDMEM_RSCRIPT 指向 Rscript.exe。",
              file=sys.stderr)
        return 3
    print(f"[r_verify] 用 {rs}")

    # R 启动时要在 TMP/TEMP 下建 R_TempDir；本机从 venv 里 fork 出来时
    # 继承的临时目录不可写（`Fatal error: cannot create 'R_TempDir'`），
    # 所以显式指到项目内目录。
    tmp = C.BUILD / "r_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp))

    r = subprocess.run([str(rs), str(ROOT / "R" / "verify_in_R.R"), str(ROOT)],
                       cwd=ROOT, text=True, encoding="utf-8", errors="replace", env=env)
    out = C.RESULTS / "r_verify.csv"
    if out.exists():
        print(f"[r_verify] 对照表已更新：{out}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())