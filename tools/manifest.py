"""为仓库产物生成 SHA256 清单，并可事后校验（"辅助材料证明"的骨架）。

为什么需要：论文里的每个数字都要能追到某个 CSV 的某一行，而那个 CSV 必须能证明
"自从出图那天起没被改过"。`git log` 能证明提交历史，但读者往往只拿到打包的补充材料
（zip / 网页附件），那时 git 元数据就没了 —— 所以要一份**与 git 无关**的哈希清单。

用法：
    # 生成（默认含受控文件；加 --data 连数据集 npz 一起算）
    .\\.venv\\Scripts\\python.exe -m tools.manifest
    .\\.venv\\Scripts\\python.exe -m tools.manifest --data

    # 校验（拿到别人给的副本后跑一遍；有任何不一致 → 退出码 1）
    .\\.venv\\Scripts\\python.exe -m tools.manifest --verify

产物：
    MANIFEST.sha256         `sha256sum -c` 兼容格式，第一行以 # 记录 commit 与时间（只覆盖仓库内文件）
    results/manifest.csv    逐文件 path,bytes,sha256,mtime
    MANIFEST.data.sha256    `--data` 时额外生成：数据集 npz 的哈希（这些文件不进 git，但来源要可证）
    results/data_manifest.csv

注意：`MANIFEST*.sha256` 自身不进清单（自指会死循环），它们由 git 记录。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"
MANIFEST_CSV = ROOT / "results" / "manifest.csv"
MANIFEST_DATA = ROOT / "MANIFEST.data.sha256"
MANIFEST_DATA_CSV = ROOT / "results" / "data_manifest.csv"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=120)
        return out.stdout.strip()
    except Exception:
        return ""


def tracked_files() -> list[Path]:
    out = _git("ls-files", "-z")
    if not out:
        return sorted(p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts)
    return sorted((ROOT / p) for p in out.split("\x00") if p)


def data_files(include: bool) -> list[Path]:
    if not include:
        return []
    files: list[Path] = []
    for pat in ("data/raw/*.npz", "data/raw/**/*.npz"):
        files += [p for p in ROOT.glob(pat) if p.is_file()]
    return sorted(set(files))


def build(include_data: bool) -> int:
    # 1) 仓库内文件（克隆即可校验）
    files = [p for p in tracked_files() if p.is_file() and p.name != MANIFEST.name]
    rows, lines = _write_manifest(files, MANIFEST, MANIFEST_CSV, "repo")
    total = sum(r["bytes"] for r in rows)
    print(f"写成 {MANIFEST.relative_to(ROOT)}：{len(rows)} 个文件，{total / 1e6:.1f} MB")

    # 2) 数据集（不进 git，但哈希要留，证明"用的就是这一份"）
    if include_data:
        dfs = data_files(True)
        if dfs:
            d_rows, _ = _write_manifest(dfs, MANIFEST_DATA, MANIFEST_DATA_CSV, "data")
            d_total = sum(r["bytes"] for r in d_rows)
            print(f"写成 {MANIFEST_DATA.relative_to(ROOT)}：{len(d_rows)} 个文件，{d_total / 1e6:.1f} MB")
        else:
            print("没有找到 data/raw/*.npz，跳过数据集清单")
    return 0


def _write_manifest(files: list[Path], out: Path, out_csv: Path, tag: str) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    lines = [f"# repo=medmem-law kind={tag} commit={commit} branch={branch} "
             f"generated={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
             f"# files={len(files)}"]
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        d = sha256(p)
        st = p.stat()
        rows.append({"path": rel, "bytes": st.st_size, "sha256": d,
                     "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))})
        lines.append(f"{d}  {rel}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "mtime"])
        w.writeheader()
        w.writerows(rows)
    print(f"写成 {out_csv.relative_to(ROOT)}")
    return rows, lines


def verify() -> int:
    if not MANIFEST.exists():
        print("找不到 MANIFEST.sha256，先跑一次生成")
        return 2
    want: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        d, rel = line.split("  ", 1)
        want[rel] = d

    bad: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    for rel, d in want.items():
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if sha256(p) != d:
            bad.append(rel)

    listed = set(want)
    for p in tracked_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel not in listed and p.is_file() and not p.name.startswith("MANIFEST"):
            extra.append(rel)

    for tag, items in (("改动", bad), ("缺失", missing), ("清单外新增", extra)):
        if items:
            print(f"[{tag}] {len(items)} 个：")
            for it in items[:20]:
                print("   ", it)
            if len(items) > 20:
                print(f"    …还有 {len(items) - 20} 个")

    if bad or missing:
        print(f"校验失败：{len(bad)} 改动 / {len(missing)} 缺失")
        return 1
    print(f"校验通过：{len(want)} 个文件哈希一致"
          + (f"；清单外新增 {len(extra)} 个（未纳入清单，不影响结论）" if extra else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="把 data/raw/*.npz 也纳入清单（数据集来源可证）")
    ap.add_argument("--verify", action="store_true", help="按现有 MANIFEST.sha256 校验")
    a = ap.parse_args()
    return verify() if a.verify else build(a.data)


if __name__ == "__main__":
    sys.exit(main())