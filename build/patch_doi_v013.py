"""一次性补丁：把 v0.1.3 的 DOI 写进 docs/PUBLISH.md 与 docs/SUPPLEMENT.md。

写这个脚本而不是用 PowerShell 内联 python -c，是因为反引号在 PowerShell 里是转义符，
内联会把 `v0.1.3` 这类 Markdown 代码片段吃掉（本次已踩一次）。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PATCHES = {
    "docs/PUBLISH.md": [
        (
            "| 分支 / 标签 | `master` / `v0.1.0`、`v0.1.1`、`v0.1.2`（附注标签，指向发布时的提交） |",
            "| 分支 / 标签 | `master` / `v0.1.0`、`v0.1.1`、`v0.1.2`、`v0.1.3`（附注标签，指向发布时的提交） |",
        ),
        (
            "| **DOI** | 概念 DOI **`10.5281/zenodo.24012026`**（永远指向最新版）；当前版本 `v0.1.2` → `10.5281/zenodo.24012549`；上一版 `v0.1.1` → `10.5281/zenodo.24012027` |",
            "| **DOI** | 概念 DOI **`10.5281/zenodo.24012026`**（永远指向最新版）；当前版本 `v0.1.3` → `10.5281/zenodo.24028919`；历史：`v0.1.2` → `10.5281/zenodo.24012549`，`v0.1.1` → `10.5281/zenodo.24012027` |",
        ),
    ],
    "docs/SUPPLEMENT.md": [
        (
            "- **已归档**：当前版本 `v0.1.2` → Zenodo 记录 [`10.5281/zenodo.24012549`](https://doi.org/10.5281/zenodo.24012549)；",
            "- **已归档**：当前版本 `v0.1.3` → Zenodo 记录 [`10.5281/zenodo.24028919`](https://doi.org/10.5281/zenodo.24028919)（含 `THIRD_PARTY_NOTICES.md` 与数据集署名）；",
        ),
        (
            "  历史版本：`v0.1.1` → `10.5281/zenodo.24012027`（代码与数字与 `v0.1.2` 一致，仅作者署名为占位）。",
            "  历史版本：`v0.1.2` → `10.5281/zenodo.24012549`（修作者署名）；`v0.1.1` → `10.5281/zenodo.24012027`（首个归档版，作者为占位）。三者代码与数字完全一致。",
        ),
    ],
}


def main() -> int:
    bad = 0
    for rel, subs in PATCHES.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        hit = 0
        for old, new in subs:
            if old in text:
                text = text.replace(old, new)
                hit += 1
            else:
                print(f"MISS {rel}: {old[:60]}")
                bad += 1
        p.write_text(text, encoding="utf-8")
        print(f"{rel}: replaced {hit}/{len(subs)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())