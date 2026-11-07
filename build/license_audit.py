"""审计本项目的第三方依赖许可（只读，不联网）。

用法：
    python -m build.license_audit            # 打印表格
    python -m build.license_audit --md       # 输出 Markdown 表格（贴进 THIRD_PARTY_NOTICES.md）

做法：扫描 .venv 里每个 dist-info 的 METADATA，取：
  - Name / Version
  - License:（自由文本，常常是空或很长）
  - License-Expression:（PEP 639，新包才有）
  - Classifier: License :: OSI Approved :: ...
再按"是否随仓库分发"分两类：本仓库只分发自己的源码，不 vendor 任何第三方代码，
所以这里的目标是**列出依赖及其许可**，而不是收集它们的 LICENSE 全文。
"""

from __future__ import annotations

import argparse
import re
import sys
from email.parser import HeaderParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
SITE = VENV / "Lib" / "site-packages"

# 我们直接 import 的包（其余是传递依赖）
DIRECT = {
    "torch", "torchvision", "numpy", "pandas", "scipy",
    "matplotlib", "pillow", "tqdm", "medmnist",
}


def _norm_license(text: str) -> str:
    """把 METADATA 里五花八门的 License 字段压成一句短话。"""
    if not text:
        return ""
    t = " ".join(text.split())
    # 常见坑：整个 Apache/BSD 全文塞进 License 字段
    if len(t) > 120:
        head = t[:120]
        for key in ("Apache License", "BSD", "MIT License", "GNU"):
            if key.lower() in t.lower():
                return key
        return head + "…"
    return t


def scan() -> list[dict]:
    rows: list[dict] = []
    if not SITE.is_dir():
        sys.exit(f"找不到 {SITE}，请先确认虚拟环境位置")
    for d in sorted(SITE.glob("*.dist-info")):
        meta = d / "METADATA"
        if not meta.is_file():
            continue
        with meta.open("r", encoding="utf-8", errors="replace") as fh:
            hdr = HeaderParser().parse(fh)
        name = (hdr.get("Name") or d.name.rsplit("-", 1)[0]).strip()
        ver = (hdr.get("Version") or "").strip()
        expr = (hdr.get("License-Expression") or "").strip()
        lic = expr or _norm_license((hdr.get("License") or "").strip())
        classifiers = [c for c in hdr.get_all("Classifier") or [] if c.startswith("License ::")]
        if not lic and classifiers:
            lic = classifiers[0].split("::")[-1].strip()
        rows.append({
            "name": name,
            "version": ver,
            "license": lic or "(未声明)",
            "direct": name.lower() in DIRECT,
            "classifiers": classifiers,
        })
    return sorted(rows, key=lambda r: (not r["direct"], r["name"].lower()))


# 手写补充：METADATA 常写不全的几项（依据官方仓库/发行说明，标注出处）
MANUAL = {
    "torch": ("BSD-3-Clause", "https://github.com/pytorch/pytorch/blob/main/LICENSE"),
    "torchvision": ("BSD-3-Clause", "https://github.com/pytorch/vision/blob/main/LICENSE"),
    "numpy": ("BSD-3-Clause", "https://github.com/numpy/numpy/blob/main/LICENSE.txt"),
    "pandas": ("BSD-3-Clause", "https://github.com/pandas-dev/pandas/blob/main/LICENSE"),
    "scipy": ("BSD-3-Clause", "https://github.com/scipy/scipy/blob/main/LICENSE.txt"),
    "matplotlib": ("PSF-2.0 (Matplotlib licence, BSD-compatible)",
                   "https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE"),
    "pillow": ("MIT-CMU (HPND)", "https://github.com/python-pillow/Pillow/blob/main/LICENSE"),
    "tqdm": ("MPL-2.0 AND MIT", "https://github.com/tqdm/tqdm/blob/master/LICENCE"),
    "medmnist": ("Apache-2.0", "https://github.com/MedMNIST/MedMNIST/blob/main/LICENSE"),
}

# 非 Python 依赖
NON_PYTHON = [
    ("DINOv2 (facebookresearch/dinov2) — 特征编码器代码", "Apache-2.0",
     "https://github.com/facebookresearch/dinov2/blob/main/LICENSE"),
    ("DINOv2 ViT-S/14 预训练权重（dinov2_vits14_pretrain.pth, 88 MB）", "Apache-2.0（随仓库发布）",
     "https://github.com/facebookresearch/dinov2#pre-trained-models"),
    ("Inception-v3 ImageNet 权重（torchvision 分发）", "BSD-3-Clause（权重随 torchvision 发布）",
     "https://github.com/pytorch/vision/blob/main/LICENSE"),
    ("R 语言与 base R（仅本地运行复算脚本，不随仓库分发）", "GPL-2.0-or-later",
     "https://www.r-project.org/Licenses/"),
    ("MedMNIST v2 — 数据集（pathmnist / bloodmnist / pneumoniamnist）", "CC BY 4.0",
     "https://medmnist.com/"),
    ("MedMNIST 224px 原始 JPEG（HF: dz-osamu/MedMNIST，作为 64px 转换的输入）", "CC BY 4.0（继承 MedMNIST v2）",
     "https://huggingface.co/datasets/dz-osamu/MedMNIST"),
    ("PyPI / 清华镜像（wheel 来源）", "—", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
]


def render_md(rows: list[dict]) -> str:
    out = []
    out.append("| 包 | 版本 | 许可 | 直接依赖 | 是否随仓库分发 |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        lic = r["license"]
        man = MANUAL.get(r["name"].lower())
        if man:
            lic = man[0]
        elif re.fullmatch(r"(?i)\s*(mit|bsd-3-clause|apache-2\.0)\s*", lic or "") is None:
            # 保留原始信息但截短
            lic = lic if len(lic) <= 48 else lic[:48] + "…"
        out.append(f"| {r['name']} | {r['version']} | {lic} | {'是' if r['direct'] else '传递'} | 否 |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    a = ap.parse_args()
    rows = scan()
    if a.md:
        print(render_md(rows))
        return 0
    print(f"{'包':<22}{'版本':<14}{'许可':<34}{'类型'}")
    print("-" * 88)
    unknown = []
    for r in rows:
        lic = MANUAL.get(r["name"].lower(), (r["license"], ""))[0]
        if lic == "(未声明)":
            unknown.append(r["name"])
        print(f"{r['name']:<22}{r['version']:<14}{lic[:32]:<34}{'直接' if r['direct'] else '传递'}")
    print("-" * 88)
    print(f"共 {len(rows)} 个发行包；直接依赖 {sum(1 for r in rows if r['direct'])} 个")
    if unknown:
        print("许可未声明（需人工确认）：" + ", ".join(unknown))
    return 0


if __name__ == "__main__":
    sys.exit(main())