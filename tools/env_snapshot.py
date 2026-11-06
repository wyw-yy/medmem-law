"""固化运行环境（"辅助材料证明"的另一半：数字是人算的，也是这台机器算的）。

产物：
    results/env_snapshot.json   机器可读
    docs/ENV.md                 人可读（可直接贴进论文附录 / GitHub README）

内容：OS、CPU、内存、GPU+驱动+CUDA、Python 与关键包版本、R 版本、
      以及本仓库依赖的"离线权重"文件哈希（否则别人复现时会在下载那步卡住）。
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results" / "env_snapshot.json"
OUT_MD = ROOT / "docs" / "ENV.md"

PKGS = ["torch", "torchvision", "numpy", "pandas", "scipy", "matplotlib",
        "pillow", "tqdm", "medmnist", "packaging"]


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PKGS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def gpu_info() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    q = _run([exe, "--query-gpu=name,driver_version,memory.total,compute_cap",
              "--format=csv,noheader,nounits"])
    parts = [p.strip() for p in q.splitlines()[0].split(",")] if q else []
    info = {"name": parts[0] if len(parts) > 0 else "",
            "driver": parts[1] if len(parts) > 1 else "",
            "vram_mib": parts[2] if len(parts) > 2 else "",
            "compute_cap": parts[3] if len(parts) > 3 else ""}
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda or ""
        info["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def r_version() -> str:
    exe = shutil.which("Rscript") or r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    if not Path(exe).is_file() and not shutil.which("Rscript"):
        return "not found"
    out = _run([exe, "--version"])
    return out.splitlines()[0].strip() if out else "not found"


def weights() -> dict[str, str]:
    """离线权重（复现时最容易卡住的两个下载）。"""
    import hashlib
    hub = ROOT / "build" / "torch_home" / "hub" / "checkpoints"
    found: dict[str, str] = {}
    if hub.is_dir():
        for p in sorted(hub.glob("*.pth")):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            found[p.name] = f"{p.stat().st_size / 1e6:.1f} MB sha256={h.hexdigest()}"
    return found


def main() -> int:
    import torch
    snap = {
        "generated": _run(["powershell", "-NoProfile", "-Command",
                           "Get-Date -Format o"]) or "",
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": pkg_versions(),
        "torch_cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "gpu": gpu_info(),
        "r": r_version(),
        "offline_weights": weights(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# 运行环境（自动生成，勿手改）", "",
          f"- 生成时间：`{snap['generated']}`",
          f"- 系统：{snap['os']}（{snap['machine']}，{snap['cpu_count']} 逻辑核）",
          f"- Python：`{snap['python']}`（`{snap['executable']}`）",
          f"- GPU：{snap['gpu'].get('name', 'n/a')}；驱动 `{snap['gpu'].get('driver', '')}`；"
          f"显存 {snap['gpu'].get('vram_mib', '')} MiB；compute capability "
          f"{snap['gpu'].get('compute_cap', '')}；torch 内置 CUDA `{snap['gpu'].get('torch_cuda', '')}`"
          f"；cuDNN `{snap['gpu'].get('cudnn', '')}`",
          f"- R：`{snap['r']}`", "",
          "## 关键包版本", "", "| 包 | 版本 |", "|---|---|"]
    for k, v in snap["packages"].items():
        md.append(f"| {k} | `{v}` |")
    if snap["offline_weights"]:
        md += ["", "## 离线权重（不随仓库分发，需按 REPRODUCE.md 预取）", "",
               "| 文件 | 大小 / sha256 |", "|---|---|"]
        for k, v in snap["offline_weights"].items():
            md.append(f"| `{k}` | {v} |")
    md += ["", "> 这台机器没有可用 GPU 时：`TORCH_HOME` 与离线权重仍必须存在，"
               "否则 `metric.load_encoder` 会以 fail-loud 方式报错（这是刻意的，"
               "禁止静默回退到 ResNet50 —— 两者的 τ 不同，会让 M 变成 0）。"]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"写出 {OUT_JSON.relative_to(ROOT)} 与 {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())