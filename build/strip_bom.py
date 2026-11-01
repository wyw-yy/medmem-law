"""剥掉误加的 UTF-8 BOM（PowerShell 的 Set-Content/Add-Content 偶尔会加）。"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUFFIX = {".py", ".md", ".csv", ".txt", ".R", ".ps1", ".cmd", ".json"}
fixed = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in SUFFIX:
        continue
    if ".venv" in p.parts or "torch_home" in p.parts or "cache" in p.parts or "data" in p.parts:
        continue
    b = p.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        p.write_bytes(b[3:])
        fixed.append(str(p.relative_to(ROOT)))
print(f"剥掉 BOM 的文件 {len(fixed)} 个")
for f in fixed:
    print("  ", f)
sys.exit(0)