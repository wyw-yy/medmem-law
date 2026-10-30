"""把 build\\wh 里的 wheel 直接解包进 venv 的 site-packages（完全绕过 pip，不触网、不会卡）。"""
import sys
import zipfile
from pathlib import Path

SP = Path(sys.argv[1] if len(sys.argv) > 1 else ".venv/Lib/site-packages").resolve()
WH = Path("build/wh")

for wheel in sorted(WH.glob("*.whl")):
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        bad = [n for n in names if "/../" in n or n.startswith("../")]
        if bad:
            raise SystemExit(f"{wheel.name} 含可疑路径 {bad[:3]}")
        z.extractall(SP)
    print(f"unpacked {wheel.name} -> {len(names)} files")