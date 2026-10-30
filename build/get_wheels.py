"""
直接下载 wheel（绕过 pip 的网络栈——本机 pip 会卡在索引获取上不返回）。

对每个包：取 PyPI 镜像的 simple 索引页 → 解析出适合 cp312/win_amd64 的 wheel
→ urllib 下载到 build\\wh\\。之后用 `pip install --no-index --no-deps` 装本地文件
（不触网，pip 就不会卡）。

用法：python build\get_wheels.py scipy matplotlib pillow packaging ...
"""
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

IDX = "https://pypi.tuna.tsinghua.edu.cn/simple"
OUT = Path(__file__).parent / "wh"
OUT.mkdir(parents=True, exist_ok=True)

GOOD_TAGS = ("cp312-cp312-win_amd64", "cp312-abi3-win_amd64", "py3-none-any", "py2.py3-none-any")
BAD = ("manylinux", "macosx", "musllinux", "win32", "cp310", "cp311", "cp313", "cp314")
_VER = re.compile(r"^[A-Za-z0-9_.+]+?-(.+?)(?:-\d.*)?$")


def vkey(v: str):
    parts = re.split(r"[.+-]", v)
    out = []
    for p in parts:
        out.append((0, int(p)) if p.isdigit() else (1, p))
    return out


def index_pages(name: str) -> list[str]:
    url = f"{IDX}/{urllib.parse.quote(name)}/"
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: 索引失败 {e}")
        return []
    hrefs = re.findall(r'href="([^"#]+)(?:#[^"]*)?"', html)
    return [urllib.parse.urljoin(url, h) for h in hrefs]


def pick(name: str):
    best = None
    for u in index_pages(name):
        fn = u.rsplit("/", 1)[-1]
        if not fn.endswith(".whl"):
            continue
        if any(b in fn for b in BAD):
            continue
        if not any(t in fn for t in GOOD_TAGS):
            continue
        # 文件名: name-version-tags.whl ；版本段里不能有 rc/dev
        try:
            ver = fn.split("-")[1]
        except IndexError:
            continue
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)*", ver):   # 只要纯数字正式版，跳过 rc/beta
            continue
        k = vkey(ver)
        if best is None or k > best[0]:
            best = (k, fn, u)
    return best


def download(name: str) -> bool:
    got = pick(name)
    if not got:
        print(f"  {name}: 没有匹配的 wheel")
        return False
    _, fn, url = got
    dst = OUT / fn
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"  {name}: 已存在 {fn}")
        return True
    print(f"  {name}: 下载 {fn}")
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            c = r.read(1 << 20)
            if not c:
                break
            f.write(c)
            done += len(c)
            if total:
                print(f"\r    {done/1e6:6.1f}/{total/1e6:6.1f} MB", end="", flush=True)
    print()
    tmp.replace(dst)
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or ["scipy", "matplotlib", "pillow", "packaging", "pyparsing",
                            "contourpy", "cycler", "fonttools", "kiwisolver",
                            "typing_extensions", "medmnist", "pandas", "numpy"]
    ok = 0
    for n in names:
        ok += bool(download(n))
    print(f"\n完成 {ok}/{len(names)}，目录 {OUT}")
    print("装配：pip install --no-index --no-deps build\\wh\\*.whl")