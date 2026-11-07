"""把 HF 上的 MedMNIST 224px 图像包转成目标分辨率的 npz（默认 64）。

背景：官方 MedMNIST+ 的 64/128/224 npz 都在 Zenodo，本机不可达。
HF 的 dz-osamu/MedMNIST 有 224px 的原始 jpg（zip）与 VLM 格式的 jsonl
（images 字段是 zip 内路径，response 形如 "1:pneumonia"）。
本脚本按 jsonl 的顺序把 224px jpg 缩放到 64×64，写成与官方 npz 同名的键：
    {split}_images (NHWC uint8), {split}_labels (N,1) int64

用法：
    python -m build.get_hf64 --name pathmnist            # 下载+转换
    python -m build.get_hf64 --name pathmnist --size 64 --skip-download
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C   # noqa: E402

BASE = "https://hf-mirror.com/datasets/dz-osamu/MedMNIST/resolve/main/"
RAW = C.DATA / "raw"
SPLITS = ("train", "val", "test")


def _req(url: str, headers: dict | None = None):
    h = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h)


def _remote_size(fname: str) -> int | None:
    """远端文件大小（HEAD）。拿不到就返回 None。"""
    req = _req(BASE + fname)
    req.method = "HEAD"
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:  # noqa: BLE001
        return None


def download(fname: str, dest: Path, tries: int = 40) -> Path:
    """带 Range 续传的下载（本机网络会间歇性断流）。

    关键：文件下满后**必须**先比一次远端大小再发 Range 请求，否则
    `Range: bytes=<EOF>-` 会被服务器判 416（Requested Range Not Satisfiable），
    旧版本把 416 当失败重试 40 次，看起来像卡死。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        have = dest.stat().st_size if dest.exists() else 0
        want = _remote_size(fname)
        if want is not None:
            if have == want:
                print(f"  完成 {fname} {have/1e6:.1f} MB（已存在且大小一致）", flush=True)
                return dest
            if have > want:  # 本地比远端还大：上次写坏了，重下
                print(f"  本地 {have} > 远端 {want}，删掉重下", flush=True)
                dest.unlink()
                have = 0
        hdrs = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(_req(BASE + fname, hdrs), timeout=60) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have else "wb"
                t0 = time.time()
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if total and have % (50 << 20) < (1 << 20):
                            print(f"    {fname}: {have/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"({have/total*100:.0f}%) {time.time()-t0:.0f}s", flush=True)
            if total is None or dest.stat().st_size >= total:
                print(f"  完成 {fname} {dest.stat().st_size/1e6:.1f} MB", flush=True)
                return dest
            print(f"  未达预期大小（{dest.stat().st_size}/{total}），重试", flush=True)
        except urllib.error.HTTPError as e:
            # 416 = Range 起点已在文件末尾，等价于"已经下完"
            if e.code == 416 and dest.exists():
                want = _remote_size(fname)
                if want is None or dest.stat().st_size >= want:
                    print(f"  完成 {fname} {dest.stat().st_size/1e6:.1f} MB（416 判定为已下满）", flush=True)
                    return dest
            print(f"  第 {i+1} 次失败：{repr(e)[:90]}", flush=True)
            time.sleep(2 + min(i, 10))
        except Exception as e:  # noqa: BLE001
            print(f"  第 {i+1} 次失败：{repr(e)[:90]}", flush=True)
            time.sleep(2 + min(i, 10))
    raise SystemExit(f"下载失败：{fname}")


def convert(name: str, size: int) -> Path:
    from PIL import Image

    zip_path = RAW / "hf64" / f"{name}_image.zip"
    out = RAW / f"{name}_{size}.npz"
    if out.exists():
        print(f"已存在 {out}，跳过")
        return out

    zf = zipfile.ZipFile(zip_path)
    # 这个包的成员名是一串绝对路径（…/medmnist224/pathmnist_n/{split}/{i}.jpg），
    # 所以按「最后两级目录 = split / 文件名」建索引，别指望 medmnist/xxx 这种干净路径。
    index: dict[tuple[str, int], str] = {}
    for nm in zf.namelist():
        parts = nm.replace("\\", "/").split("/")
        if len(parts) < 2 or not parts[-1].lower().endswith(".jpg"):
            continue
        try:
            index[(parts[-2], int(Path(parts[-1]).stem))] = nm
        except ValueError:
            continue
    print(f"zip 内 {len(index)} 张 jpg，split={sorted({s for s, _ in index})}", flush=True)

    arrays = {}
    for split in SPLITS:
        raw_lines = [ln for ln in (RAW / "hf64" / f"{name}_{split}.jsonl")
                     .read_text(encoding="utf-8").splitlines() if ln.strip()]
        # 这些 jsonl 会因重复下载而**整段重复一遍**（hf-mirror 重定向后丢了 Range 头，
        # 变成从 0 开始追加）。按 images 路径去重，保留首次出现的顺序。
        lines, seen = [], set()
        for ln in raw_lines:
            try:
                key = json.loads(ln)["images"][0]
            except Exception:  # noqa: BLE001
                continue
            if key in seen:
                continue
            seen.add(key)
            lines.append(ln)
        if len(lines) != len(raw_lines):
            print(f"  {split}: jsonl {len(raw_lines)} 行 → 去重后 {len(lines)} 行", flush=True)
        n = len(lines)
        imgs = np.zeros((n, size, size, 3), np.uint8)
        labs = np.zeros((n, 1), np.int64)
        t0 = time.time()
        for i, ln in enumerate(lines):
            j = json.loads(ln)
            rel = j["images"][0].lstrip("/")
            try:
                key = (rel.split("/")[-2], int(Path(rel).stem))
            except Exception:  # noqa: BLE001
                key = (split, i)
            member = index.get(key) or index.get((split, i))
            if member is None:
                raise SystemExit(f"{split}/{i} 在 zip 里找不到（已知 {len(index)} 项）")
            with zf.open(member) as fh:
                im = Image.open(io.BytesIO(fh.read()))
                im = im.convert("RGB")
                if im.size != (size, size):
                    im = im.resize((size, size), Image.LANCZOS)
                imgs[i] = np.asarray(im, np.uint8)
            lab = j["response"].split(":")[0].strip()
            labs[i, 0] = int(lab)
            if (i + 1) % 10000 == 0:
                print(f"  {split} {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
        arrays[f"{split}_images"] = imgs
        arrays[f"{split}_labels"] = labs
        print(f"  {split}: {imgs.shape} labels {np.bincount(labs.ravel()).tolist()[:10]}", flush=True)

    np.savez_compressed(out, **arrays)
    print(f"写出 {out} {out.stat().st_size/1e6:.1f} MB")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pathmnist")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    d = RAW / "hf64"
    d.mkdir(parents=True, exist_ok=True)
    if not a.skip_download:
        download(f"{a.name}_image.zip", d / f"{a.name}_image.zip")
        for split in SPLITS:
            download(f"{a.name}_{split}.jsonl", d / f"{a.name}_{split}.jsonl")
    convert(a.name, a.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())