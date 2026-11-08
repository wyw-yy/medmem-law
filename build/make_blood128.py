"""从 224px 源转出 128px 的 bloodmnist，供"分辨率阶梯"用 64 → 128 → 224 三点。

为什么不用 pathmnist：89996 张 × 128² × 3 = 4.4 GB，加载要整块进内存；
bloodmnist 只有 11959+1712+3421 = 17092 张，128px 约 840 MB。

为什么必须是同一数据集：64px 与 224px 的结论现在都建立在 bloodmnist N=250 seed 21
这一个子集上（64px M=0.97、224px M=0），中间插一个 128px 才能谈"随分辨率怎么走"。

用法：python -m build.make_blood128
"""
from __future__ import annotations

import numpy as np
from PIL import Image

import config as C

SRC = C.RAW / "bloodmnist_224.npz"
DST = C.RAW / "bloodmnist_128.npz"
SIZE = 128


def down(img: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray(img).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.uint8
    )


def main() -> None:
    if DST.exists() and DST.stat().st_size > 100_000_000:
        print(f"[skip] {DST.name} 已存在 {(DST.stat().st_size/1e6):.0f} MB")
        return
    z = np.load(SRC)
    out = {}
    for split in ("train", "val", "test"):
        imgs = z[f"{split}_images"]
        new = np.empty((len(imgs), SIZE, SIZE, 3), dtype=np.uint8)
        for i, img in enumerate(imgs):
            new[i] = down(img)
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i+1}/{len(imgs)}", flush=True)
        out[f"{split}_images"] = new
        out[f"{split}_labels"] = z[f"{split}_labels"]
        print(f"[{split}] {imgs.shape} -> {new.shape}", flush=True)
    np.savez_compressed(DST, **out)
    print(f"写出 {DST.name} {(DST.stat().st_size/1e6):.0f} MB")


if __name__ == "__main__":
    main()