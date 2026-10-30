"""
预下载 MedMNIST v2 数据（64×64 主用 + 28×28 调试），并打印切分大小。

用法：python -m tools.download_data [--sizes 64,28] [--dataset pathmnist]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C
import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--sizes", default=str(C.IMAGE_SIZE))
    a = ap.parse_args()
    names = a.dataset or list(C.DATASETS)
    sizes = [int(s) for s in a.sizes.split(",")]

    for size in sizes:
        for name in names:
            try:
                p = D.ensure_dataset(name, size)
                tr, _ = D.load_split(name, "train", size)
                va, _ = D.load_split(name, "val", size)
                te, _ = D.load_split(name, "test", size)
                print(f"{name:<15} {size}px  train={len(tr):>6} val={len(va):>5} test={len(te):>5} "
                      f"shape={tr.shape[1:]} file={p.name} ({p.stat().st_size/1e6:.0f} MB)")
            except Exception as e:  # noqa: BLE001
                print(f"{name:<15} {size}px  失败：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()