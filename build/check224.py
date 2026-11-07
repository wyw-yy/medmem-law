"""核对 224px 转换产物的切分尺寸，并检查是否出现重复图像（用于排查 jsonl/zip 索引问题）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def lines_of(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    raw = Path("data/raw")
    for name in ["bloodmnist", "pathmnist", "pneumoniamnist"]:
        for sp in ["train", "val", "test"]:
            f = raw / "hf64" / f"{name}_{sp}.jsonl"
            ls = lines_of(f)
            if not ls:
                continue
            rels = [json.loads(ln)["images"][0] for ln in ls]
            print(f"{name:14s} {sp:5s} jsonl_lines={len(ls):6d} uniq_rel={len(set(rels)):6d}")

    for name in ["bloodmnist", "pneumoniamnist"]:
        p = raw / f"{name}_224.npz"
        if not p.exists():
            print(f"{name}_224.npz 缺失")
            continue
        z = np.load(p)
        shapes = {k: z[k].shape for k in z.keys() if k.endswith("_images")}
        print(f"{name}_224.npz  {shapes}  文件 {p.stat().st_size / 1e6:.0f} MB")
        tr = z["train_images"]
        # 重复检测：对前 400 张做 md5，看是否与自身重复（doubling 会表现为 i 与 i*n/2 相同）
        n = len(tr)
        idx = [0, 1, 2, n // 2, n // 2 + 1, n - 1]
        hashes = {i: hashlib.md5(tr[i].tobytes()).hexdigest()[:10] for i in idx if i < n}
        print("   train 抽样哈希:", hashes, "  唯一值", len(set(hashes.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())