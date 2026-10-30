"""
泄漏审计：在动手训练之前，先确认 train/test 之间没有重复图。

为什么必须先做：τ 由"非成员参照集到训练集的最大相似度"分位数决定。如果参照集里
本来就混进了训练集的原图/近似图，那条分布会被推到 1.0 附近，τ 跟着抬高，
之后所有生成样本都不"超标"——记忆率恒为 0，整篇课题的技术判断就崩了。

输出：results/leak_report.json，并在有明显泄漏时以非零退出码提醒。

用法：python -m tools.check_leak            # 全部数据集
      python -m tools.check_leak --dataset pathmnist
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C
import data as D


def _hashes(arr: np.ndarray) -> set[str]:
    return {hashlib.md5(a.tobytes()).hexdigest() for a in arr}


def _downsample(arr: np.ndarray, k: int = 8) -> np.ndarray:
    """8×8 缩略图（灰度），用于近似重复检测；比特征编码快且不依赖 GPU。"""
    from PIL import Image

    out = np.empty((len(arr), k * k), dtype=np.float32)
    for i, im in enumerate(arr):
        g = Image.fromarray(im).convert("L").resize((k, k), Image.BILINEAR)
        out[i] = np.asarray(g, dtype=np.float32).reshape(-1)
    out -= out.mean(1, keepdims=True)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-8)


def audit(dataset: str, thr: float = 0.995) -> dict:
    tr, _ = D.load_split(dataset, "train")
    te, _ = D.load_split(dataset, "test")
    va, _ = D.load_split(dataset, "val")

    h_tr = _hashes(tr)
    exact_test = sum(hashlib.md5(a.tobytes()).hexdigest() in h_tr for a in te)
    exact_val = sum(hashlib.md5(a.tobytes()).hexdigest() in h_tr for a in va)

    # 近似重复：参照集每张到训练集的 8×8 缩略图余弦最大值
    q = _downsample(te[:3000])
    b = _downsample(tr)
    sims = (q @ b.T).max(1)
    near = int((sims >= thr).sum())

    rep = {
        "dataset": dataset,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "exact_dup_test_in_train": int(exact_test),
        "exact_dup_val_in_train": int(exact_val),
        "near_dup_test_in_train_ge_%.3f" % thr: near,
        "thumb_max_sim_mean": float(sims.mean()),
        "thumb_max_sim_p99": float(np.quantile(sims, 0.99)),
        "verdict": "OK" if (exact_test == 0 and near == 0) else "LEAK",
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--thr", type=float, default=0.995)
    a = ap.parse_args()
    names = a.dataset or list(C.DATASETS)

    reps = []
    for name in names:
        r = audit(name, a.thr)
        reps.append(r)
        print(f"{name:<16} train={r['n_train']:>6} test={r['n_test']:>5} "
              f"完全重复(test)= {r['exact_dup_test_in_train']} "
              f"近似重复(test)= {r['near_dup_test_in_train_ge_%.3f' % a.thr]} "
              f"缩略图相似 p99={r['thumb_max_sim_p99']:.3f} → {r['verdict']}")

    out = C.RESULTS / "leak_report.json"
    out.write_text(json.dumps(reps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    if any(r["verdict"] != "OK" for r in reps):
        sys.exit(2)


if __name__ == "__main__":
    main()