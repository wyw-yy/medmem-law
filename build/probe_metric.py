"""验证记忆率度量能不能区分「真记忆」与「只是相似」。

三个分布要看：
  A. 非成员真图(test) → 训练集最大余弦：这是 τ 的来源，应该明显低于 B
  B. 训练图的近复制品(轻微增广) → 训练集最大余弦：这是"记忆"应有的样子
  C. 生成图 → 训练集最大余弦
对 raw / 全维白化 / 截断白化 三种特征各算一遍，选能让 A 和 B 分开的那种。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
import data as D
import metric as MT


def pct(x, ps=(50, 90, 99, 99.5, 100)):
    return {f"p{p}": round(float(np.percentile(x, p)), 4) for p in ps}


def augment(imgs, seed=0):
    """训练图的近复制品：±6 亮度、±2px 平移（模拟扩散模型"背下来"的输出）。"""
    rng = np.random.default_rng(seed)
    out = imgs.astype(np.int16).copy()
    out = np.clip(out + rng.integers(-15, 16, size=(len(out), 1, 1, 1)), 0, 255).astype(np.uint8)
    shifted = np.empty_like(out)
    for i, im in enumerate(out):
        dy, dx = rng.integers(-2, 3, size=2)
        shifted[i] = np.roll(np.roll(im, dy, axis=0), dx, axis=1)
    return shifted


def main():
    size = C.IMAGE_SIZE
    train = D.train_subset("pathmnist", 1000, seed=1, size=size)
    test = D.reference_set("pathmnist", 500, size=size)
    fake = augment(train[:256])

    enc = MT.load_encoder()
    f_tr = MT.encode(train, enc)
    f_te = MT.encode(test, enc)
    f_fa = MT.encode(fake, enc)

    print(f"\n[26px={size}] 特征维 {f_tr.shape[1]}")
    print("raw 特征：所有向量 L2 归一化后的余弦")
    s_te = MT.max_cos_sim(f_te, f_tr)
    s_fa = MT.max_cos_sim(f_fa, f_tr)
    print("  A 非成员真图 ", pct(s_te))
    print("  B 近复制品   ", pct(s_fa))

    for shrink in (1e-3, 1e-1):
        mu, W = MT.fit_whitening(f_tr, shrinkage=shrink)
        b_tr = MT.apply_whitening(f_tr, mu, W)
        s_te = MT.max_cos_sim(MT.apply_whitening(f_te, mu, W), b_tr)
        s_fa = MT.max_cos_sim(MT.apply_whitening(f_fa, mu, W), b_tr)
        print(f"全维白化 shrink={shrink}:")
        print("  A 非成员真图 ", pct(s_te))
        print("  B 近复制品   ", pct(s_fa))

    # 截断白化：只保留前 k 个主方向（k 取训练样本数的一半，避免放大噪声方向）
    mu = f_tr.mean(0, keepdims=True)
    x = f_tr - mu
    cov = (x.T @ x) / max(len(x) - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    for k in (64, 128, 256):
        vk = v[:, -k:]
        wk = np.clip(w[-k:], 1e-8, None)

        def tr(f):
            z = ((f - mu) @ vk) / np.sqrt(wk)
            return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)

        s_te = MT.max_cos_sim(tr(f_te), tr(f_tr))
        s_fa = MT.max_cos_sim(tr(f_fa), tr(f_tr))
        print(f"截断白化 k={k}:")
        print("  A 非成员真图 ", pct(s_te))
        print("  B 近复制品   ", pct(s_fa))


if __name__ == "__main__":
    main()