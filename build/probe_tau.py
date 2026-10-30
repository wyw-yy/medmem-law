"""在真实的小 N 设定下验证 τ 与可分离性（raw vs 截断白化）。

关注三件事：
  A. 非成员真图 p99.5（= τ）要明显低于 B 的下四分位，否则度量没分辨力
  B. 生成图（真跑出来的 checkpoint）在 τ 之上有多少 → 记忆率 M
  C. 换个 N，τ 会不会漂（τ 必须由 N 决定，因为 bank 大小变了）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
import data as D
import metric as MT


def augment(imgs, seed=0):
    rng = np.random.default_rng(seed)
    out = np.clip(imgs.astype(np.int16) + rng.integers(-15, 16, size=(len(imgs), 1, 1, 1)),
                  0, 255).astype(np.uint8)
    sh = np.empty_like(out)
    for i, im in enumerate(out):
        dy, dx = rng.integers(-2, 3, size=2)
        sh[i] = np.roll(np.roll(im, dy, axis=0), dx, axis=1)
    return sh


def report(tag, m, tau=None):
    p = {f"p{q}": round(float(np.percentile(m, q)), 4) for q in (50, 90, 99, 99.5, 100)}
    extra = f"  M(τ)={float((m >= tau).mean()):.3f}" if tau is not None else ""
    print(f"  {tag:<18} {p}{extra}")


def main():
    enc = MT.load_encoder()
    test = D.reference_set("pathmnist", 2000, size=C.IMAGE_SIZE)
    f_te_raw = MT.encode(test, enc)

    for n in (250, 1000, 4000):
        train = D.train_subset("pathmnist", n, seed=1, size=C.IMAGE_SIZE)
        f_tr = MT.encode(train, enc)
        fake = MT.encode(augment(train[:256]), enc)
        mu, W = MT.fit_whitening(f_tr)
        print(f"\n=== N={n}  bank={f_tr.shape[0]}  k={W.shape[1]} ===")

        s_te = MT.max_cos_sim(MT.apply_whitening(f_te_raw, mu, W), MT.apply_whitening(f_tr, mu, W))
        s_fa = MT.max_cos_sim(MT.apply_whitening(fake, mu, W), MT.apply_whitening(f_tr, mu, W))
        cal = MT.calibrate_tau(s_te)
        tau = cal["tau"]
        report("A 非成员真图", s_te, tau)
        report("B 近复制品", s_fa, tau)
        print(f"  τ={tau:.4f}  FPR={cal['fpr']}  去泄漏={cal['n_leak_dropped']}  "
              f"B 检出率={float((s_fa >= tau).mean()):.3f}")

    # 生成样本：用刚才 smoke run 的 4 个 checkpoint
    print("\n=== 生成样本（pathmnist_n250_s1，N=250 bank） ===")
    train = D.train_subset("pathmnist", 250, seed=1, size=C.IMAGE_SIZE)
    mu, W = MT.fit_whitening(MT.encode(train, enc))
    sims = {}
    for f in sorted(Path(C.CACHE, "samples", "pathmnist_n250_s1").glob("e*.npy")):
        gen = np.load(f)
        s = MT.max_cos_sim(MT.apply_whitening(MT.encode(gen, enc), mu, W),
                           MT.apply_whitening(MT.encode(train, enc), mu, W))
        sims[f.stem] = s
        report(f.stem, s)
    tau = MT.calibrate_tau(MT.max_cos_sim(
        MT.apply_whitening(f_te_raw, mu, W),
        MT.apply_whitening(MT.encode(train, enc), mu, W)))["tau"]
    print(f"  τ(N=250)={tau:.4f}")


if __name__ == "__main__":
    main()