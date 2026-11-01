"""记忆化类型学：被"复刻"的到底是训练集里的哪一类样本？

动机（答辩时必被问到）：「记忆率 80%」这个数字很抽象，评审会问——复刻的是哪些图？
是罕见病例，还是训练集里本来就重复的那几张？

文献（Carlini 2023 / Somepalli 2023）给的假说：被抓到复刻的样本往往是**离群样本**或
**重复样本**。我们在 MedMNIST 上能直接检验，因为标签和全部训练图都在本地：

  H1 少数派假说：被复刻的训练图在训练集里反复出现（同一个 target 被多个生成样本命中）
  H2 离群假说：被复刻的训练图在训练集内更"孤立"（到第 10 近邻的余弦距离更大）
  H3 类别偏斜：被复刻的图集中在少数几个病理类别（相对训练集分布被放大）

输出：
  results/typology.csv           每个 run 一行（命中数、去重目标数、top1 占比、离群度对比…）
  results/typology_targets.csv   每个被命中的训练图一行（train_idx、标签、命中次数、离群度）
  results/typology_classes.csv   每个类别一行（训练集占比 vs 被命中占比）

用法：
    python -m build.typology                       # 默认三个代表 run
    python -m build.typology --runs pathmnist_n250_s1 --epochs 5125
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
import data as D  # noqa: E402
import metric as MT  # noqa: E402

DEFAULT = [
    ("pathmnist_n250_s1", 5125),
    ("pathmnist_n1000_s1", 2400),
    ("bloodmnist_n250_s1", 5125),
]


def parse_run(run: str) -> tuple[str, int, int]:
    """pathmnist_n250_s1 → (pathmnist, 250, 1)。"""
    parts = run.split("_")
    ds = parts[0]
    n = int(parts[1][1:])
    seed = int(parts[2][1:])
    return ds, n, seed


def tau_of(run: str, epoch: int) -> float:
    d = pd.read_csv(C.RESULTS / "metrics_long.csv")
    m = d[(d["run"] == run) & (d["epoch"] == epoch)]
    if m.empty:
        raise SystemExit(f"{run} e{epoch} 不在 metrics_long.csv 里")
    return float(m["tau"].iloc[0])


def atyp(feats_w: np.ndarray, k: int = 10, chunk: int = 256) -> np.ndarray:
    """每个训练图到"第 k 近邻（不含自己）"的余弦距离；越大越孤立。"""
    n = len(feats_w)
    out = np.full(n, np.nan, dtype=np.float32)
    kk = min(k, n - 1)
    if kk <= 0:
        return out
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = feats_w[s:e] @ feats_w.T
        for i in range(e - s):
            sims[i, s + i] = -2.0          # 排除自己
        part = np.partition(sims, -kk, axis=1)[:, -kk:]
        out[s:e] = 1.0 - part.min(axis=1)  # 第 k 近的相似度 → 距离
    return out


def run_one(run: str, epoch: int, encoder, mu, W):
    ds, n, seed = parse_run(run)
    tau = tau_of(run, epoch)
    gen = np.load(C.CACHE / "samples" / run / f"e{epoch:04d}.npy")
    train_all, labels_all = D.load_split(ds, "train")
    idx = D.train_subset_indices(ds, n, seed)
    train = train_all[idx]
    labels = labels_all[idx]

    f_tr = MT.apply_whitening(MT.encode(train, encoder), mu, W)
    f_gen = MT.apply_whitening(MT.encode(gen, encoder), mu, W)

    sims = f_gen @ f_tr.T                       # [n_gen, N]
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(best)), best]
    hit = best_sim >= tau

    hits = pd.Series(best[hit]).value_counts().sort_values(ascending=False)
    n_hit = int(hit.sum())

    # H2：被命中目标的离群度 vs 全体
    at = atyp(f_tr)
    at_hit = at[hits.index.to_numpy()] if len(hits) else np.array([])

    # H3：类别占比
    lab = pd.Series(labels).value_counts().sort_index()
    lab_hit = pd.Series(labels[hits.index.to_numpy()]).value_counts().sort_index() if len(hits) \
        else pd.Series(dtype=int)
    rows_cls = []
    for c in lab.index:
        ntr = int(lab[c])
        nh = int(lab_hit.get(c, 0))
        rows_cls.append(dict(run=run, epoch=epoch, label=int(c), n_train=ntr, n_hits=nh,
                             train_share=ntr / len(labels),
                             hit_share=nh / max(n_hit, 1),
                             overrep=(nh / max(n_hit, 1)) / (ntr / len(labels))))

    summary = dict(
        run=run, dataset=ds, n=n, seed=seed, epoch=epoch, tau=tau,
        n_gen=len(gen), n_hit=n_hit, M=n_hit / len(gen),
        distinct_targets=int(len(hits)),
        repeat_ratio=(n_hit / len(hits)) if len(hits) else np.nan,
        top1_share=(hits.iloc[0] / n_hit) if len(hits) else np.nan,
        top3_share=(hits.iloc[:3].sum() / n_hit) if len(hits) else np.nan,
        hit_classes=int(lab_hit.size),
        max_overrep=float(np.nanmax([r["overrep"] for r in rows_cls])) if rows_cls else np.nan,
        atyp_mem_med=float(np.nanmedian(at_hit)) if len(at_hit) else np.nan,
        atyp_all_med=float(np.nanmedian(at)),
        sim_med=float(np.median(best_sim[hit])) if n_hit else np.nan,
        sim_max=float(best_sim.max()),
    )
    targets = pd.DataFrame({
        "run": run, "epoch": epoch, "train_idx": hits.index.to_numpy() if len(hits) else [],
        "label": labels[hits.index.to_numpy()] if len(hits) else [],
        "hits": hits.to_numpy() if len(hits) else [],
        "sim_max": [float(sims[best == t, t].max()) for t in (hits.index if len(hits) else [])],
        "atyp": at[hits.index.to_numpy()] if len(hits) else [],
    })
    return summary, targets, pd.DataFrame(rows_cls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[r for r, _ in DEFAULT])
    ap.add_argument("--epochs", nargs="*", type=int, default=[e for _, e in DEFAULT])
    a = ap.parse_args()
    if len(a.epochs) != len(a.runs):
        raise SystemExit("--runs 与 --epochs 数量要一致")

    encoder = MT.load_encoder()
    sums, tgts, clss = [], [], []
    for run, epoch in zip(a.runs, a.epochs):
        ds, n, seed = parse_run(run)
        # 白化矩阵必须与该 (n, seed) 的训练集一致（和评分时同一条路径）
        _tr, _ = D.load_split(ds, "train")
        idx = D.train_subset_indices(ds, n, seed)
        f_tr = MT.encode(_tr[idx], encoder)
        mu, W = MT.fit_whitening(f_tr)
        print(f"[typology] {run} e{epoch} …", flush=True)
        s, t, c = run_one(run, epoch, encoder, mu, W)
        sums.append(s)
        tgts.append(t)
        clss.append(c)
        print(f"  命中 {s['n_hit']}/{s['n_gen']}（M={s['M']:.3f}）；"
              f"去重目标 {s['distinct_targets']} 个，top1 占 {s['top1_share']:.2f}；"
              f"离群度 命中 {s['atyp_mem_med']:.3f} vs 全体 {s['atyp_all_med']:.3f}", flush=True)

    pd.DataFrame(sums).to_csv(C.RESULTS / "typology.csv", index=False, encoding="utf-8")
    pd.concat(tgts, ignore_index=True).to_csv(C.RESULTS / "typology_targets.csv",
                                             index=False, encoding="utf-8")
    pd.concat(clss, ignore_index=True).to_csv(C.RESULTS / "typology_classes.csv",
                                             index=False, encoding="utf-8")
    print(f"\n写出 results/typology{{,_targets,_classes}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())