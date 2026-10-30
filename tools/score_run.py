"""
对一个已训练 run 的所有 checkpoint 算指标，追加进 results/metrics_long.csv。

为什么训练和评分分离：
- 评分（编码器 + 白化 + 近邻 + FID）是后处理的定义问题，可能要反复改（换编码器、
  换 FPR、换去重策略）。训练是贵的、不可逆的一半。分开后，改指标不用重训。
- 幂等：已在 CSV 里的 (run, epoch) 直接跳过，可以随时补算新加的 run。

用法
  python -m tools.score_run --all                 # 所有 run 的所有未评分 epoch
  python -m tools.score_run --run pathmnist_n4000_s1 --fid
  python -m tools.score_run --run pathmnist_n4000_s1 --epochs 50,100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C
import data as D
import metric as MT

CSV = C.RESULTS / "metrics_long.csv"
COLS = ["run", "dataset", "n", "seed", "epoch", "M", "tau", "sim_mean", "sim_p50", "sim_p95",
        "sim_max", "n_gen", "ref_max_sim_mean", "n_leak_dropped", "fid", "encoder",
        "M_050", "M_060", "M_070", "M_080", "M_090", "M_095"]


def parse_tag(run: str) -> tuple[str, int, int]:
    # pathmnist_n4000_s1
    parts = run.split("_")
    return parts[0], int(parts[1][1:]), int(parts[2][1:])


def load_csv() -> pd.DataFrame:
    if CSV.exists():
        d = pd.read_csv(CSV)
        for c in COLS:
            if c not in d.columns:
                d[c] = np.nan
        return d[COLS]
    return pd.DataFrame(columns=COLS)


def real_fid_stats(dataset: str, enc_inception, device: str):
    """real 一侧的 inception 统计量缓存（参照集 2000 张，算一次用到底）。"""
    # 缓存键必须带分辨率：28px 的 real 统计量配 64px 生成图会系统性偏高。
    cache = C.CACHE / "feats" / f"fid_real_{dataset}_{C.IMAGE_SIZE}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["mu"], d["cov"]
    ref = D.reference_set(dataset)
    f = MT.inception_feats(ref, enc_inception, device=device)
    mu, cov = MT.fid_stats(f)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=mu, cov=cov)
    return mu, cov


def score_run(run: str, do_fid: bool, only_epochs: list[int] | None, device: str = C.DEVICE,
              refill: bool = False, fid_stride: int = 1) -> int:
    dataset, n, seed = parse_tag(run)
    ck_dir = C.CACHE / "samples" / run
    files = sorted(ck_dir.glob("e*.npy"))
    if not files:
        print(f"[{run}] 没有采样文件，跳过")
        return 0

    df = load_csv()
    mine = df[df["run"] == run] if len(df) else df
    # 「已评」判定要分 M 和 FID 两个层次：
    #   - M 是每个点都算的 → 有 M 就算完成；
    #   - FID 是按 stride 抽样的 → 没有 fid 的点永远不该被当成"待办"，否则每次
    #     队列重启都会把整条曲线重算一遍（浪费几十分钟，永不收敛）。
    done_m = set(mine.loc[mine["M"].notna(), "epoch"].astype(int)) if len(mine) else set()
    done_f = set(mine.loc[mine["fid"].notna(), "epoch"].astype(int)) if len(mine) else set()

    all_eps = sorted(int(p.stem[1:]) for p in files)
    idx_of = {ep: i for i, ep in enumerate(all_eps)}
    def want_fid(ep: int) -> bool:
        return fid_stride <= 1 or (idx_of[ep] % fid_stride == 0)

    todo = []
    for p in files:
        ep = int(p.stem[1:])
        if only_epochs is not None and ep not in only_epochs:
            continue
        need_m = ep not in done_m
        need_f = do_fid and ep not in done_f and want_fid(ep)
        if need_m or need_f:
            todo.append((ep, p))
    if not todo:
        print(f"[{run}] 已全部评过分（M {len(done_m)} 点，FID {len(done_f)} 点）")
        return 0

    train_u8 = D.train_subset(dataset, n, seed)
    ref_u8 = D.reference_set(dataset)
    enc = MT.load_encoder(device=device)
    f_train = MT.encode(train_u8, enc)
    mu, W = MT.fit_whitening(f_train)
    bank = MT.apply_whitening(f_train, mu, W)
    ref_bank = MT.apply_whitening(MT.encode(ref_u8, enc), mu, W)
    cal = MT.calibrate_tau(MT.max_cos_sim(ref_bank, bank))
    enc_tag = f"{C.ENCODER}:{enc[1]}"
    # 编码器来源审计：不同编码器的 τ 量纲不同（实测 dinov2_s 0.6721 / resnet50 0.8203），
    # 混进同一个 CSV 会让 M 从 7.8% 变成 0 —— 所以来源不一致就直接拒绝写。
    if len(mine) and "encoder" in mine.columns:
        prev = set(mine["encoder"].dropna().astype(str).unique())
        if prev and prev != {enc_tag}:
            raise SystemExit(f"[{run}] 编码器不一致：CSV 里是 {sorted(prev)}，本次是 {enc_tag}。"
                             f"要么改回原编码器，要么先删掉该 run 的旧行再重算。")
    print(f"[{run}] τ={cal['tau']:.4f}  FPR={cal['fpr']}  参照={cal['n_ref']}  "
          f"去泄漏={cal['n_leak_dropped']}  enc={enc_tag}  k={W.shape[1]}  n_train={len(train_u8)}")

    inc, fid_real = None, None
    if do_fid:
        inc = MT._inception(device)
        fid_real = real_fid_stats(dataset, inc, device)

    rows = []
    t0 = time.time()
    for ep, path in todo:
        gen = np.load(path)
        f_gen = MT.apply_whitening(MT.encode(gen, enc), mu, W)
        res = MT.memorization_rate(f_gen, bank, cal["tau"], thresholds=C.FIXED_THRESHOLDS)
        row = {"run": run, "dataset": dataset, "n": n, "seed": seed, "epoch": ep,
               "M": res["M"], "tau": cal["tau"], "sim_mean": res["sim_mean"],
               "sim_p50": res["sim_p50"],
               "sim_p95": res["sim_p95"], "sim_max": res["sim_max"], "n_gen": res["n_gen"],
               "ref_max_sim_mean": cal["ref_max_sim_mean"],
               "n_leak_dropped": cal["n_leak_dropped"], "encoder": enc_tag}
        for k, v in res.items():
            if k.startswith("M_"):
                row[k] = v
        if inc is not None and want_fid(ep):
            mu_g, c_g = MT.fid_stats(MT.inception_feats(gen, inc, device=device))
            row["fid"] = MT.fid_from_stats(fid_real[0], fid_real[1], mu_g, c_g)
        rows.append(row)
        print(f"  e{ep:04d}  M={res['M']:.4f}  sim_max={res['sim_max']:.3f}"
              + (f"  fid={row['fid']:.1f}" if "fid" in row else ""), flush=True)
        # 每 10 个点落一次盘：跑到一半被打断不会白干（CSV 是唯一的真相来源）
        if len(rows) % 10 == 0:
            _flush(df, rows, run)
            print(f"  …已落盘 {len(rows)} 行", flush=True)

    _flush(df, rows, run)
    out = load_csv()
    print(f"[{run}] 追加 {len(rows)} 行，用时 {time.time()-t0:.0f}s → {CSV}")
    return len(rows)


def _flush(df: pd.DataFrame, rows: list[dict], run: str) -> None:
    """把已算好的行合并进 CSV（幂等：同 (run, epoch) 以新值为准）。"""
    new = pd.DataFrame(rows)
    base = df if len(df) else pd.DataFrame(columns=COLS)
    keep = ~((base["run"] == run) & base["epoch"].isin(new["epoch"])) if len(base) else slice(None)
    out = pd.concat([base[keep], new], ignore_index=True)
    for c in ("n", "seed", "epoch"):
        out[c] = out[c].astype(int)
    out = out.sort_values(["dataset", "n", "seed", "epoch"]).reset_index(drop=True)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fid", action="store_true", help="同时算 FID（慢：inception 每点 ~15s）")
    ap.add_argument("--fid-stride", type=int, default=1,
                    help="每隔几个测量点算一次 FID（15 s/点，全算 15 个 run 要 2 h）")
    ap.add_argument("--epochs", default=None, help="逗号分隔，只评这些 epoch")
    ap.add_argument("--refill", action="store_true",
                    help="重算已有但缺 fid（或 M）的行；配合 --fid 可给旧行补 FID")
    a = ap.parse_args()

    only = [int(x) for x in a.epochs.split(",")] if a.epochs else None
    if a.all:
        runs = sorted(p.name for p in (C.CACHE / "samples").glob("*") if p.is_dir())
    else:
        runs = a.run or []
    if not runs:
        raise SystemExit("给 --run <tag> 或 --all")
    total = 0
    for r in runs:
        total += score_run(r, a.fid, only, refill=a.refill, fid_stride=a.fid_stride)
    print(f"共追加 {total} 行")


if __name__ == "__main__":
    main()