"""
训练一个 run：N 张图 × E 轮，每 CKPT_EVERY 轮存 checkpoint + 采一批样本。

一个 run 产出 20 个"测量点"（E=25,50,...,500），这是本课题省算力的核心：
不是每个 (N,E) 组合训练一次，而是训练一次读出整条曲线。

可续跑：中途被杀掉（8 GB 卡 + 笔记本降频是常态），重跑同一条命令会从最后一个
checkpoint 接着训。

闭环早停（贡献 C1）：--stop eps 时，每个 checkpoint 现场算一次记忆率 M，
M ≥ eps 就立刻停 —— 这就是策略 D，与开环律 E*(N,ε) 对照。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import config as C
import data as D
import model as M


def _ckpt_dir(run: str) -> Path:
    p = C.CACHE / "ckpt" / run
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sample_dir(run: str) -> Path:
    p = C.CACHE / "samples" / run
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_ckpt(run: str, net, opt, epoch: int, keep: int = C.KEEP_CKPTS):
    """存 checkpoint：权重存 fp16（26 MB 而不是 52 MB），优化器状态单独一份只保留最新。

    存完就删掉倒数第 keep 个之前的 —— 32 个 double-precision ckpt 一个 run 要 4.8 GB，
    15 个 run 就是 70 GB，本机只剩 34 GB。样本已经落盘，旧 ckpt 没有保留价值。
    """
    d = _ckpt_dir(run)
    sd = {k: v.detach().to("cpu", torch.float16) for k, v in net.state_dict().items()}
    torch.save({"model": sd, "epoch": epoch}, d / f"e{epoch:04d}.pt")
    torch.save({"epoch": epoch, "opt": opt.state_dict()}, d / "opt.pt")
    olds = sorted(d.glob("e*.pt"))[:-keep]
    for p in olds:
        p.unlink(missing_ok=True)


def _load_ckpt(run: str, net, opt, device: str) -> int:
    ckpts = sorted(_ckpt_dir(run).glob("e*.pt"))
    if not ckpts:
        return 0
    ck = torch.load(ckpts[-1], map_location=device, weights_only=False)
    net.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    op = _ckpt_dir(run) / "opt.pt"
    if op.exists():
        st = torch.load(op, map_location=device, weights_only=False)
        if int(st["epoch"]) == int(ck["epoch"]):
            opt.load_state_dict(st["opt"])
    return int(ck["epoch"])


def _latest_epoch(run: str) -> int:
    eps = sorted(int(p.stem[1:]) for p in _ckpt_dir(run).glob("e*.pt"))
    return eps[-1] if eps else 0


def _log(msg: str, path: Path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class WeightEMA:
    """权重的指数滑动平均（EMA）。

    为什么必须要：用原始 SGD 权重采样，同一个 run 的不同 checkpoint 会时不时采出
    "低对比度、接近数据集均值"的图（实测 pathmnist N=250：e2240 的 M=0.88，e2400 的
    M=0.09，而且像素 std 从 33 掉到 27）。于是 M(E) 曲线出现假的锯齿，律拟合 R² 塌掉。
    改用 EMA 权重采样后，同一条 run 的 M 变成单调（实测用最后 4 个 ckpt 等权平均验证：
    M=0.930、sim_mean=0.892，回到"好 checkpoint"的水平）。
    这是采样稳定性问题，不是记忆化真的时有时无 —— 这点值得写进论文的度量章节。
    """

    def __init__(self, net, decay: float = C.EMA_DECAY):
        self.decay = decay
        self.net = copy.deepcopy(net).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.n = 0

    @torch.no_grad()
    def update(self, net):
        # 前若干步用预热系数，避免 EMA 停在随机初始化上
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for pe, pn in zip(self.net.parameters(), net.parameters()):
            pe.mul_(d).add_(pn.detach().float(), alpha=1 - d)
        for be, bn in zip(self.net.buffers(), net.buffers()):
            be.copy_(bn)
        self.n += 1

    def sample_net(self):
        return self.net


def _save_ckpt(run: str, net, opt, epoch: int, keep: int = C.KEEP_CKPTS, ema: "WeightEMA | None" = None):
    """存 checkpoint：权重存 fp16（26 MB 而不是 52 MB），优化器状态单独一份只保留最新。

    存完就删掉倒数第 keep 个之前的 —— 32 个 double-precision ckpt 一个 run 要 4.8 GB，
    15 个 run 就是 70 GB，本机只剩 34 GB。样本已经落盘，旧 ckpt 没有保留价值。
    """
    d = _ckpt_dir(run)
    sd = {k: v.detach().to("cpu", torch.float16) for k, v in net.state_dict().items()}
    blob = {"model": sd, "epoch": epoch}
    if ema is not None:
        blob["ema"] = {k: v.detach().to("cpu", torch.float16) for k, v in ema.net.state_dict().items()}
        blob["ema_n"] = ema.n
    torch.save(blob, d / f"e{epoch:04d}.pt")
    torch.save({"epoch": epoch, "opt": opt.state_dict()}, d / "opt.pt")
    olds = sorted(d.glob("e*.pt"))[:-keep]
    for p in olds:
        p.unlink(missing_ok=True)


def _load_ckpt(run: str, net, opt, device: str, ema: "WeightEMA | None" = None) -> int:
    ckpts = sorted(_ckpt_dir(run).glob("e*.pt"))
    if not ckpts:
        return 0
    ck = torch.load(ckpts[-1], map_location=device, weights_only=False)
    net.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    if ema is not None and "ema" in ck:
        ema.net.load_state_dict({k: v.float() for k, v in ck["ema"].items()})
        ema.n = int(ck.get("ema_n", 0))
    op = _ckpt_dir(run) / "opt.pt"
    if op.exists():
        st = torch.load(op, map_location=device, weights_only=False)
        if int(st["epoch"]) == int(ck["epoch"]):
            opt.load_state_dict(st["opt"])
    return int(ck["epoch"])


def train_run(dataset: str, n: int, seed: int, epochs: int = C.E_MAX,
              stop_eps: float | None = None, amp: bool = True,
              sample_n: int = C.N_SAMPLE, device: str = C.DEVICE,
              ckpt_every: int = C.CKPT_EVERY) -> dict:
    run = C.run_tag(dataset, n, seed)
    log_path = C.BUILD / "logs" / f"{run}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    imgs_u8 = D.train_subset(dataset, n, seed)
    x_all = torch.from_numpy(D.to_float_tensor(imgs_u8))

    net = M.UNet().to(device)
    diff = M.Diffusion(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=C.LR, betas=(0.9, 0.99))
    ema = WeightEMA(net)
    n_par = M.n_params(net)

    # ---------------- 续跑
    start_epoch = 0
    if _ckpt_dir(run).glob("e*.pt"):
        start_epoch = _load_ckpt(run, net, opt, device, ema)
        if start_epoch:
            _log(f"续跑：从 epoch {start_epoch} 继续（EMA n={ema.n}）", log_path)

    _log(f"run={run} N={n} E<={epochs} param={n_par/1e6:.2f}M "
         f"device={device} amp={amp} stop_eps={stop_eps}", log_path)

    # 闭环早停要用的度量组件（懒加载，避免不用时浪费显存）
    scorer = None
    if stop_eps is not None:
        import metric as MT

        enc = MT.load_encoder(device=device)
        ref_u8 = D.reference_set(dataset)
        f_train = MT.encode(imgs_u8, enc)
        mu, W = MT.fit_whitening(f_train)
        bank = MT.apply_whitening(f_train, mu, W)
        ref_bank = MT.apply_whitening(MT.encode(ref_u8, enc), mu, W)
        cal = MT.calibrate_tau(MT.max_cos_sim(ref_bank, bank))
        _log(f"闭环校准 τ={cal['tau']:.4f}（FPR={cal['fpr']}, 参照 {cal['n_ref']} 张, "
             f"去掉泄漏 {cal['n_leak_dropped']} 张）", log_path)
        scorer = (MT, enc, bank, cal["tau"])

    bs = min(C.BATCH, n)
    t_start = time.time()
    hist = []
    stopped_early = None

    for epoch in range(start_epoch + 1, epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            xb = x_all[perm[i : i + bs]].to(device, non_blocking=True)
            if len(xb) < 2:
                continue
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = diff.loss(net, xb)
            else:
                loss = diff.loss(net, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
            opt.step()
            ema.update(net)
            tot += float(loss.detach()) * len(xb)
            cnt += len(xb)
        hist.append({"epoch": epoch, "loss": tot / max(cnt, 1)})

        if epoch % ckpt_every == 0 or epoch == epochs:
            _save_ckpt(run, net, opt, epoch, ema=ema)

            t0 = time.time()
            # 采样用 EMA 权重（见 WeightEMA 的注释：非 EMA 权重会随机塌成均值图）
            gen = diff.ddim_sample(ema.sample_net(), sample_n, (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE),
                                   steps=C.DDIM_STEPS, device=device)
            gen_u8 = D.to_uint8(gen.float().cpu().numpy())
            np.save(_sample_dir(run) / f"e{epoch:04d}.npy", gen_u8)
            net.train()

            msg = (f"epoch {epoch:4d}/{epochs} loss={hist[-1]['loss']:.4f} "
                   f"sample={len(gen_u8)} ({time.time()-t0:.1f}s)")
            if scorer is not None:
                MT, enc, bank, tau = scorer
                res = MT.memorization_rate(MT.apply_whitening(MT.encode(gen_u8, enc), mu, W), bank, tau)
                msg += f" M={res['M']:.4f} sim_max={res['sim_max']:.3f}"
                if res["M"] >= stop_eps:
                    stopped_early = epoch
                    msg += "  >>> 触发早停"
            el = time.time() - t_start
            done = epoch - start_epoch
            msg += f" | {el/done:.1f}s/epoch ETA {(epochs-epoch)*el/done/60:.1f}min"
            _log(msg, log_path)
            if stopped_early:
                break

    total = time.time() - t_start
    out = {
        "run": run, "dataset": dataset, "n": n, "seed": seed,
        "params_M": n_par / 1e6, "epochs_target": epochs,
        "epochs_done": int(hist[-1]["epoch"]) if hist else start_epoch,
        "stopped_early_at": stopped_early,
        "stop_eps": stop_eps,
        "seconds": round(total, 1),
        "s_per_epoch": round(total / max(len(hist), 1), 2),
        "loss_curve": hist,
        "image_size": C.IMAGE_SIZE, "base_ch": C.BASE_CH, "batch": bs,
        "lr": C.LR, "ddim_steps": C.DDIM_STEPS, "timesteps": C.TIMESTEPS,
        "ckpt_every": ckpt_every,
        "ema_decay": C.EMA_DECAY, "sample_from": "ema",
    }
    MT = None
    (C.RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS / "runs" / f"{run}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log(f"完成：{out['epochs_done']} epoch，{total/60:.1f} min，"
         f"{out['s_per_epoch']:.1f}s/epoch → results/runs/{run}.json", log_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=C.E_MAX)
    ap.add_argument("--stop-eps", type=float, default=None,
                    help="闭环早停的 ε；不给则跑满 epochs")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--sample-n", type=int, default=C.N_SAMPLE)
    ap.add_argument("--ckpt-every", type=int, default=C.CKPT_EVERY,
                    help="每多少 epoch 存 checkpoint + 采样；主网格按 config.ckpt_every_for 缩放")
    a = ap.parse_args()
    try:
        train_run(a.dataset, a.n, a.seed, a.epochs, a.stop_eps, not a.no_amp, a.sample_n,
              ckpt_every=a.ckpt_every)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()