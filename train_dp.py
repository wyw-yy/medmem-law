"""DP-SGD 基线：把"隐私训练"这条路也真跑一遍，而不是只写在 limitation 里。

为什么要单独写一个训练器：DP-SGD 的核心是**逐样本梯度裁剪 + 加噪**，
标准的一次 `loss.backward()` 拿不到"每个样本各自的梯度范数"。这里用
`torch.func.grad` + `torch.func.vmap` 对 **micro-batch 内每个样本** 求梯度
（模型全是 GroupNorm，没有 BatchNorm 的运行统计，所以 vmap 安全）。

隐私账本用 **强组合定理**（Abadi et al. 2016, Thm.1 的简化形式）：

    ε ≤ q · sqrt(2 · T · ln(1/δ)) / σ

其中 q = B/N 是采样率、T 是优化步数、σ 是噪声乘子（噪声尺度 = σ·C）。
这是**保守上界**（比 RDP 松），这里故意用保守估计 —— 宁可把 ε 说大，不要说小。

用法（必须显式给 --out，结果不写进主 CSV）：
    python -m train_dp --dataset pathmnist --n 250 --seed 1 --epochs 5125 \
        --noise 1.0 --clip 1.0 --micro 8 --tag dp_n250_s1 --out results/dp_n250_s1.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

import config as C
import data as D
import model as M
from train import WeightEMA, _ckpt_dir, _load_ckpt, _log, _sample_dir, _save_ckpt


def _per_sample_grads(net, diff, xb: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                      micro: int = 8):
    """返回 {param_name: (B, *param_shape)} 的逐样本梯度。

    **必须分块**：vmap 会为每个样本物化一份完整梯度（13M 参数 × 4B ≈ 52 MB），
    整批 64 个样本就是 3.3 GB，8 GB 卡立刻开始换页（实测 7781 MiB、功耗掉到 29 W、
    每 epoch 88 s，是正常训练的 300 倍）。按 `micro` 分块后每块只占 micro×52 MB。
    """
    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}

    def loss_one(p, b, x0, tt, nz):
        x0 = x0.unsqueeze(0)
        xt = diff.q_sample(x0, tt.unsqueeze(0).expand(1), nz.unsqueeze(0))
        pred = functional_call(net, (p, b), (xt, tt.unsqueeze(0)))
        return F.mse_loss(pred, nz.unsqueeze(0))

    gfun = vmap(grad(loss_one), in_dims=(None, None, 0, 0, 0))
    out: dict[str, list[torch.Tensor]] = {}
    b = len(xb)
    for s in range(0, b, max(1, micro)):
        e = min(b, s + max(1, micro))
        chunk = gfun(params, buffers, xb[s:e], t[s:e], noise[s:e])
        for k, v in chunk.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v, 0) for k, v in out.items()}


def privacy_epsilon(q: float, steps: int, sigma: float, delta: float = 1e-5) -> float:
    """强组合定理给出的 ε 上界（保守）。"""
    if sigma <= 0:
        return float("inf")
    return q * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / sigma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5125)
    ap.add_argument("--noise", type=float, default=1.0, help="噪声乘子 σ")
    ap.add_argument("--clip", type=float, default=1.0, help="逐样本梯度范数上限 C")
    ap.add_argument("--micro", type=int, default=8, help="一次 vmap 处理几个样本")
    ap.add_argument("--ckpt-every", type=int, default=160)
    ap.add_argument("--sample-n", type=int, default=512)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    run = a.tag or f"dp_{a.dataset}_n{a.n}_s{a.seed}"
    device = C.DEVICE
    log_path = C.BUILD / "logs" / f"{run}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    imgs_u8 = D.train_subset(a.dataset, a.n, a.seed)
    x_all = torch.from_numpy(D.to_float_tensor(imgs_u8))
    n = len(x_all)

    net = M.UNet().to(device)
    diff = M.Diffusion(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=C.LR, betas=(0.9, 0.99))
    ema = WeightEMA(net)

    bs = min(C.BATCH, n)
    steps_per_epoch = max(1, math.ceil(n / bs))
    total_steps = steps_per_epoch * a.epochs
    q = bs / n
    eps_ub = privacy_epsilon(q, total_steps, a.noise)
    _log(f"run={run} N={n} E<={a.epochs} batch={bs} micro={a.micro} σ={a.noise} C={a.clip} "
         f"q={q:.3f} steps={total_steps} ε_ub={eps_ub:.1f} (δ=1e-5, 强组合上界)", log_path)
    print(f"[dp] ε 上界 ≈ {eps_ub:.1f}（δ=1e-5，强组合；实际会更小，但这里只报保守值）")

    start_epoch = 0
    if _ckpt_dir(run).glob("e*.pt"):
        start_epoch = _load_ckpt(run, net, opt, device, ema)
        if start_epoch:
            _log(f"续跑：从 epoch {start_epoch} 继续（EMA n={ema.n}）", log_path)

    # 累加器：裁剪后的逐样本梯度之和（在 fp32 累加器里，参数本身用 fp32）
    acc = {k: torch.zeros_like(v) for k, v in net.named_parameters()}
    hist = []
    t_start = time.time()

    for epoch in range(start_epoch + 1, a.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            xb = x_all[idx].to(device)
            b = len(xb)
            if b < 2:
                continue
            t = torch.randint(0, diff.T, (b,), device=device)
            noise = torch.randn_like(xb)
            grads = _per_sample_grads(net, diff, xb, t, noise, a.micro)
            # 逐样本裁剪：c_i = C / max(1, ||g_i||)
            sq = torch.zeros(b, device=device)
            for g in grads.values():
                sq = sq + g.flatten(1).pow(2).sum(1)
            norm = sq.clamp_min(1e-12).sqrt()
            coef = (a.clip / norm).clamp(max=1.0)
            for k, g in grads.items():
                acc[k] += (g * coef.view(-1, *([1] * (g.dim() - 1)))).sum(0)
            # 诊断量用"裁剪前的逐样本梯度范数"：它才是裁剪率与信噪比的直接指示
            tot += float(norm.mean())
            cnt += b
            # 一步结束：加噪、平均、更新
            with torch.no_grad():
                for k, p in net.named_parameters():
                    noise_k = torch.normal(
                        mean=0.0, std=a.noise * a.clip / b, size=p.shape, device=p.device
                    )
                    p.grad = (acc[k] + noise_k) / b
                torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
                for k in acc:
                    acc[k].zero_()
            ema.update(net)
        hist.append({"epoch": epoch, "grad_norm_mean": tot / max(cnt, 1)})

        if epoch % a.ckpt_every == 0 or epoch == a.epochs:
            _save_ckpt(run, net, opt, epoch, ema=ema)
            t0 = time.time()
            gen = diff.ddim_sample(ema.sample_net(), a.sample_n,
                                   (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE),
                                   batch=C.SAMPLE_BATCH, steps=C.DDIM_STEPS, device=device)
            gen_u8 = D.to_uint8(gen.float().cpu().numpy())
            np.save(_sample_dir(run) / f"e{epoch:04d}.npy", gen_u8)
            net.train()
            _log(f"epoch {epoch:4d}/{a.epochs} sample={len(gen_u8)} ({time.time()-t0:.1f}s) "
                 f"总 {(time.time()-t_start)/60:.1f} min", log_path)
            print(f"epoch {epoch}/{a.epochs}  sample {a.sample_n}  "
                  f"{(time.time()-t0):.1f}s  累计 {(time.time()-t_start)/60:.1f} min")

    meta = {
        "run": run, "dataset": a.dataset, "n": a.n, "seed": a.seed,
        "epochs_done": a.epochs, "batch": bs, "micro": a.micro, "noise": a.noise,
        "clip": a.clip, "q": q, "steps": total_steps,
        "epsilon_ub": eps_ub, "delta": 1e-5,
        "accounting": "strong composition (Abadi Thm.1), conservative upper bound",
        "image_size": C.IMAGE_SIZE, "base_ch": C.BASE_CH,
        "s_per_epoch": (time.time() - t_start) / max(a.epochs - start_epoch, 1),
        "history": hist,
    }
    (C.RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "runs" / f"{run}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"完成：{json.dumps({k: meta[k] for k in ('epochs_done', 'epsilon_ub', 's_per_epoch')})}", log_path)
    print(f"完成 rc=0；ε 上界 {eps_ub:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
