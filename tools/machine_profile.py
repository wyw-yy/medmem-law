"""
本机实测：模型参数量、显存、s/step、s/epoch、采样耗时、FLOPS，并给出各规模
run 的预计时长。所有排期数字都应该来自这里，而不是纸面估算（这台机器降频很凶）。

用法：python -m tools.machine_profile [--n 4000] [--epochs 500]
输出：结果打印 + 追加写入 build/machine_profile.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C
import model as M


def _fmt(x):
    return f"{x:.2f}"


def bench_step(net, diff, bs: int, device: str, amp: bool, iters: int = 12) -> float:
    x = torch.randn(bs, C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    for i in range(3):  # warmup
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diff.loss(net, x)
        else:
            loss = diff.loss(net, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        if amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diff.loss(net, x)
        else:
            loss = diff.loss(net, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=C.E_MAX)
    ap.add_argument("--iters", type=int, default=12)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    props = torch.cuda.get_device_properties(0) if dev == "cuda" else None
    net = M.UNet().to(dev)
    diff = M.Diffusion(device=dev)
    bs = min(C.BATCH, a.n)

    res = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name if props else "cpu",
        "sm": props.multi_processor_count if props else 0,
        "vram_gb": round(props.total_memory / 1e9, 2) if props else 0,
        "driver": platform.version(),
        "params_M": round(M.n_params(net) / 1e6, 3),
        "image_size": C.IMAGE_SIZE,
        "batch": bs,
    }

    for amp in (True, False):
        tag = "bf16" if amp else "fp32"
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
        t = bench_step(net, diff, bs, dev, amp, a.iters)
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        res[f"s_per_step_{tag}"] = round(t, 4)
        res[f"peak_gb_{tag}"] = round(peak, 2)
        res[f"steps_per_s_{tag}"] = round(1 / t, 1)
        steps_per_epoch = max(1, a.n // bs)
        res[f"s_per_epoch_{tag}"] = round(t * steps_per_epoch, 2)
        res[f"hours_full_{tag}"] = round(t * steps_per_epoch * a.epochs / 3600, 2)
        # 采样：每 CKPT_EVERY 轮一次，每次 N_SAMPLE 张、DDIM_STEPS 步
        t0 = time.time()
        _ = diff.ddim_sample(net, 32, (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE),
                             steps=C.DDIM_STEPS, device=dev)
        torch.cuda.synchronize() if dev == "cuda" else None
        res[f"sample32_{tag}_s"] = round(time.time() - t0, 2)

    print(json.dumps(res, ensure_ascii=False, indent=2))
    out = C.BUILD / "machine_profile.json"
    hist = []
    if out.exists():
        try:
            hist = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(hist, dict):
                hist = [hist]
        except Exception:  # noqa: BLE001
            hist = []
    hist.append(res)
    out.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()