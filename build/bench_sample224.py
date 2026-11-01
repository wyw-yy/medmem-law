"""224px 采样 batch 的显存/速度基准（决定 B9 用哪个 batch）。

背景：train.py 的 DDIM 采样按 SAMPLE_BATCH 分批。224px 下 batch=64 会超过 8 GB 显存，
Windows 驱动退回"共享显存"（系统内存），实测比小 batch 慢 70 倍以上。

用法：python -m build.bench_sample224 [ckpt]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MEDMEM_IMAGE_SIZE", "224")
os.environ.setdefault("MEDMEM_BATCH", "16")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
import model as Mdl

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(C.CACHE / "ckpt" / "pneumoniamnist_n1000_s21" / "e0032.pt")


def main() -> int:
    print(f"device={C.DEVICE} size={C.IMAGE_SIZE} ckpt={os.path.basename(CKPT)}")
    net = Mdl.UNet(in_ch=C.CHANNELS, base=C.BASE_CH).to(C.DEVICE)
    state = torch.load(CKPT, map_location=C.DEVICE)
    net.load_state_dict(state["model"])
    net.eval()
    diff = Mdl.Diffusion(device=C.DEVICE)
    shape = (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE)
    n = 16
    for b in (2, 4, 8, 16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out = diff.ddim_sample(net, n, shape, steps=20, device=C.DEVICE, batch=b)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        per_img = dt / n
        print(f"batch={b:2d}  n={n}  steps=20  {dt:6.2f}s  {per_img:5.2f}s/img  peak={peak:.2f}GB", flush=True)
        del out
    return 0


if __name__ == "__main__":
    sys.exit(main())