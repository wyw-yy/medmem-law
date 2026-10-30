"""对已有 checkpoint 重新采样（统一 sample_n），避免不同 N/epoch 的样本数不一致影响 FID。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config as C
import data as D
import model as M

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=C.N_SAMPLE)
ap.add_argument("--epochs", default="", help="逗号分隔，只重采这些 epoch；空=全部")
a = ap.parse_args()

ck_dir = C.CACHE / "ckpt" / a.run
sp_dir = C.CACHE / "samples" / a.run
sp_dir.mkdir(parents=True, exist_ok=True)
want = {int(e) for e in a.epochs.split(",") if e} if a.epochs else None

diff = M.Diffusion(device=C.DEVICE)
net = M.UNet().to(C.DEVICE)
for ck in sorted(ck_dir.glob("e*.pt")):
    ep = int(ck.stem[1:])
    if want is not None and ep not in want:
        continue
    net.load_state_dict(torch.load(ck, map_location=C.DEVICE, weights_only=False)["model"])
    net.eval()
    gen = diff.ddim_sample(net, a.n, (C.CHANNELS, C.IMAGE_SIZE, C.IMAGE_SIZE),
                           steps=C.DDIM_STEPS, device=C.DEVICE)
    np.save(sp_dir / f"e{ep:04d}.npy", D.to_uint8(gen.float().cpu().numpy()))
    print(f"{a.run} e{ep:04d} -> {a.n} 张", flush=True)