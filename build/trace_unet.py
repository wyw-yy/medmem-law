import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import model as M

net = M.UNet()
x = torch.zeros(2, 3, 64, 64)
t = torch.zeros(2, dtype=torch.long)
print("params M", M.n_params(net) / 1e6)

temb = net.time(t)
print("temb", temb.shape)
h = net.conv_in(x)
print("conv_in", h.shape)
skips = []
for i, d in enumerate(net.downs):
    h, sk = d(h, temb)
    skips.append(sk)
    print(f"down{i} skip={tuple(sk.shape)} out={tuple(h.shape)}")
h = net.mid1(h, temb)
h = net.mid2(h, temb)
print("mid", h.shape)
for i, (u, sk) in enumerate(zip(net.ups, reversed(skips))):
    print(f"up{i} up.in={u.up.in_channels} skip_ch={u.b1.conv1.in_channels - u.up.in_channels} "
          f"x={tuple(h.shape)} skip={tuple(sk.shape)}")
    h = u(h, sk, temb)
    print(f"     out={tuple(h.shape)}")