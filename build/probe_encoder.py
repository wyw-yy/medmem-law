"""探测特征编码器可用性（DINOv2 走 torch.hub + fbaipublicfiles，本机可能被墙）。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as C
import metric as ME

out = {}
rng = np.random.default_rng(0)
imgs = rng.integers(0, 255, size=(8, C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8)

for name in ("dinov2_s", "resnet50"):
    t0 = time.time()
    try:
        model, dim, resize, stats = ME.load_encoder(name)
        t_load = time.time() - t0
        t1 = time.time()
        f = ME.encode(imgs, encoder=(model, dim, resize, stats), batch=8)
        t_enc = time.time() - t1
        out[name] = {"ok": True, "dim": int(dim), "resize": resize,
                     "load_s": round(t_load, 1), "encode8_s": round(t_enc, 2),
                     "feat_norm_mean": float(np.linalg.norm(f, axis=1).mean())}
        print(name, "OK", out[name], flush=True)
    except Exception as e:  # noqa: BLE001
        out[name] = {"ok": False, "err": f"{type(e).__name__}: {e}", "load_s": round(time.time() - t0, 1)}
        print(name, "FAIL", out[name], flush=True)

Path("build/encoder_probe.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")