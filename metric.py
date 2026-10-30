"""
记忆率 M 的度量：特征编码 → 白化 → 近邻相似度 → 阈值校准 → 记忆率。

与 naive 做法的区别（这也是贡献 C2）：
1. **阈值由非成员集校准**，不是拍脑袋 0.5/0.6/0.7。对参照集（test，从未训练）
   计算"每张图到训练集的最大余弦相似度"，取 (1-FPR) 分位数作为 τ。
   这样 τ 的含义是：**一张没被训练的图有 FPR 的概率被误判为"记忆"**。论文里可写死。
2. **去重感知**：参照集里如果本来就有训练集的重复图（跨 split 泄漏），它们会把
   τ 抬高 → 记忆率永远为 0。所以先用 LEAK_COS 把重复图踢出校准集，并记录数量。
3. 同时报告固定阈值下的记忆率，做敏感性分析（审稿人会问 τ 换一个怎么办）。

特征：DINOv2 ViT-S/14（自监督，对医学图像比 ImageNet 监督特征更稳）或 ResNet50 兜底。
白化：训练集特征上估计均值/协方差（含收缩），白化后 L2 归一化，再取余弦——
等价于 Mahalanobis 距离下的近邻，避免"所有图都很像"的特征塌缩现象。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config as C


# ------------------------------------------------------------------ 编码器
class _Wrap(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def _hub_dir() -> Path:
    th = os.environ.get("TORCH_HOME")
    base = Path(th) if th else (Path.home() / ".cache" / "torch")
    return base / "hub"


def _load_dinov2_offline(arch: str):
    """repo 与预训练权重都已在本地缓存时，完全离线加载（不碰网络）。

    为什么要这个：torch.hub.load 即使全部命中缓存，也会先去 GitHub 拉一次 ref，
    本机网络间歇性 RemoteDisconnected（实测），在 6 小时的夜间队列里就是必崩。
    """
    hub = _hub_dir()
    repo = hub / "facebookresearch_dinov2_main"
    w = hub / "checkpoints" / f"{arch}_pretrain.pth"
    if not (repo.is_dir() and w.is_file()):
        return None
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import importlib

    B = importlib.import_module("dinov2.hub.backbones")
    m = getattr(B, arch)(pretrained=False)
    sd = torch.load(w, map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[metric] ⚠️ 离线权重与结构有差异：missing={len(missing)} unexpected={len(unexpected)}")
    return m


def _load_dinov2(arch: str, tries: int = 3):
    m = _load_dinov2_offline(arch)
    if m is not None:
        return m
    last = None
    for i in range(tries):
        try:
            return torch.hub.load("facebookresearch/dinov2", arch, verbose=False)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[metric] DINOv2 第 {i + 1}/{tries} 次加载失败：{type(e).__name__}: {e}")
            time.sleep(5 * (i + 1))
    raise last if last is not None else RuntimeError("DINOv2 加载失败（未知原因）")


def load_encoder(name: str = C.ENCODER, device: str = C.DEVICE, allow_fallback: bool | None = None):
    """返回 (model, dim, resize, normalize_stats)。model 输出 [B, dim] 已 L2 归一化。

    ⚠️ 编码器决定 τ 的量纲，也就决定 M 的数值：dinov2_s 在 N=250 上 τ≈0.6721，
    resnet50 在同样数据上 τ≈0.8203（实测）—— 混用会让记忆率从 7.8% 掉到 0%。
    所以默认**静默回退是被禁止的**：请求 dinov2 却在加载时失败，直接抛错。
    真要回退，显式设 MEDMEM_ALLOW_ENCODER_FALLBACK=1。
    """
    if allow_fallback is None:
        allow_fallback = os.environ.get("MEDMEM_ALLOW_ENCODER_FALLBACK", "0") == "1"
    if name.startswith("dinov2"):
        arch = {"dinov2_s": "dinov2_vits14", "dinov2_b": "dinov2_vitb14"}.get(name, "dinov2_vits14")
        try:
            m = _load_dinov2(arch)
            m = _Wrap(m).to(device).eval()
            dim = m(torch.zeros(1, 3, 126, 126, device=device)).shape[-1]
            return m, dim, 224, _IMAGENET
        except Exception as e:  # noqa: BLE001
            if not allow_fallback:
                raise RuntimeError(
                    f"DINOv2 加载失败：{type(e).__name__}: {e}\n"
                    f"不能静默回退到 ResNet50 —— 两者的 τ 不同（0.67 vs 0.82），"
                    f"会让 M 变成 0。检查 {os.environ.get('TORCH_HOME')}/hub 是否完整，"
                    f"或显式设 MEDMEM_ALLOW_ENCODER_FALLBACK=1 表示确实要用 resnet50。"
                ) from e
            print(f"[metric] DINOv2 不可用（{type(e).__name__}: {e}），回退 ResNet50")

    from torchvision.models import resnet50, ResNet50_Weights

    w = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=w)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    return _Wrap(net), 2048, 224, w.transforms()


_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _prep(imgs_u8: np.ndarray, resize: int, mean, std, device):
    """uint8 NHWC → 归一化后的 NCHW，resize 到编码器输入尺寸。"""
    x = torch.from_numpy(imgs_u8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != resize:
        x = F.interpolate(x, size=(resize, resize), mode="bicubic", align_corners=False, antialias=True)
    m = torch.tensor(mean, device=device)[None, :, None, None]
    s = torch.tensor(std, device=device)[None, :, None, None]
    return (x.to(device) - m) / s


@torch.no_grad()
def encode(imgs_u8: np.ndarray, encoder=None, batch: int = 64, device: str = C.DEVICE) -> np.ndarray:
    """uint8 NHWC → float32 [N,dim]，已 L2 归一化（原始空间，未白化）。"""
    if encoder is None:
        encoder = load_encoder(device=device)
    model, _, resize, stats = encoder
    if isinstance(stats, tuple):
        mean, std = stats
    else:  # torchvision transforms 对象
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    out = []
    for i in range(0, len(imgs_u8), batch):
        x = _prep(imgs_u8[i : i + batch], resize, mean, std, device)
        f = model(x).float()
        out.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------ 白化
def fit_whitening(feats: np.ndarray, shrinkage: float = 1e-3,
                  max_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """训练集特征上估计白化矩阵，返回 (mu, W)，W 形状 [d, k]。

    两个坑（实测踩过）：
    1. 特征维 d=384 常大于样本数 n（N=250 的 run 就是），协方差秩亏，
       未截断的白化会把零空间里的噪声放大 30 倍以上，导致所有余弦都掉到 0.05 附近、
       τ 和记忆率一起失去意义。所以截断到 k ≤ n/2 个主方向。
    2. 直接对最大特征值归一化会让弱方向主导相似度，加 shrinkage 收缩。
    """
    mu = feats.mean(0, keepdims=True)
    x = feats - mu
    cov = (x.T @ x) / max(len(x) - 1, 1)
    d = cov.shape[0]
    cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / d * np.eye(d, dtype=cov.dtype)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    if max_dim is None:
        max_dim = max(32, min(d, len(feats) // 2))
    k = min(max_dim, d)
    w = np.clip(w[-k:], 1e-8, None)
    v = v[:, -k:]
    W = v / np.sqrt(w)                      # [d, k]：投影到主子空间后再按方差归一
    return mu, W.astype(np.float32)


def apply_whitening(feats: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    z = (feats - mu) @ W
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


# ------------------------------------------------------------------ 近邻
def max_cos_sim(query: np.ndarray, bank: np.ndarray, chunk: int = 256) -> np.ndarray:
    """每行 query 与 bank 的最大余弦相似度。两边都已 L2 归一化。"""
    out = np.empty(len(query), dtype=np.float32)
    bt = torch.from_numpy(bank)
    for i in range(0, len(query), chunk):
        q = torch.from_numpy(query[i : i + chunk])
        out[i : i + chunk] = (q @ bt.T).max(dim=1).values.numpy()
    return out


def calibrate_tau(ref_max_sims: np.ndarray, fpr: float = C.FID_FPR,
                  leak_cos: float = C.LEAK_COS) -> dict:
    """τ = 干净参照集上 (1-fpr) 分位。返回 τ 及泄漏诊断信息。"""
    leaks = ref_max_sims >= leak_cos
    clean = ref_max_sims[~leaks]
    if len(clean) < 50:
        clean = ref_max_sims          # 极端情况：不去泄漏，退回全体
        leaks = np.zeros_like(ref_max_sims, dtype=bool)
    tau = float(np.quantile(clean, 1.0 - fpr))
    return {
        "tau": tau,
        "fpr": fpr,
        "n_ref": int(len(ref_max_sims)),
        "n_leak_dropped": int(leaks.sum()),
        "ref_max_sim_mean": float(clean.mean()),
        "ref_max_sim_p99": float(np.quantile(clean, 0.99)),
    }


def memorization_rate(gen_feats: np.ndarray, train_bank: np.ndarray, tau: float,
                      thresholds=None) -> dict:
    """M = 近邻相似度 ≥ τ 的生成样本占比。

    thresholds: 额外报固定阈值下的记忆率（键名 M_050 / M_060 …）。
    主结论用 τ（按 FPR 校准）；固定阈值那几列是**稳健性检验**，回答
    "结论是不是靠调阈值调出来的"。
    """
    sims = max_cos_sim(gen_feats, train_bank)
    out = {
        "M": float((sims >= tau).mean()),
        "sim_mean": float(sims.mean()),
        "sim_p50": float(np.quantile(sims, 0.50)),
        "sim_p95": float(np.quantile(sims, 0.95)),
        "sim_max": float(sims.max()),
        "n_gen": int(len(sims)),
    }
    for t in (thresholds or ()):
        key = "M_%03d" % int(round(float(t) * 100))
        out[key] = float((sims >= t).mean())
    return out


# ------------------------------------------------------------------ FID
def _inception(device: str = C.DEVICE):
    from torchvision.models import inception_v3, Inception_V3_Weights

    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, init_weights=False)
    m.fc = torch.nn.Identity()
    return m.to(device).eval()


@torch.no_grad()
def inception_feats(imgs_u8: np.ndarray, model=None, batch: int = 50,
                    device: str = C.DEVICE) -> np.ndarray:
    """FID 用的 inception-v3 2048 维 pool3 特征。"""
    if model is None:
        model = _inception(device)
    assert len(imgs_u8) >= batch, "FID 至少需要 batch 张图（inception 要求）"
    out = []
    for i in range(0, len(imgs_u8), batch):
        x = _prep(imgs_u8[i : i + batch], 299, *_IMAGENET, device)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def fid_stats(feats: np.ndarray, ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian 统计量。样本数远小于 2048 维时协方差奇异，sqrtm 会告警并给出垃圾值，
    所以加一个相对 ridge（按 trace/d 缩放），让 FID 在 256~512 张样本下也稳定可比。"""
    mu = feats.mean(0)
    cov = np.cov(feats, rowvar=False)
    d = cov.shape[0]
    cov = cov + ridge * np.trace(cov) / d * np.eye(d, dtype=cov.dtype)
    return mu, cov


def fid_from_stats(mu1, c1, mu2, c2) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    # scipy >= 1.18 去掉了 sqrtm 的 disp 参数，用 try 兼容新旧版本
    try:
        covmean = linalg.sqrtm(c1 @ c2, disp=False)[0]
    except TypeError:
        covmean = linalg.sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * covmean))


# ------------------------------------------------------------------ 一个 run 的全套度量
def eval_checkpoint(gen_u8: np.ndarray, train_u8: np.ndarray, ref_u8: np.ndarray,
                    encoder=None, inception=None, fid_real_stats=None) -> dict:
    """一次算完：校准 τ、记忆率、固定阈值敏感性、FID（若给了 real 统计）。"""
    encoder = encoder or load_encoder()
    f_train = encode(train_u8, encoder)
    f_ref = encode(ref_u8, encoder)
    f_gen = encode(gen_u8, encoder)

    mu, W = fit_whitening(f_train)
    b_train = apply_whitening(f_train, mu, W)
    b_ref = apply_whitening(f_ref, mu, W)
    b_gen = apply_whitening(f_gen, mu, W)

    cal = calibrate_tau(max_cos_sim(b_ref, b_train))
    res = memorization_rate(b_gen, b_train, cal["tau"])
    res.update({f"tau_{k}": v for k, v in cal.items()})

    sims = max_cos_sim(b_gen, b_train)
    for th in C.FIXED_THRESHOLDS:
        res[f"M_fixed_{th}"] = float((sims >= th).mean())

    if fid_real_stats is not None and inception is not None:
        mu_r, c_r = fid_real_stats
        mu_g, c_g = fid_stats(inception_feats(gen_u8, inception))
        res["fid"] = fid_from_stats(mu_r, c_r, mu_g, c_g)
    return res


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)