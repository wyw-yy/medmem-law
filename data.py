"""
MedMNIST v2 数据加载 + 嵌套子集。

设计要点
- 只用官方 train/val/test 切分。test 集是「非成员参照集」，用于校准阈值，绝不参与训练。
- 同一 seed 下，N=250 的子集是 N=500 的子集：用固定 permutation 取前缀，
  这样不同 N 的曲线是"同一条数据流的截断"，比较才干净（避免子集噪声混淆 N 效应）。
- 图像统一成 float32 [-1,1]、3 通道、IMAGE_SIZE×IMAGE_SIZE。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

import config as C

# 数据来源（本机网络实测）：
#   zenodo.org  : DNS 解析失败，直连 IP 返回 403 → 64px 官方 npz 拿不到
#   hf-mirror   : 可用，2 MB/s；albertvillanova/medmnist-v2 提供 28px 的 {name}.npz，
#                 键名/切分与官方 npz 完全一致，所以下游代码不用改
_HF_URL = "https://hf-mirror.com/datasets/albertvillanova/medmnist-v2/resolve/main/data/{name}.npz"
_ZENODO_URL = "https://zenodo.org/records/10519652/files/{name}_{size}.npz?download=1"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}


def _npz_path(name: str, size: int = C.IMAGE_SIZE) -> Path:
    return C.RAW / f"{name}_{size}.npz"


def _download(url: str, path: Path, tries: int = 80) -> Path:
    """断点续传 + 重试。只认 Content-Length 完全对齐，避免半截 npz 被当成功。"""
    part = path.with_suffix(path.suffix + ".part")
    for k in range(tries):
        have = part.stat().st_size if part.exists() else 0
        total = None
        headers = dict(_UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                crange = r.headers.get("Content-Range")
                if crange and "/" in crange:
                    total = int(crange.split("/")[1])
                elif r.headers.get("Content-Length"):
                    total = have + int(r.headers["Content-Length"])
                else:
                    total = None
                if have and r.status == 200:      # 服务器忽略 Range
                    have = 0
                with open(part, "ab") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if have % (32 << 20) < (1 << 20):
                            msg = f"{have/1e6:.0f}/{total/1e6:.0f} MB" if total else f"{have/1e6:.0f} MB"
                            print(f"[data] {msg}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[data] 第 {k+1} 次尝试失败：{type(e).__name__}: {e}（已有 {have/1e6:.1f} MB）", flush=True)
        if total is not None and part.exists() and part.stat().st_size >= total:
            part.replace(path)
            print(f"[data] 完成 {path.name} {(path.stat().st_size/1e6):.1f} MB", flush=True)
            return path
    raise RuntimeError(f"下载失败：{url}")


def ensure_dataset(name: str, size: int = C.IMAGE_SIZE) -> Path:
    """确保 data/raw/{name}_{size}.npz 存在，返回路径。

    28px 走 HF 镜像（albertvillanova 的官方 npz）；64px 由 build/get_hf64.py
    从 MedMNIST+ 的 224px 源转出 —— 官方 64px 在 Zenodo，本机不可达，
    所以这里**不要**静默回落到 Zenodo（会卡几分钟再报一个看不懂的错）。
    """
    path = _npz_path(name, size)
    if path.exists() and path.stat().st_size > 500_000:
        return path
    if size == 28:
        url = _HF_URL.format(name=name)
        print(f"[data] 下载 {url}", flush=True)
        return _download(url, path)
    raise FileNotFoundError(
        f"缺少 {path.name}。{size}px 版本需要用 224px 源本地转换：\n"
        f"    python -m build.get_hf64 --name {name} --size {size}\n"
        f"（官方 npz 在 Zenodo，本机网络不可达；不要试图回落到 Zenodo）")


def load_split(name: str, split: str, size: int = C.IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """返回 (images uint8 [N,H,W,3], labels [N])。灰度数据集自动复制成 3 通道。"""
    path = ensure_dataset(name, size)
    d = np.load(path)
    imgs = d[f"{split}_images"]
    labels = d[f"{split}_labels"]
    if imgs.ndim == 3:                      # (N,H,W) 灰度
        imgs = imgs[..., None]
    if imgs.shape[-1] == 1:                 # 灰度 → 3 通道
        imgs = np.repeat(imgs, 3, axis=-1)
    elif imgs.shape[-1] > 3:
        imgs = imgs[..., :3]                # 理论上不会发生
    if imgs.dtype != np.uint8:
        imgs = imgs.astype(np.uint8)
    return imgs, np.asarray(labels).reshape(-1).astype(np.int64)


def train_subset_indices(name: str, n: int, seed: int, size: int = C.IMAGE_SIZE) -> np.ndarray:
    """嵌套子集：同一 seed 的 permutation 前缀。"""
    imgs, _ = load_split(name, "train", size)
    if n > len(imgs):
        raise ValueError(f"{name} 训练集只有 {len(imgs)} 张，要不了 N={n}")
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(imgs))
    return np.sort(perm[:n])                 # 排序：DataLoader 里读起来局部性更好


def train_subset(name: str, n: int, seed: int, size: int = C.IMAGE_SIZE) -> np.ndarray:
    imgs, _ = load_split(name, "train", size)
    return imgs[train_subset_indices(name, n, seed, size)]


def reference_set(name: str, n_ref: int = C.N_REF, size: int = C.IMAGE_SIZE) -> np.ndarray:
    """非成员参照集：官方 test 集前 n_ref 张。用于校准 τ 和算 FID 的 real 统计。"""
    imgs, _ = load_split(name, "test", size)
    if n_ref and n_ref < len(imgs):
        imgs = imgs[:n_ref]
    return imgs


def to_float_tensor(imgs_u8: np.ndarray) -> np.ndarray:
    """uint8 [N,H,W,3] → float32 [N,3,H,W] in [-1,1]（NCHW，给 torch 用）。"""
    x = imgs_u8.astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def to_uint8(imgs_f: np.ndarray) -> np.ndarray:
    """float [-1,1] NCHW → uint8 NHWC。"""
    x = (imgs_f + 1.0) * 127.5
    x = np.clip(x, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1))