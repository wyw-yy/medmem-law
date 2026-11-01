"""
全局常量与路径。所有脚本只从这里读配置，别在别处硬编码数字。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
BUILD = ROOT / "build"
DOCS = ROOT / "docs"

for _p in (DATA, RAW, CACHE, RESULTS, FIGS, BUILD, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# 本机 ~/.cache/torch 建目录被拒（WinError 5），把 torch.hub 的缓存指到项目内
os.environ.setdefault("TORCH_HOME", str(BUILD / "torch_home"))
(BUILD / "torch_home").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据集
# MedMNIST v2，官方 train/val/test 切分，CC BY 4.0，pip/urllib 直接下。
# 主实验用 pathmnist，另两个做跨数据集验证。
DATASETS = {
    "pathmnist": {"desc": "结肠病理 H&E (9 类)", "rgb": True},
    "bloodmnist": {"desc": "血细胞显微 (8 类)", "rgb": True},
    "pneumoniamnist": {"desc": "胸部 X 光 (2 类)", "rgb": False},
}

# 分辨率：本机网络下 zenodo（64px 官方 npz 唯一来源）DNS 解析失败、直连 IP 返回 403，
# 唯一稳定可用的镜像是 HF hf-mirror 上的 albertvillanova/medmnist-v2（只有 28px npz）。
# 所以主网格跑 28px；若 64px 源恢复（见 docs/NOTES.md 的三条备选路线），改回 64 重跑即可，
# 全部下游代码只依赖这个常量。
IMAGE_SIZE = int(os.environ.get("MEDMEM_IMAGE_SIZE", 64))
                       # 64 | 28 | 224（记忆化是分辨率相关的，写论文时必须写清用了哪档）
                       # 64px 由 build/get_hf64.py 从 MedMNIST+ 的 224px 源转出，
                       # 切分与官方一致：train 89996 / val 10004 / test 7180。
                       # 28px 是 pilot（results/pilot28/），只用来先验证方法链路。
                       # 224px 只能按数据集逐个跑（8 GB 显存放不下 batch 64），用
                       # MEDMEM_IMAGE_SIZE=224 + 独立 seed 避免和 64px 的 cache 撞车。
BATCH = int(os.environ.get("MEDMEM_BATCH", 64))   # 大分辨率必须调小（224px 建议 16）
CHANNELS = 3             # 统一升到 3 通道，UNet 不用改

# ---------------------------------------------------------------- 数据规模网格
N_GRID = [250, 500, 1000, 2000, 4000]   # 几何级数，log 轴等距
SEEDS = [1, 2, 3]
N_REF = 2000             # 非成员参照集（校准阈值用）
N_SAMPLE = 512           # 每个 checkpoint 生成多少张
CKPT_EVERY = 25          # 每多少 epoch 存一次 + 采样一次
E_MAX = 800              # 默认最大轮次
KEEP_CKPTS = 4           # 滚动保留几个 checkpoint（权重存 fp16，一个 26 MB；全留会吃满磁盘）

# ---------------------------------------------------------------- 每档 N 的训练预算
# 关键设计：预算按"优化步数"给，不按 epoch 给。
# 理由：记忆化在时间上发生在"步"这个尺度（NeurIPS 2025 Oral 的 τ_mem ∝ N 就是步数单位），
# 所以给每档 N 相同步数，5 个 run 耗时相同（≈25 min），而 epoch 数按 N 反比缩放 ——
# 这样 E/N 正好是"每张图被看了多少遍 / batch"，是理论上该塌缩的横轴。
STEP_BUDGET = 20000      # 每个 run 的训练步数
MEASURE_POINTS = 32      # 每个 run 采多少个测量点（epoch 间隔随之缩放，采样成本也就均匀了）


def epochs_for(n: int, budget: int = STEP_BUDGET, step: int = 25) -> int:
    """给定数据规模 N，算出等预算下的 epoch 数（取整到 step 的倍数，最低一个 step）。"""
    e = budget * BATCH / n
    return max(step, int(round(e / step)) * step)


def ckpt_every_for(epochs: int, points: int = MEASURE_POINTS, step: int = 5) -> int:
    """让每个 run 恰好有 ~points 个测量点。"""
    return max(step, int(round(epochs / points / step)) * step)

# ---------------------------------------------------------------- 模型
BASE_CH = 32             # UNet 基通道。8G 显存下 32 是甜点，48 会 OOM
# BATCH 在 §分辨率 那一段定义（可由 MEDMEM_BATCH 覆盖，供 224px 敏感性实验用）
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
LR = 2e-4
GRAD_CLIP = 1.0
# EMA：采样必须用 EMA 权重。实测非 EMA 权重会把一部分 checkpoint 采成"接近数据集均值"
# 的低对比度图（M 从 0.88 掉到 0.09），M(E) 曲线出假锯齿，律拟合直接崩。
EMA_DECAY = 0.999

# ---------------------------------------------------------------- 采样
DDIM_STEPS = 50          # 采样步数（不是训练步数）
# DDIM 采样的每批张数。主网格 64px 用 64 没问题；224px 用 64 会撑爆 8 GB
# 显存并触发 Windows 共享显存换页（实测比 batch=8 慢 70 倍以上）。
SAMPLE_BATCH = int(os.environ.get("MEDMEM_SAMPLE_BATCH", 64))

# ---------------------------------------------------------------- 记忆率度量
ENCODER = os.environ.get("MEDMEM_ENCODER", "dinov2_s")   # dinov2_s | resnet50
FID_FPR = 0.005          # 阈值校准的假阳性率（0.5%）
LEAK_COS = 0.99          # 参照集里有图与训练集这么像 → 判为重复，踢出校准集
FIXED_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]   # 敏感性分析用的固定阈值
# 64px + DINOv2 下校准 τ ≈ 0.735、sim_max 能到 0.99，所以高阈值段才有信息量；
# 0.5–0.7 保留是为了和 28px pilot 对照。

# ---------------------------------------------------------------- 控制律
EPS_MEM = 0.01           # 可接受的记忆率（1%），安全早停的目标

# ---------------------------------------------------------------- 运行
DEVICE = "cuda"
NUM_WORKERS = 0          # Windows 上 >0 容易在夜间队列里挂住
DPI = 150


def run_tag(dataset: str, n: int, seed: int) -> str:
    """一个 run 的唯一名字。目录名、CSV 里的 run 列都用它。"""
    return f"{dataset}_n{n}_s{seed}"