"""
夜间队列：按顺序把一批 run 训练 + 评分跑完，可随时中断、重启后续跑。

为什么需要它：主网格是 5 档 N × 3 seed = 15 个 run。笔记本不可能一次跑完，中间一定被
关盖/断电/显存警告打断。队列把"哪些 run 已完成"记在 build/queue_state.json，
启动时自动跳过已完成项（train.run 自身也能从 checkpoint 续跑）。

预算口径：每档 N 给相同的**步数**预算（config.STEP_BUDGET），所以每个 run 耗时接近，
epoch 数按 N 反比缩放（config.epochs_for）。因此本队列里不要传 --epochs 覆盖，
除非做降级实验。

用法
  python -m tools.night_queue --plan main --list    # 只看计划与预算
  python -m tools.night_queue --plan main           # 主网格：5 N × 3 seed，按 seed 优先
  python -m tools.night_queue --plan main --seed 1  # 只跑 seed 1
  python -m tools.night_queue --plan main --no-score --hours 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C

STATE = C.BUILD / "queue_state.json"


def _entry(dataset: str, n: int, seed: int, stop_eps=None, epochs: int | None = None,
           stop_law: bool = False, eps: float | None = None) -> dict:
    ep = C.epochs_for(n) if epochs is None else int(epochs)
    return dict(dataset=dataset, n=n, seed=seed, epochs=ep,
                ckpt_every=C.ckpt_every_for(ep), stop_eps=stop_eps,
                stop_law=stop_law, eps=C.EPS_MEM if eps is None else eps)


# 深度档（不等步数）：等步数预算下 N=2000/4000 全是 M=0（低于检出限），
# 律的形状只剩 3 档可拟合，留一 N 外推检验必挂（实测 LOO R²≈0.44）。
# 所以给大 N 单独加预算，让它们的 M 也进入可测区间（目标 M≈0.05–0.2）。
# 预算按"M=0.02 所需步数 ∝ N^1.3"估：N=1000≈1.1e4、2000≈2.6e4、4000≈6.4e4 步。
DEEP = {1000: 2400, 2000: 2900, 4000: 1400}   # epoch（×N/BATCH = 步数）

# seed 1 的最终预算（= 等步数 + 深度档合并）。复现实验必须用同一套预算，
# 否则不同 seed 的曲线长度不同，平均/拟合都会串。
FINAL = {250: 5125, 500: 2550, 1000: 2400, 2000: 2900, 4000: 1400}


def build_plan(name: str) -> list[dict]:
    if name == "mvp":
        return [_entry("pathmnist", n, 1) for n in (250, 1000, 4000)]
    if name == "main":
        # 按 seed 优先：先把 seed 1 的五档跑完，就有了一条完整曲线，而不是
        # 一堆不完整的半条曲线。
        return [_entry("pathmnist", n, s) for s in C.SEEDS for n in C.N_GRID]
    if name == "deep":
        # 只给大 N 加预算（小 N 在等步数下已经饱和），用于律的形状识别。
        return [_entry("pathmnist", n, s, epochs=DEEP[n]) for s in (1,) for n in (1000, 2000, 4000)]
    if name == "replicate":
        # seed 2/3 复现：预算与 seed 1 完全一致（FINAL）。
        return [_entry("pathmnist", n, s, epochs=FINAL[n]) for s in (2, 3) for n in C.N_GRID]
    if name == "cross":
        # 跨数据集推广性：两档 N（250 已有饱和、1000 能看到起跳），预算与
        # pathmnist 的 N=250/1000 完全一致，这样 γ 的口径可比。
        return [_entry(ds, n, 1, epochs=FINAL[n]) for ds in ("bloodmnist", "pneumoniamnist")
                for n in (250, 1000)]
    if name == "night":
        # deep 之后的一整夜：先补跨数据集（PPT 明明写了要有），再跑 seed 2 复现。
        return build_plan("cross") + [_entry("pathmnist", n, 2, epochs=FINAL[n]) for n in C.N_GRID]
    if name == "controller":
        return [_entry("pathmnist", n, 1, stop_eps=C.EPS_MEM) for n in C.N_GRID]
    if name == "ctl":
        # 真·开环律早停（策略 C）：在训练循环里按 E*(N,eps) 自己停。
        # 用新 seed 4，避免覆盖主网格的 seed 1 结果；对照用 seed 1 的满预算曲线。
        return [_entry("pathmnist", n, 4, epochs=FINAL[n], stop_law=True) for n in (250, 1000)]
    raise SystemExit(f"未知 plan: {name}")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"done": [], "failed": [], "scored": []}


def save_state(s: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def _done_epochs(run: str) -> int:
    rj = C.RESULTS / "runs" / f"{run}.json"
    if not rj.exists():
        return 0
    try:
        return int(json.loads(rj.read_text(encoding="utf-8")).get("epochs_done", 0))
    except Exception:  # noqa: BLE001
        return 0


def _skey(run: str) -> str:
    """scored 的键要带分辨率 —— 28px 评过的 run 换了 64px 必须重评。"""
    return f"{run}@{C.IMAGE_SIZE}"


def score(run: str) -> None:
    cmd = [str(C.ROOT / ".venv" / "Scripts" / "python.exe"), "-u", "-m", "tools.score_run",
           "--run", run, "--fid", "--fid-stride", "4"]
    print(f"  评分 {run} …", flush=True)
    r = subprocess.run(cmd, cwd=str(C.ROOT))
    if r.returncode != 0:
        print(f"  评分 {run} 返回 {r.returncode}（不阻塞队列）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main")
    ap.add_argument("--epochs", type=int, default=None, help="降级用：把每个 run 的 epoch 压到不超过这个数")
    ap.add_argument("--seed", type=int, default=None, help="只跑某一批 seed")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--hours", type=float, default=0, help=">0 时超过该时长后不再开始新 run")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    plan = build_plan(a.plan)
    if a.seed is not None:
        plan = [p for p in plan if p["seed"] == a.seed]
    if a.epochs:
        for p in plan:
            p["epochs"] = min(p["epochs"], a.epochs)
            p["ckpt_every"] = C.ckpt_every_for(p["epochs"])
    if a.list:
        tot = 0.0
        for p in plan:
            tag = C.run_tag(p["dataset"], p["n"], p["seed"])
            print(f"{tag:28s} epochs<={p['epochs']:5d} ckpt_every={p['ckpt_every']:4d} "
                  f"points~{p['epochs']//p['ckpt_every']:3d} stop_eps={p.get('stop_eps')}")
        print(f"共 {len(plan)} 个 run")
        return

    state = load_state()
    t0 = time.time()
    for i, p in enumerate(plan):
        run = C.run_tag(p["dataset"], p["n"], p["seed"])
        if a.hours and (time.time() - t0) / 3600 > a.hours:
            print(f"已用 {(time.time()-t0)/60:.0f} min，超过 --hours，收工")
            break
        # 用「实际训练到第几轮」判跳过，不能只看 state["done"]：
        # deep 档给同一个 run 加了预算，只按 state 会把它们全部误跳过。
        done_ep = _done_epochs(run)
        if done_ep >= p["epochs"]:
            print(f"[{i+1}/{len(plan)}] {run} 已训练到 {done_ep} 轮，跳过训练")
            if run not in state["done"]:
                state["done"].append(run)
                save_state(state)
        else:
            print(f"\n[{i+1}/{len(plan)}] {run} 训练 epochs<={p['epochs']} "
                  f"ckpt_every={p['ckpt_every']}（已用 {(time.time()-t0)/60:.1f} min）", flush=True)
            try:
                import train

                stop_epoch, stop_rule = None, ""
                if p.get("stop_law"):
                    import law as L

                    j = json.loads((C.RESULTS / "law_fit.json").read_text(encoding="utf-8"))
                    best = j["best"]
                    e_star = L.solve_E_star(best, j["fits"][best]["params"],
                                            float(p["n"]), float(p.get("eps", C.EPS_MEM)))
                    if e_star == e_star:  # 非 nan
                        stop_epoch = int(-(-e_star // 1))
                        stop_rule = f"{best} eps={p.get('eps', C.EPS_MEM)} E*={e_star:.0f}"
                        print(f"  [law] {stop_rule} → 第 {stop_epoch} 轮停", flush=True)
                    else:
                        print("  [law] 该 N 永远达不到 eps —— 数据本身安全", flush=True)

                train.train_run(p["dataset"], p["n"], p["seed"],
                                epochs=p["epochs"], stop_eps=p.get("stop_eps"),
                                ckpt_every=p["ckpt_every"],
                                stop_epoch=stop_epoch, stop_rule=stop_rule)
                state["done"].append(run)
            except Exception as e:  # noqa: BLE001
                print(f"{run} 训练失败：{type(e).__name__}: {e}")
                state["failed"].append({"run": run, "error": f"{type(e).__name__}: {e}"})
            save_state(state)

        if not a.no_score and _skey(run) not in state.get("scored", []):
            try:
                score(run)
                state.setdefault("scored", []).append(_skey(run))
            except Exception as e:  # noqa: BLE001
                print(f"{run} 评分失败：{type(e).__name__}: {e}")
            save_state(state)

    print(f"\n队列结束：done={len(state['done'])} scored={len(state.get('scored', []))} "
          f"failed={len(state['failed'])} 总用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()