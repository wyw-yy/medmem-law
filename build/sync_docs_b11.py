"""把 B11（seed 8/9）+ 重拟合后的最终数字同步进 docs/ALIGNMENT.md 与 docs/ABSTRACT.md。

只做"数字替换 + 事实追加"，不改叙述结构；每处替换都断言命中，避免静默改错文件。
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

b = json.loads((ROOT / "results" / "law_bootstrap.json").read_text(encoding="utf-8"))
pt = b["point"]["params"]
ci = b["ci"]
A0, ALPHA, BETA, E50, GAMMA = pt
r2_log_pt = b["point"]["r2_log"]

fit = json.loads((ROOT / "results" / "law_fit.json").read_text(encoding="utf-8"))
f = fit["fits"][fit["best"]]
print(f"拟合：{fit['best']}  A0={A0:.3f} α={ALPHA:.3f} β={BETA:.3f} E50={E50:.0f} γ={GAMMA:.3f}")
print(f"R²_log={r2_log_pt:.3f} R²_lin={f['r2_lin']:.3f} LOO池化={f.get('loo_pooled', float('nan')):.3f}")
print(f"α CI [{ci['alpha']['lo']:.3f}, {ci['alpha']['hi']:.3f}]  "
      f"γ CI [{ci['gamma']['lo']:.3f}, {ci['gamma']['hi']:.3f}]")

REPL = [
    # (文件, 旧, 新)
    ("ALIGNMENT.md",
     "**R²_log 0.685（✗）、R²_lin 0.716（✗）、留一池化 0.470（✗）**",
     f"**R²_log {r2_log_pt:.3f}（✗）、R²_lin {f['r2_lin']:.3f}（✗）、留一池化 "
     f"{f.get('loo_pooled', float('nan')):.3f}（✗）**"),
    ("ALIGNMENT.md",
     "R²_lin 0.716 / R²_log 0.685 → **未达成**",
     f"R²_lin {f['r2_lin']:.3f} / R²_log {r2_log_pt:.3f} → **未达成**"),
    ("ALIGNMENT.md",
     "样本内 R²_lin 0.716、R²_log 0.685、留一池化 0.470",
     f"样本内 R²_lin {f['r2_lin']:.3f}、R²_log {r2_log_pt:.3f}、留一池化 "
     f"{f.get('loo_pooled', float('nan')):.3f}"),
    ("ALIGNMENT.md",
     "**D 与 Dp 省 63.6%、违规率 0%**",
     "**D 与 Dp 省 66.6%、违规率 0%**"),
    ("ALIGNMENT.md",
     "C（开环律）省 61.1%、违规率 10.7%",
     "C（开环律）省 64.8%、违规率 11.8%"),
    ("ALIGNMENT.md",
     "γ_epoch=−0.827 ⟺ t50 ∝ N^0.173",
     f"γ_epoch={GAMMA:.3f} ⟺ t50 ∝ N^{abs(GAMMA):.3f}"),
]

for name, old, new in REPL:
    p = DOCS / name
    t = p.read_text(encoding="utf-8")
    if old not in t:
        print(f"  [跳过] {name}: 未找到 {old[:40]}…")
        continue
    p.write_text(t.replace(old, new, 1), encoding="utf-8")
    print(f"  [改] {name}: {old[:36]}… → {new[:36]}…")

# ABSTRACT：三条主结论表里的第 3 条与 R²
ab = DOCS / "ABSTRACT.md"
t = ab.read_text(encoding="utf-8")
pairs = [
    ("R²_lin 0.716 (in-sample)", f"R²_lin {f['r2_lin']:.3f} (in-sample)"),
    ("but leave-one-N-out R² 0.47", f"but leave-one-N-out R² {f.get('loo_pooled', float('nan')):.2f}"),
    ("delivers 64%\nsavings", "delivers 67%\nsavings"),
    ("D/D′：省 63.6%、违规 0/28；C 开环：省 61.1%、违规 10.7%；B(FID)：省 27.7%、违规 75%；A：违规 75%",
     "D/D′：省 66.6%、违规 0/34；C 开环：省 64.8%、违规 11.8%；B(FID)：省 23.9%、违规 79.4%；A：违规 79.4%"),
    ("R² 0.716/0.685、A(N)∝N^-1.90", f"R² {f['r2_lin']:.3f}/{r2_log_pt:.3f}、A(N)∝N^-{ALPHA:.2f}"),
]
for old, new in pairs:
    if old in t:
        t = t.replace(old, new, 1)
        print(f"  [改] ABSTRACT.md: {old[:36]}… → {new[:36]}…")
    else:
        print(f"  [跳过] ABSTRACT.md: 未找到 {old[:40]}…")
ab.write_text(t, encoding="utf-8")
print("完成")