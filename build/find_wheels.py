"""列出 pytorch cu128 索引里 cp312 win_amd64 的 torch / torchvision wheel 直链。"""
import re
import urllib.request

BASE = "https://download.pytorch.org/whl/cu128"
for pkg in ("torch", "torchvision"):
    url = f"{BASE}/{pkg}/"
    print(f"--- {url}")
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    print(f"    page {len(html)/1e6:.1f} MB")
    links = re.findall(r'href="([^"]+\.whl)(#[^"]*)?"', html)
    cand = [l for l in links if "cp312-cp312-win_amd64" in l and pkg in l.split("/")[-1]]
    cand = sorted(set(cand))
    for c in cand[-6:]:
        print("   ", c)