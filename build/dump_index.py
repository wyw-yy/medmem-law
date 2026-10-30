import re
import urllib.request

for pkg in ("torch", "torchvision"):
    url = f"https://download.pytorch.org/whl/cu128/{pkg}/"
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
    hrefs = re.findall(r'href="([^"]+)"', html)
    whl = [h for h in hrefs if h.endswith(".whl")]
    print(f"=== {pkg}: {len(hrefs)} hrefs, {len(whl)} whl")
    for h in whl[:6]:
        print("   ", h)
    win = [h for h in whl if "win_amd64" in h]
    print(f"    win_amd64: {len(win)}")
    for h in win[-4:]:
        print("   ", h)
    if not whl:
        print("    RAW:", html[:600])