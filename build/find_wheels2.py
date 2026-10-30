import re
import urllib.request

BASE = "https://download.pytorch.org/whl/cu128"


def index(pkg):
    html = urllib.request.urlopen(f"{BASE}/{pkg}/", timeout=60).read().decode("utf-8", "ignore")
    hrefs = [h.split("#")[0] for h in re.findall(r'href="([^"]+)"', html)]
    return [h for h in hrefs if h.endswith(".whl")]


for pkg in ("torch", "torchvision"):
    whls = index(pkg)
    win = [h for h in whls if "win_amd64" in h and "cp312" in h]
    print(f"=== {pkg}: {len(whls)} wheels, {len(win)} cp312-win")
    for h in win[-5:]:
        print("   ", h.rsplit("/", 1)[-1], "\n      ", h)

    if win:
        u = win[-1]
        try:
            req = urllib.request.Request(u, method="HEAD")
            t0 = __import__("time").time()
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f"    HEAD ok {r.status} size={int(r.headers.get('Content-Length', 0))/1e9:.2f} GB "
                      f"in {__import__('time').time()-t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"    HEAD FAIL {type(e).__name__}: {e}")