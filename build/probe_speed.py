"""测速：HF 镜像 vs（DoH 解析后 IP 直连）Zenodo。各下 8 秒。"""
import socket
import time
import urllib.request

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}


def speed(label, url, seconds=8, host=None, ip=None):
    if host and ip:
        orig = socket.getaddrinfo

        def patched(h, *a, **k):
            if h == host:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]
            return orig(h, *a, **k)

        socket.getaddrinfo = patched
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=15) as r:
            while time.time() - t0 < seconds:
                b = r.read(1 << 20)
                if not b:
                    break
                got += len(b)
    except Exception as e:  # noqa: BLE001
        print(f"{label}: ERR {type(e).__name__}: {e}  got={got/1e6:.1f}MB in {time.time()-t0:.1f}s")
    else:
        el = time.time() - t0
        print(f"{label}: {got/1e6:.1f} MB in {el:.1f}s = {got/1e6/el:.2f} MB/s")
    finally:
        if host and ip:
            socket.getaddrinfo = orig


def doh(name):
    for u in [f"https://223.5.5.5/resolve?name={name}&type=A",
              f"https://doh.pub/dns-query?name={name}&type=A"]:
        try:
            import json
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                u, headers={**H, "Accept": "application/dns-json"}), timeout=15))
            ips = [a["data"] for a in r.get("Answer", []) if a.get("type") == 1]
            if ips:
                return ips
        except Exception as e:  # noqa: BLE001
            print(f"  doh {u} -> {type(e).__name__}")
    return []


if __name__ == "__main__":
    speed("hf-mirror pathmnist", "https://hf-mirror.com/datasets/albertvillanova/medmnist-v2/resolve/main/data/pathmnist.npz")
    ips = doh("zenodo.org")
    print("zenodo A:", ips)
    if ips:
        speed("zenodo by IP (pathmnist_64)",
              "https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1",
              host="zenodo.org", ip=ips[0])