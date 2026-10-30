"""健壮的 resumable 下载器（绕开 pip/requests，纯 urllib + Range 续传 + 重试）。

用法：
  python build\fetch.py <url> <out_path> [--seconds-cap N]
"""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def fetch(url, out, cap_bytes=None, chunk=1 << 20, max_stall=90):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    attempts = 0
    while attempts < 200:
        attempts += 1
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=max_stall) as r:
                total = r.headers.get("Content-Length")
                crange = r.headers.get("Content-Range")
                if crange and "/" in crange:
                    total_n = int(crange.split("/")[1])
                elif total:
                    total_n = have + int(total)
                else:
                    total_n = None
                if have and r.status == 200:
                    # 服务器不支持 Range，重头来
                    have = 0
                    part.unlink(missing_ok=True)
                t0 = time.time()
                got = have
                with open(part, "ab") as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        if total_n and got % (16 << 20) < chunk:
                            mb = got / 1e6
                            el = max(time.time() - t0, 1e-6)
                            print(f"[fetch] {mb:.1f}/{total_n/1e6:.1f} MB  "
                                  f"{(got - have)/1e6/el:.2f} MB/s  try {attempts}", flush=True)
                        if cap_bytes and got >= cap_bytes:
                            print(f"[fetch] 达到 cap {cap_bytes/1e6:.0f}MB，停止", flush=True)
                            return got
                if total_n and part.stat().st_size >= total_n:
                    part.replace(out)
                    print(f"[fetch] DONE {out} {out.stat().st_size/1e6:.1f} MB", flush=True)
                    return out.stat().st_size
                print(f"[fetch] 连接提前结束 at {got/1e6:.1f} MB, 重试", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] try {attempts} 失败 {type(e).__name__}: {e}", flush=True)
            time.sleep(min(3 + attempts, 20))
    raise SystemExit("下载失败")


if __name__ == "__main__":
    out = sys.argv[2]
    cap = None
    if "--seconds-cap" in sys.argv:
        idx = sys.argv.index("--seconds-cap")
        cap = int(float(sys.argv[idx + 1]) * 1e6)
    fetch(sys.argv[1], out, cap)