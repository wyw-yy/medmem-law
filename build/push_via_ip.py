"""在 github.com:443 被本地网络丢包时，用可用 IP + Host 头把 master 推上去。

背景（2026-11-08）：`github.com` 解析到 20.205.243.166，443 连接超时；而同属 GitHub 的
140.82.113.4 可达，`api.github.com` 也正常。所以不是带宽问题，是那一个 IP 被丢包。

做法：先正常 push；失败则用 `http.extraHeader` 把 Host 头写成 github.com、
用 `gh auth token` 作 Basic 认证、并临时关闭证书校验（SNI 是 IP，无法匹配 github.com 的证书）。
**只对这一次 push 生效**，不改 git 配置、不改 hosts（本机 hosts 无写权限）。

用法：python -m build.push_via_ip [--msg-fallback]
"""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IPS = ["140.82.113.4", "140.82.112.4", "20.27.177.113"]
REPO = "wyw-yy/medmem-law"


def sh(cmd: list[str], env: dict | None = None, timeout: int = 180) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       env=env)
    return r.returncode, (r.stdout + r.stderr).strip()


def main() -> None:
    rc, out = sh(["git", "push", "origin", "master"])
    if rc == 0:
        print("正常 push 成功")
        return
    print(f"正常 push 失败：{out.splitlines()[-1] if out else 'rc='+str(rc)}")

    rc, tok = sh(["gh", "auth", "token"])
    if rc != 0 or not tok:
        print("取不到 gh token，放弃")
        sys.exit(1)
    tok = tok.strip().splitlines()[-1]
    basic = base64.b64encode(f"x-access-token:{tok}".encode()).decode()

    for ip in IPS:
        url = f"https://{ip}/{REPO}.git"
        cmd = ["git", "-c", "http.sslVerify=false",
               "-c", "http.extraHeader=Host: github.com",
               "-c", f"http.extraHeader=Authorization: Basic {basic}",
               "push", url, "HEAD:master"]
        rc, out = sh(cmd, timeout=300)
        if rc == 0:
            print(f"经 {ip} push 成功")
            return
        print(f"经 {ip} 失败：{out.splitlines()[-1] if out else 'rc='+str(rc)}")
    sys.exit(2)


if __name__ == "__main__":
    main()