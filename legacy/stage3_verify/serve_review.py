#!/usr/bin/env python3
"""带落盘能力的静态服务器 —— 替代 python -m http.server 用于 adjudicate.html。

为什么需要它：`python -m http.server` 只读不写。adjudicate.html 的裁决默认存在
浏览器 localStorage 里，而 localStorage 绑定 origin（scheme://host:port）——
换端口、换浏览器、清缓存都会丢；而"导出 JSON"下载到的是**你本地机器**，
还得再传回开发机才能跑 apply_human_adj.py。

本脚本加一个 POST /save 端点，页面每答一题就把全量裁决写到开发机磁盘上，
所以关页面、换端口、断网重连都不丢，也不需要手动传文件。

用法（在开发机上）:
    python scripts/serve_review.py --dir . --port 8899
然后本地浏览器开 http://<开发机>:8899/adjudicate.html

访问控制：本机的 http.server 被换成了带 ACL 的版本（不在白名单的客户端一律 403）。
本脚本复用同一套白名单机制，行为与 `python -m http.server` 一致 —— 如果你现在能用
`python -m http.server` 看到页面，那用这个脚本也能看到。若被 403，用
--allowedip / --allowednet 追加你本地机器的 IP。
"""
import argparse, json, os, shutil, threading, time
import http.server as _hs
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def setup_acl(allowedip=None, allowednet=None, allowedbns=None):
    """复用本机 patch 过的 http.server 的 ACL 白名单机制。

    这台机器的 /usr/lib/python3.10/http/server.py 被替换成了带访问控制的版本
    （Secure_SimpleHTTP）：模块级 ACL_ON=True、IP_WHITELIST=[]，SimpleHTTPRequestHandler.do_GET
    会先查白名单，不在就直接 403。白名单由 `python -m http.server` 的 main 启动一个
    update_acl 后台线程填充（含 127.0.0.1、本机 IP、BNS 解析出的一批 IP）。

    我们直接 import 该模块当库用，那个线程不会自动起，于是 IP_WHITELIST 恒为空 → 全部 403。
    这里手动起同一个线程，行为与 `python -m http.server` 完全一致，既不绕过公司的访问控制，
    也不改动系统文件。
    """
    if not getattr(_hs, "ACL_ON", False) or not hasattr(_hs, "update_acl"):
        return False
    t = threading.Thread(target=_hs.update_acl,
                         args=(allowedip, allowednet, allowedbns), daemon=True)
    t.start()
    for _ in range(50):                     # 等首轮 BNS 解析填好白名单
        if _hs.IP_WHITELIST:
            break
        time.sleep(0.2)
    return True


def make_handler(root, save_path):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, fmt, *a):
            # 静音每次 GET，只在保存时打点，避免刷屏
            pass

        def _acl_ok(self):
            if not getattr(_hs, "ACL_ON", False):
                return True
            ip = self.client_address[0]
            return (ip in _hs.IP_WHITELIST
                    or self.is_ip_in_subnet(ip, _hs.NET_WHITELIST))

        def do_POST(self):
            # 写接口与读接口用同一套 ACL，避免只拦 GET 不拦 POST
            if not self._acl_ok():
                self.send_error(403, "Access is denied")
                return
            if self.path.rstrip("/") != "/save":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                data = json.loads(body)
                # 先写临时文件再 rename：避免答题中途崩溃留下半截 JSON
                tmp = save_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, save_path)
                # 每 20 次留一个带时间戳的快照，防误操作（比如页面上点了重置）
                cnt = sum(1 for a in data.get("answers", []) if a.get("human"))
                if cnt and cnt % 20 == 0:
                    shutil.copy(save_path, f"{save_path}.{cnt:04d}")
                print(f"[{time.strftime('%H:%M:%S')}] saved {cnt} answers -> {save_path}",
                      flush=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "count": cnt}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())

        def do_GET(self):
            # 页面启动时拉一次已存裁决，实现跨浏览器/跨端口续答
            if self.path.rstrip("/") == "/load":
                if not self._acl_ok():
                    self.send_error(403, "Access is denied")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                if os.path.exists(save_path):
                    self.wfile.write(open(save_path, "rb").read())
                else:
                    self.wfile.write(b'{"answers":[]}')
                return
            super().do_GET()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--save", default=None,
                    help="裁决落盘路径，默认 <dir>/out/human_adjudication.json")
    ap.add_argument("--allowedip", default=None, help="额外放行的客户端 IP，逗号分隔")
    ap.add_argument("--allowednet", default=None, help="额外放行的网段 CIDR，逗号分隔")
    ap.add_argument("--allowedbns", default=None, help="额外放行的 BNS")
    a = ap.parse_args()
    root = os.path.abspath(a.dir)
    save = a.save or os.path.join(root, "out", "human_adjudication.json")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    if setup_acl(a.allowedip, a.allowednet, a.allowedbns):
        print(f"[ACL] 已启用本机访问控制，白名单 {len(_hs.IP_WHITELIST)} 个 IP"
              f"（不在名单内的客户端会收到 403；用 --allowedip/--allowednet 追加）")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(root, save))
    print(f"serving {root} on 0.0.0.0:{a.port}")
    print(f"裁决落盘 -> {save}")
    print(f"打开     -> http://<开发机地址>:{a.port}/adjudicate.html")
    if os.path.exists(save):
        try:
            d = json.load(open(save))
            print(f"已有裁决 {sum(1 for x in d.get('answers',[]) if x.get('human'))} 条，页面会自动续答")
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
