#!/usr/bin/env python3
"""The browser client's network, for a hosted server.

A browser can only open WebSockets, and the game speaks TCP, so every socket
the client opens is a WebSocket to this relay, which opens the TCP connection
and passes bytes both ways (see src/platform/web_socket.cpp).

    wowee-relay.py --listen 127.0.0.1:8090 --allow 127.0.0.1:3724 --allow 127.0.0.1:8085

Unlike tools/serve-wasm.py's built-in relay, which is for localhost testing
and will connect anywhere it is asked, this one opens a connection only to a
target named on its command line. Without that, anyone who can reach the page
could use the server to reach anything the server can reach - including the
machines behind it.

It listens on localhost and expects a web server in front of it, which is
where TLS and any password live: the page asks for wss://<host>/relay, and
that route is proxied here.
"""
import argparse
import base64
import hashlib
import http.server
import socket
import struct
import sys
import threading
import urllib.parse

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class RelayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    allowed: set[tuple[str, int]] = set()

    def log_message(self, fmt, *args):
        sys.stderr.write("relay: " + (fmt % args) + "\n")

    def do_GET(self):
        route = urllib.parse.urlsplit(self.path)
        if route.path != "/relay":
            self.send_error(404, "only /relay?target=host:port")
            return
        target = urllib.parse.parse_qs(route.query).get("target", [""])[0]
        host, _, port = target.rpartition(":")
        key = self.headers.get("Sec-WebSocket-Key")
        if not host or not port.isdigit() or not key:
            self.send_error(400, "want /relay?target=host:port as a WebSocket")
            return
        if (host, int(port)) not in self.allowed:
            self.log_message("refused %s", target)
            self.send_error(403, "not a target this relay serves")
            return
        try:
            upstream = socket.create_connection((host, int(port)), timeout=10)
        except OSError as e:
            self.log_message("%s unreachable: %s", target, e)
            self.send_error(502, f"{target} unreachable")
            return
        upstream.settimeout(None)
        upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        self.log_message("open to %s", target)

        client = self.connection
        send_lock = threading.Lock()

        def send_frame(opcode, payload):
            n = len(payload)
            if n < 126:
                head = struct.pack("!BB", 0x80 | opcode, n)
            elif n < 65536:
                head = struct.pack("!BBH", 0x80 | opcode, 126, n)
            else:
                head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
            with send_lock:
                client.sendall(head + payload)

        def upstream_to_client():
            try:
                while True:
                    data = upstream.recv(65536)
                    if not data:
                        break
                    send_frame(0x2, data)
            except OSError:
                pass
            try:
                send_frame(0x8, b"")
            except OSError:
                pass

        pump = threading.Thread(target=upstream_to_client, daemon=True)
        pump.start()
        rfile = self.rfile
        try:
            while True:
                head = rfile.read(2)
                if len(head) < 2:
                    break
                opcode, n = head[0] & 0x0F, head[1] & 0x7F
                if n == 126:
                    n = struct.unpack("!H", rfile.read(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", rfile.read(8))[0]
                mask = rfile.read(4) if head[1] & 0x80 else b"\0\0\0\0"
                data = bytearray(rfile.read(n))
                for i in range(len(data)):
                    data[i] ^= mask[i & 3]
                if opcode == 0x8:          # close
                    break
                if opcode == 0x9:          # ping
                    send_frame(0xA, bytes(data))
                elif opcode in (0x0, 0x1, 0x2):
                    upstream.sendall(data)
        except OSError:
            pass
        finally:
            try:
                upstream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            upstream.close()
            pump.join(timeout=2)
            self.close_connection = True
            self.log_message("closed %s", target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8090", help="host:port to listen on")
    ap.add_argument("--allow", action="append", required=True, metavar="HOST:PORT",
                    help="a game server this relay may open a connection to; repeatable")
    args = ap.parse_args()

    for target in args.allow:
        host, _, port = target.rpartition(":")
        if not host or not port.isdigit():
            ap.error(f"--allow wants host:port, not {target!r}")
        RelayHandler.allowed.add((host, int(port)))

    host, _, port = args.listen.rpartition(":")
    with http.server.ThreadingHTTPServer((host, int(port)), RelayHandler) as httpd:
        print(f"Relay on {args.listen} to {', '.join(sorted(f'{h}:{p}' for h, p in RelayHandler.allowed))}",
              flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
