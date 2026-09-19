#!/usr/bin/env python3
"""Serves the wasm build to a browser on this machine.

The client runs worker threads, which need SharedArrayBuffer, which a browser
only enables for a page that is cross-origin isolated - so the page has to be
served with the two headers below. `python3 -m http.server` does not send them,
and without them the page stops at "SharedArrayBuffer is not defined".

    tools/serve-wasm.py [dir] [port]      # defaults: build-wasm/bin 8080
    then open http://localhost:8080/wowee.html

Game data is served from Data/ beside the page (the build links it there) and
fetched by the client a file at a time; see src/platform/web_data.cpp.

It is also the client's network: a browser can only open WebSockets, so
/relay?target=host:port turns one into a TCP connection to that game server
and passes bytes both ways (src/platform/web_socket.cpp). It listens on
localhost only - it will connect anywhere it is asked to.
"""
import base64
import functools
import hashlib
import http.server
import os
import socket
import struct
import sys
import threading
import urllib.parse


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    # Keep-alive: the client fetches its game data a file at a time - a few
    # thousand on entering the world - and HTTP/1.0 opened a connection for
    # every one of them.
    protocol_version = "HTTP/1.1"

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

    # The browser build mounts Data/ and fetches files from it as they are
    # read, but it has to be told which exist: this path answers with every
    # file under Data/, one relative path a line (see src/platform/web_data.cpp).
    INDEX_PATH = "/Data/.wowee-index"

    def data_index(self):
        root = os.path.join(self.directory, "Data")
        lines = []
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            rel = os.path.relpath(dirpath, root)
            for name in filenames:
                if name.startswith(".wowee-index"):
                    continue
                lines.append(name if rel == "." else f"{rel}/{name}")
        return ("\n".join(sorted(lines)) + "\n").encode()

    def send_index(self, with_body):
        body = self.data_index()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?")[0].replace("//", "/")
        if route == self.INDEX_PATH:
            return self.send_index(True)
        if route == "/relay":
            return self.relay()
        return super().do_GET()

    # WebSocket relay ------------------------------------------------------

    WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def relay(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        target = query.get("target", [""])[0]
        host, _, port = target.rpartition(":")
        key = self.headers.get("Sec-WebSocket-Key")
        if not host or not port.isdigit() or not key:
            self.send_error(400, "want /relay?target=host:port as a WebSocket")
            return
        try:
            upstream = socket.create_connection((host, int(port)), timeout=10)
        except OSError as e:
            self.log_message("relay: %s unreachable: %s", target, e)
            self.send_error(502, f"{target} unreachable")
            return
        upstream.settimeout(None)
        upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        accept = base64.b64encode(hashlib.sha1((key + self.WS_GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        self.log_message("relay: open to %s", target)

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
            self.log_message("relay: closed %s", target)

    def do_HEAD(self):
        if self.path.split("?")[0].replace("//", "/") == self.INDEX_PATH:
            return self.send_index(False)
        return super().do_HEAD()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "build-wasm", "bin")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    handler = functools.partial(IsolatedHandler, directory=directory)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {directory} at http://localhost:{port}/wowee.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
