#!/usr/bin/env python3
"""Serves the wasm build to a browser on this machine.

The client runs worker threads, which need SharedArrayBuffer, which a browser
only enables for a page that is cross-origin isolated - so the page has to be
served with the two headers below. `python3 -m http.server` does not send them,
and without them the page stops at "SharedArrayBuffer is not defined".

    tools/serve-wasm.py [dir] [port]      # defaults: build-wasm/bin 8080
    then open http://localhost:8080/wowee.html
"""
import functools
import http.server
import os
import sys


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

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
