#!/usr/bin/env bash
# Builds the WebAssembly dependencies Emscripten does not ship as ports.
#
# SDL2 and zlib come from Emscripten's own ports (-sUSE_SDL=2, -sUSE_ZLIB=1)
# and glm and the Vulkan headers through FetchContent in CMakeLists. OpenSSL
# has no port and does not build with CMake, so it is built here once, the same
# way tools/build-android-deps.sh does it, and handed to the client build with
# -DOPENSSL_ROOT_DIR.
#
#   source ~/emsdk/emsdk_env.sh
#   tools/build-wasm-deps.sh
#
# The result lands in build-wasm-deps/, which is gitignored.
set -euo pipefail

OPENSSL_VERSION="${OPENSSL_VERSION:-3.5.1}"

if ! command -v emcc >/dev/null; then
    echo "emcc not on PATH. Activate the SDK first:" >&2
    echo "  source ~/emsdk/emsdk_env.sh" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/build-wasm-deps"
PREFIX="$WORK/wasm32"
mkdir -p "$WORK"

if [ -f "$PREFIX/lib/libcrypto.a" ]; then
    echo "OpenSSL already built: $PREFIX"
else
    TARBALL="$WORK/openssl-$OPENSSL_VERSION.tar.gz"
    SRC="$WORK/openssl-$OPENSSL_VERSION"
    [ -f "$TARBALL" ] || curl -fL --retry 3 -o "$TARBALL" \
        "https://github.com/openssl/openssl/releases/download/openssl-$OPENSSL_VERSION/openssl-$OPENSSL_VERSION.tar.gz"
    [ -d "$SRC" ] || tar -xzf "$TARBALL" -C "$WORK"

    # linux-generic32 because wasm32 is a 32-bit target with no assembly
    # OpenSSL knows how to emit. -pthread because the client links with shared
    # memory, and wasm-ld refuses to mix objects built without atomics into
    # that. no-dso: there is no dlopen to load engines or providers with, and
    # no-sock: the client does its own networking and never asks OpenSSL to.
    (cd "$SRC" && CC=emcc AR=emar RANLIB=emranlib CFLAGS="-O2 -pthread" \
        ./Configure linux-generic32 no-asm no-shared no-dso no-engine no-sock \
            no-tests no-apps no-docs --prefix="$PREFIX" --openssldir="$PREFIX/ssl" \
            --libdir=lib)
    make -C "$SRC" -j"$(( $(nproc) / 2 ))" build_libs
    make -C "$SRC" install_dev
fi

echo
echo "Configure the client with:"
echo "  emcmake cmake -S . -B build-wasm -DOPENSSL_ROOT_DIR=$PREFIX"
