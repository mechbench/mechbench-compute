#!/usr/bin/env bash
# Build the CPython WASI guest from a pinned tag with wasi-sdk, and a
# trimmed standard library to mount beside it. Prints the sha256 + size
# of python.wasm and the stdlib. Needs a host python3.13 (Homebrew),
# git, curl, tar; no other toolchain (wasi-sdk and the wasmtime CLI are
# fetched here). All modules build STATIC, so the stdlib is pure .py.
set -euo pipefail
cd "$(dirname "$0")"
CPYTHON_TAG=v3.13.3
WASI_SDK=wasi-sdk-34
WASMTIME=v48.0.2

mkdir -p build/upstream && cd build/upstream

# wasi-sdk (clang targeting wasm32-wasip1)
if [ ! -d wasi-sdk ]; then
  curl -sL -o wasi-sdk.tar.gz \
    "https://github.com/WebAssembly/wasi-sdk/releases/download/${WASI_SDK}/${WASI_SDK}.0-arm64-macos.tar.gz"
  tar xzf wasi-sdk.tar.gz && mv "${WASI_SDK}.0-arm64-macos" wasi-sdk
fi
# wasmtime CLI (the build's host-runner for the fresh python.wasm)
if [ ! -x wasmtime/wasmtime ]; then
  curl -sL -o wasmtime.tar.xz \
    "https://github.com/bytecodealliance/wasmtime/releases/download/${WASMTIME}/wasmtime-${WASMTIME}-aarch64-macos.tar.xz"
  tar xf wasmtime.tar.xz && mv "wasmtime-${WASMTIME}-aarch64-macos" wasmtime
fi
# CPython at the pinned tag
if [ ! -d cpython ]; then
  git clone -q --depth 1 --branch "${CPYTHON_TAG}" https://github.com/python/cpython.git cpython
fi

export WASI_SDK_PATH="$PWD/wasi-sdk"
export PATH="$PWD/wasmtime:$PATH"
cd cpython
# wasi.py sets SOURCE_DATE_EPOCH from the tag's commit — reproducible.
python3.13 Tools/wasm/wasi.py build

WASM=cross-build/wasm32-wasip1/python.wasm
cd "$(dirname "$0")"/../..    # back to guests/cpython
cp "build/upstream/cpython/${WASM}" build/python.wasm

# The stdlib to mount: pure .py, minus test suites, the IDE, tk,
# packaging bootstrap, and caches. ~10 MB, 530 files.
rm -rf build/stdlib && cp -R build/upstream/cpython/Lib build/stdlib
( cd build/stdlib
  rm -rf test idlelib tkinter turtledemo turtle.py ensurepip lib2to3 \
         pydoc_data site-packages
  find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true )

echo "python.wasm  sha256 $(shasum -a 256 build/python.wasm | cut -d' ' -f1)"
echo "python.wasm  size   $(stat -f %z build/python.wasm 2>/dev/null || stat -c %s build/python.wasm)"
echo "stdlib       files  $(find build/stdlib -name '*.py' | wc -l | tr -d ' ')"
echo "stdlib       digest (mount by fs-snapshot digest at register time)"
echo "toolchain    ${CPYTHON_TAG} + ${WASI_SDK} + wasmtime ${WASMTIME}; host $(python3.13 --version)"
