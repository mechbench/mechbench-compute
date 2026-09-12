#!/usr/bin/env bash
# Build mbshell.wasm from pinned sources. Prints the sha256 and size to
# record in mechbench_compute/guests.py. Needs go and tinygo on PATH
# (brew trust tinygo-org/tools && brew install go tinygo).
set -euo pipefail
cd "$(dirname "$0")"
BUSYBOX_COMMIT=13f3053fa3ddf4589baa610feb268f076a9555ea
SH_VERSION=v3.12.0

mkdir -p build/upstream
if [ ! -d build/upstream/go-busybox ]; then
  git clone -q https://github.com/rcarmo/go-busybox build/upstream/go-busybox
fi
git -C build/upstream/go-busybox checkout -q "$BUSYBOX_COMMIT"

rm -rf build/upstream/mvdan-sh
go mod download "mvdan.cc/sh/v3@$SH_VERSION"
cp -R "$(go env GOMODCACHE)/mvdan.cc/sh/v3@$SH_VERSION" build/upstream/mvdan-sh
chmod -R u+w build/upstream/mvdan-sh
patch -s -p1 -d build/upstream/mvdan-sh < mvdan-sh-wasi.patch

tinygo build -target=wasip1 -opt=z -no-debug -o build/mbshell.wasm .
echo "sha256 $(shasum -a 256 build/mbshell.wasm | cut -d' ' -f1)"
echo "size   $(stat -f %z build/mbshell.wasm 2>/dev/null || stat -c %s build/mbshell.wasm)"
echo "source go-busybox@$BUSYBOX_COMMIT + mvdan.cc/sh/v3@$SH_VERSION + mvdan-sh-wasi.patch"
