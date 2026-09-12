#!/usr/bin/env bash
# Build mbshell.wasm from pinned sources with standard Go's wasip1 port.
# Prints the sha256 and size to record in mechbench_compute/guests.py.
# Needs go on PATH (brew install go). -trimpath and -buildvcs=false keep
# the build independent of where the tree sits and of its git state, so
# the hash reproduces on any machine with the same toolchain.
set -euo pipefail
cd "$(dirname "$0")"
BUSYBOX_COMMIT=13f3053fa3ddf4589baa610feb268f076a9555ea
SH_VERSION=v3.12.0

mkdir -p build/upstream
if [ ! -d build/upstream/go-busybox ]; then
  git clone -q https://github.com/rcarmo/go-busybox build/upstream/go-busybox
fi
git -C build/upstream/go-busybox checkout -q --force "$BUSYBOX_COMMIT"
git -C build/upstream/go-busybox clean -fdq
patch -s -p1 -d build/upstream/go-busybox < go-busybox-wasi.patch

rm -rf build/upstream/mvdan-sh
go mod download "mvdan.cc/sh/v3@$SH_VERSION"
cp -R "$(go env GOMODCACHE)/mvdan.cc/sh/v3@$SH_VERSION" build/upstream/mvdan-sh
chmod -R u+w build/upstream/mvdan-sh
patch -s -p1 -d build/upstream/mvdan-sh < mvdan-sh-wasi.patch

# -buildvcs=false: this module sits inside the compute git repo, and Go
# would otherwise stamp the binary with that repo's commit and dirty flag —
# a different hash on every machine and every commit. Provenance is the
# pinned inputs, not the tree the recipe happened to be run from.
GOOS=wasip1 GOARCH=wasm go build -trimpath -buildvcs=false -o build/mbshell.wasm .
echo "sha256 $(shasum -a 256 build/mbshell.wasm | cut -d' ' -f1)"
echo "size   $(stat -f %z build/mbshell.wasm 2>/dev/null || stat -c %s build/mbshell.wasm)"
echo "source go-busybox@$BUSYBOX_COMMIT + go-busybox-wasi.patch + mvdan.cc/sh/v3@$SH_VERSION + mvdan-sh-wasi.patch; $(go version | cut -d' ' -f3)"
