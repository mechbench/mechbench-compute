# cpython — the sandbox's Python guest

CPython 3.13.3 compiled to `wasip1` with wasi-sdk, plus its standard
library as a directory to mount read-only. Task 000453.

## Shape

`python.wasm` (~28 MB) is the interpreter; every module builds STATIC
(the WASI build produces no shared objects), so the standard library
is **pure `.py` files** — `build/stdlib/`, ~10 MB, 530 files after
trimming test suites, the IDE, tk, and the packaging bootstrap.

This is the mount design 000358/000360 wanted: the stdlib is a
read-only tree beside the working snapshot, not frozen into the binary,
so a user can extend it with pure-Python packages by layering another
mount. It runs with the stdlib preopened at `/usr/local/lib/python3.13`
and `PYTHONHOME=/usr/local`.

## What works (smoke-tested in the sandbox)

- `python -c`, scripts, `os.walk` + read + write into the snapshot.
- The common stdlib: json, re, math, collections, datetime, hashlib
  (HACL backends — md5/sha1/sha2/sha3/blake2, no OpenSSL), base64,
  textwrap, itertools, functools, statistics, decimal, csv, io,
  unicodedata.
- Determinism: under strict mode the hash seed and `random` are
  virtualized, so `hash('x')` and a seeded RNG are stable run to run.
- A failed allocation is a catchable `MemoryError`, not a hard trap.

## What does NOT work (by design or by platform)

- Network: `socket` imports but cannot connect (no sockets).
- `subprocess` / `os.fork`: "wasi does not support processes."
- Missing optional modules (no C libs under WASI): `ssl`, `hashlib`'s
  OpenSSL backend, `lzma`, `bz2`, `zlib`, `ctypes`, `sqlite3`,
  `readline`, `_uuid`. `hashlib` still has its built-in HACL hashes.
- `sys.exit(n)` for n >= 126: WASI hosts reject `proc_exit` outside
  [0, 126) and CPython calls it directly, so the status clamps to 126.
  A `sitecustomize` hook writing `/.mechbench/exit` (the sandbox side
  channel) is the fix — see 000453.

## Build

    ./build.sh    # fetches wasi-sdk + wasmtime CLI, builds, trims,
                  # prints sha256/size for guests.py

Toolchain for the recorded build: CPython v3.13.3, wasi-sdk-34,
wasmtime v48.0.2, host Homebrew python3.13. A cold build is a few
minutes (it compiles a build-Python then cross-compiles the host).

## Licensing

CPython is under the PSF License Agreement — permissive, redistribution
allowed with the license text. `NOTICE`/`LICENSE` ship beside the
hosted artifact (as for mbshell).
