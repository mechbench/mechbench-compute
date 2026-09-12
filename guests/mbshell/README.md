# mbshell — the sandbox's shell guest

go-busybox's applets behind an in-process POSIX shell, compiled to
`wasip1`. Task 000359.

## Why not busybox's own `sh`

go-busybox's `ash` is a real fork/exec shell: every pipeline stage is a
child process, and WASI preview1 cannot create one. Upstream stubs it
out under wasm (`ash: not supported in wasm`). The C busybox has the
same shape. So the shell has to run every command **in the same
process**, on goroutines, with pipes made of `io.Pipe`.

[mvdan/sh](https://github.com/mvdan/sh) (BSD-3-Clause) interprets the
same language that way already and hands every simple command to an
exec handler. `main.go` is that handler: a table lookup into the
go-busybox applets, wired to the pipe ends through `core.Stdio`. No
process is ever created; `find . | wc -l` is two goroutines and a pipe.

## What is patched, and why (`mvdan-sh-wasi.patch`, v3.12.0)

- `os.Pipe()` → `io.Pipe()` at the three sites (pipelines, heredocs,
  here-strings). TinyGo's `os.Pipe` on wasip1 returns ENOSYS.
- The interpreter's `stdin` is an `io.Reader`, not an `*os.File`;
  `stdinFile` is the identity. The file type only existed so exec'd
  processes could inherit a descriptor.
- `access()` on wasip1 is an existence check. WASI exposes no
  permission bits, and a stat reporting `0o000` made `cd` fail
  silently.
- `read -t` keeps its deadline only when stdin supports one.

`/dev/null` is served by `main.go`'s open handler — the sandbox
preopens one directory and there is no `/dev` in it.

## What does not work in this guest

- Applets that exec a command themselves — `xargs`, `find -exec`,
  `time`, `timeout`, `watch`, `nohup`, `nice` — fail with
  `exec: "wc": executable file not found`. Routing them through the
  in-process table needs a seam in go-busybox (a fork), not done yet.
- `awk` panics: TinyGo lacks `reflect.Type.NumIn`, which goawk needs.
- Standard Go's `GOOS=wasip1` will not build go-busybox as-is
  (`ls` reads `Stat_t.Blocks`, absent there); TinyGo is the toolchain.

## Build

    ./build.sh     # prints sha256 + size for guests.py

Toolchain used for the recorded hash: go 1.27.1, tinygo 0.42.0
(LLVM 22.1.4), macOS arm64. Output ≈ 2.8 MB.

## Licensing, before hosting

mvdan/sh is BSD-3-Clause. go-busybox's README says MIT but the tree at
the pinned commit has **no LICENSE file**; hosting a built artifact is
redistribution and waits on that (see 000359).
