# mbshell — the sandbox's shell guest

go-busybox's applets behind an in-process POSIX shell, standard Go
compiled to `wasip1`. Task 000359.

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

## Why standard Go, not TinyGo

go-busybox builds with TinyGo for size (2.8 MB). Every guest failure
found by running it under the sandbox traced to TinyGo's wasm target:
`recover()` does nothing, so any applet panic kills the sandbox;
`os.File.Read` on a directory returns a negative count, which bufio
panics on (`wc -c .`, `sed p .`); `reflect.Type.NumIn` is missing, so
`awk` panics. Standard Go's `wasip1` port (official since 1.21) has
none of these. Same source, one build flag, 15.4 MB (3.8 MB gzipped),
fetched once. 3–5 ms per call.

## What is patched, and why

`mvdan-sh-wasi.patch` (v3.12.0):

- `os.Pipe()` → `io.Pipe()` at the three sites (pipelines, heredocs,
  here-strings). WASI has no pipe syscall.
- The interpreter's `stdin` is an `io.Reader`, not an `*os.File`;
  `stdinFile` is the identity. The file type only existed so exec'd
  processes could inherit a descriptor.
- `access()` on wasip1 is an existence check. WASI exposes no
  permission bits, and a stat reporting `0o000` made `cd` fail
  silently.
- `read -t` keeps its deadline only when stdin supports one.

`go-busybox-wasi.patch` (commit `13f3053`):

- `core.RunCommand`: a hook that runs a command by name in-process.
  `xargs` calls it instead of `exec.Command`; `time` and `timeout` get
  wasm variants that use it (`timeout` parses its duration and does
  not enforce it — nothing to signal, and the sandbox's wall cap is
  the only clock).
- `ls` reads `Stat_t.Blocks`, which wasip1's `Stat_t` lacks.

`main.go` itself: `/dev/null` from the open handler (one preopened
directory, no `/dev`); `$0` from `sh -c CMD NAME`; and exit statuses
≥ 126 — which WASI hosts refuse to carry — written to
`/.mechbench/exit`, a second preopen the runtime reads back.

## What does not work in this guest

- `find -exec` — not implemented upstream.
- `nice`, `nohup`, `setsid`, `ionice`, `taskset`, `watch`,
  `start-stop-daemon` — process management; they exec and fail.
- Network applets (`wget`, `nc`, `dig`) — no sockets to open.

## Build

    ./build.sh     # prints sha256 + size for guests.py

Reproducible: `-trimpath`, `-buildvcs=false` (the recipe lives inside
a git repo and Go would otherwise stamp the binary with its commit and
dirty flag — CI caught that), pinned upstream commit and module
version, toolchain named in the output (go 1.27.1 for the recorded
hash). Same hash from inside the repo and from a copy outside any git
repo. Needs `go` on PATH.

## Hosting and licensing

The built guest is a **GitHub release on this repo**, tagged
`mbshell-<sha12>`, with the gzipped binary, `SHA256SUMS`, and `NOTICE`
beside it. `guests.ensure("mbshell")` fetches it on first use,
decompresses, and verifies the hash of the decompressed bytes against
the pin in `mechbench_compute/guests.py`. To re-pin: run `build.sh`,
create a release with the new tag and assets, update the pin and URL.

`NOTICE` carries every license in the binary: go-busybox is MIT **as
declared in its README** (the tree has no LICENSE file; upstream
issue #3 asks for one), mvdan/sh BSD-3-Clause, goawk MIT, golang.org/x
and the Go runtime BSD-3-Clause.

Nobody needs Go to *use* the guest. Rebuilding it needs go 1.27.1
(a 65 MB download, 270 MB installed), git, patch, and network access
to proxy.golang.org and GitHub; no C toolchain. A cold build is under
four seconds.
