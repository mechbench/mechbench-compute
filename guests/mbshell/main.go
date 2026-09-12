// mbshell: go-busybox's applets behind an in-process POSIX shell.
//
// go-busybox's own ash is a fork/exec shell and WASI cannot spawn a
// process, so upstream stubs it out under wasm. mvdan/sh interprets
// the same language but runs pipelines on goroutines and hands every
// command to an exec handler — which here is a table lookup into the
// applets, wired to the pipe ends through core.Stdio. No process is
// ever created; `find . | wc -l` is two goroutines and an io.Pipe.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"mvdan.cc/sh/v3/expand"
	"mvdan.cc/sh/v3/interp"
	"mvdan.cc/sh/v3/syntax"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-busybox/pkg/applets/awk"
	"github.com/rcarmo/go-busybox/pkg/applets/cat"
	"github.com/rcarmo/go-busybox/pkg/applets/cp"
	"github.com/rcarmo/go-busybox/pkg/applets/cut"
	"github.com/rcarmo/go-busybox/pkg/applets/diff"
	"github.com/rcarmo/go-busybox/pkg/applets/dig"
	"github.com/rcarmo/go-busybox/pkg/applets/echo"
	"github.com/rcarmo/go-busybox/pkg/applets/find"
	"github.com/rcarmo/go-busybox/pkg/applets/free"
	"github.com/rcarmo/go-busybox/pkg/applets/grep"
	"github.com/rcarmo/go-busybox/pkg/applets/gunzip"
	"github.com/rcarmo/go-busybox/pkg/applets/gzip"
	"github.com/rcarmo/go-busybox/pkg/applets/head"
	"github.com/rcarmo/go-busybox/pkg/applets/ionice"
	"github.com/rcarmo/go-busybox/pkg/applets/kill"
	"github.com/rcarmo/go-busybox/pkg/applets/killall"
	"github.com/rcarmo/go-busybox/pkg/applets/logname"
	"github.com/rcarmo/go-busybox/pkg/applets/ls"
	"github.com/rcarmo/go-busybox/pkg/applets/mkdir"
	"github.com/rcarmo/go-busybox/pkg/applets/mv"
	"github.com/rcarmo/go-busybox/pkg/applets/nc"
	"github.com/rcarmo/go-busybox/pkg/applets/nice"
	"github.com/rcarmo/go-busybox/pkg/applets/nohup"
	"github.com/rcarmo/go-busybox/pkg/applets/nproc"
	"github.com/rcarmo/go-busybox/pkg/applets/pgrep"
	"github.com/rcarmo/go-busybox/pkg/applets/pidof"
	"github.com/rcarmo/go-busybox/pkg/applets/printf"
	"github.com/rcarmo/go-busybox/pkg/applets/pkill"
	"github.com/rcarmo/go-busybox/pkg/applets/ps"
	"github.com/rcarmo/go-busybox/pkg/applets/pwd"
	"github.com/rcarmo/go-busybox/pkg/applets/renice"
	"github.com/rcarmo/go-busybox/pkg/applets/rm"
	"github.com/rcarmo/go-busybox/pkg/applets/rmdir"
	"github.com/rcarmo/go-busybox/pkg/applets/sed"
	"github.com/rcarmo/go-busybox/pkg/applets/setsid"
	"github.com/rcarmo/go-busybox/pkg/applets/sleep"
	"github.com/rcarmo/go-busybox/pkg/applets/sort"
	"github.com/rcarmo/go-busybox/pkg/applets/ss"
	"github.com/rcarmo/go-busybox/pkg/applets/startstopdaemon"
	"github.com/rcarmo/go-busybox/pkg/applets/tail"
	"github.com/rcarmo/go-busybox/pkg/applets/tar"
	"github.com/rcarmo/go-busybox/pkg/applets/taskset"
	"github.com/rcarmo/go-busybox/pkg/applets/time"
	"github.com/rcarmo/go-busybox/pkg/applets/timeout"
	"github.com/rcarmo/go-busybox/pkg/applets/top"
	"github.com/rcarmo/go-busybox/pkg/applets/tr"
	"github.com/rcarmo/go-busybox/pkg/applets/uniq"
	"github.com/rcarmo/go-busybox/pkg/applets/uptime"
	"github.com/rcarmo/go-busybox/pkg/applets/users"
	"github.com/rcarmo/go-busybox/pkg/applets/w"
	"github.com/rcarmo/go-busybox/pkg/applets/watch"
	"github.com/rcarmo/go-busybox/pkg/applets/wc"
	"github.com/rcarmo/go-busybox/pkg/applets/wget"
	"github.com/rcarmo/go-busybox/pkg/applets/who"
	"github.com/rcarmo/go-busybox/pkg/applets/whoami"
	"github.com/rcarmo/go-busybox/pkg/applets/xargs"
	"github.com/rcarmo/go-busybox/pkg/core"
)

type appletFunc func(stdio *core.Stdio, args []string) int

var applets = map[string]appletFunc{
	"echo":              echo.Run,
	"awk":               awk.Run,
	"cat":               cat.Run,
	"ls":                ls.Run,
	"cp":                cp.Run,
	"mv":                mv.Run,
	"free":              free.Run,
	"pidof":             pidof.Run,
	"printf":            printf.Run,
	"pgrep":             pgrep.Run,
	"pkill":             pkill.Run,
	"logname":           logname.Run,
	"nice":              nice.Run,
	"nproc":             nproc.Run,
	"rm":                rm.Run,
	"rmdir":             rmdir.Run,
	"head":              head.Run,
	"kill":              kill.Run,
	"killall":           killall.Run,
	"tail":              tail.Run,
	"wc":                wc.Run,
	"find":              find.Run,
	"sort":              sort.Run,
	"mkdir":             mkdir.Run,
	"pwd":               pwd.Run,
	"renice":            renice.Run,
	"uniq":              uniq.Run,
	"cut":               cut.Run,
	"grep":              grep.Run,
	"sed":               sed.Run,
	"tr":                tr.Run,
	"diff":              diff.Run,
	"ps":                ps.Run,
	"ss":                ss.Run,
	"dig":               dig.Run,
	"gzip":              gzip.Run,
	"gunzip":            gunzip.Run,
	"tar":               tar.Run,
	"sleep":             sleep.Run,
	"uptime":            uptime.Run,
	"whoami":            whoami.Run,
	"who":               who.Run,
	"users":             users.Run,
	"w":                 w.Run,
	"top":               top.Run,
	"time":              time.Run,
	"timeout":           timeout.Run,
	"setsid":            setsid.Run,
	"nohup":             nohup.Run,
	"watch":             watch.Run,
	"taskset":           taskset.Run,
	"ionice":            ionice.Run,
	"xargs":             xargs.Run,
	"start-stop-daemon": startstopdaemon.Run,
	"wget":              wget.Run,
	"nc":                nc.Run,
}

func init() {
	applets["sh"] = shell
	applets["ash"] = shell
	// Applets that would exec a command (xargs, time, timeout) call
	// back into this table instead. See go-busybox-wasi.patch.
	core.RunCommand = func(stdio *core.Stdio, name string, args []string) (int, bool) {
		run, ok := applets[name]
		if !ok {
			return 127, false
		}
		return call(run, stdio, name, args), true
	}
}

// call runs one applet and turns a panic inside it into a failed
// command rather than a dead sandbox. Known panics: TinyGo's
// os.File.Read returns a negative count on a directory and bufio
// panics on it (`wc -c .`); awk trips a reflect gap.
func call(run appletFunc, stdio *core.Stdio, name string, args []string) (code int) {
	defer func() {
		if p := recover(); p != nil {
			fmt.Fprintf(stdio.Err, "%s: internal error: %v\n", name, p)
			code = 2
		}
	}()
	return run(stdio, args)
}

// exit ends the program with `code`. WASI hosts reject proc_exit
// outside [0, 126) and discard the number, so a status of 126 or more
// — 127 is "command not found" — goes through the sandbox's side
// channel when it is mounted, and the process exits 125.
func exit(code int) {
	if code >= 126 {
		_ = os.WriteFile("/.mechbench/exit", []byte(strconv.Itoa(code)), 0o644)
		code = 125
	}
	os.Exit(code)
}

func main() {
	stdio := core.DefaultStdio()
	applet, args := resolveApplet(os.Args)
	if applet == "" {
		printAppletList(stdio)
		exit(core.ExitUsage)
	}
	run, ok := applets[applet]
	if !ok {
		stdio.Errorf("busybox: applet not found: %s\n", applet)
		printAppletList(stdio)
		exit(core.ExitUsage)
	}
	exit(call(run, stdio, applet, args))
}

func resolveApplet(args []string) (string, []string) {
	if len(args) == 0 {
		return "", nil
	}
	if len(args) > 1 && filepath.Base(args[0]) == "busybox" {
		return args[1], args[2:]
	}
	return filepath.Base(args[0]), args[1:]
}

func printAppletList(stdio *core.Stdio) {
	stdio.Println("Currently defined functions:")
	for name := range applets {
		stdio.Print(" ", name)
	}
	stdio.Println()
}

// dispatch is the exec handler: every simple command the shell would
// have exec'd becomes an in-process applet call on the shell's own
// stdio, in the shell's own directory.
func dispatch(next interp.ExecHandlerFunc) interp.ExecHandlerFunc {
	return func(ctx context.Context, args []string) error {
		hc := interp.HandlerCtx(ctx)
		if len(args) == 0 {
			return nil
		}
		run, ok := applets[args[0]]
		if !ok {
			fmt.Fprintf(hc.Stderr, "sh: %s: not found\n", args[0])
			return interp.ExitStatus(127)
		}
		if hc.Dir != "" {
			if err := os.Chdir(hc.Dir); err != nil {
				fmt.Fprintf(hc.Stderr, "sh: cd %s: %v\n", hc.Dir, err)
				return interp.ExitStatus(1)
			}
		}
		code := call(run, &core.Stdio{In: hc.Stdin, Out: hc.Stdout, Err: hc.Stderr}, args[0], args[1:])
		if code != 0 {
			return interp.ExitStatus(code)
		}
		return nil
	}
}

// shell is `sh -c CMD [NAME [ARG…]]`, `sh FILE [ARG…]`, or `sh` reading
// the script on stdin. NAME becomes $0, as POSIX says.
func shell(stdio *core.Stdio, args []string) int {
	var src io.Reader
	name := "sh"
	params := args
	switch {
	case len(args) >= 2 && args[0] == "-c":
		src = strings.NewReader(args[1])
		params = args[2:]
		if len(params) > 0 {
			name, params = params[0], params[1:]
		}
	case len(args) >= 1 && !strings.HasPrefix(args[0], "-"):
		f, err := os.Open(args[0])
		if err != nil {
			stdio.Errorf("sh: %v\n", err)
			return 127
		}
		defer f.Close()
		src, name, params = f, args[0], args[1:]
	default:
		src = stdio.In
	}
	prog, err := syntax.NewParser().Parse(src, name)
	if err != nil {
		stdio.Errorf("sh: %v\n", err)
		return 2
	}
	cwd, _ := os.Getwd()
	if cwd == "" {
		cwd = "/"
	}
	runner, err := interp.New(
		interp.ExecHandlers(dispatch),
		interp.OpenHandler(openWithDevNull(interp.DefaultOpenHandler())),
		interp.StdIO(stdio.In, stdio.Out, stdio.Err),
		interp.Env(expand.ListEnviron(os.Environ()...)),
		interp.Dir(cwd),
		interp.Params(params...),
	)
	if err != nil {
		stdio.Errorf("sh: %v\n", err)
		return 2
	}
	err = runner.Run(context.Background(), prog)
	if err == nil {
		return 0
	}
	var status interp.ExitStatus
	if errors.As(err, &status) {
		return int(status)
	}
	stdio.Errorf("sh: %v\n", err)
	return 1
}

// openWithDevNull serves /dev/null itself: the sandbox preopens one
// directory and there is no /dev in it. Everything else is the default.
func openWithDevNull(next interp.OpenHandlerFunc) interp.OpenHandlerFunc {
	return func(ctx context.Context, path string, flag int, perm os.FileMode) (io.ReadWriteCloser, error) {
		if path == "/dev/null" {
			return devNull{}, nil
		}
		return next(ctx, path, flag, perm)
	}
}

type devNull struct{}

func (devNull) Read([]byte) (int, error)    { return 0, io.EOF }
func (devNull) Write(p []byte) (int, error) { return len(p), nil }
func (devNull) Close() error                { return nil }
