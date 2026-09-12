module mechbench.ai/mbshell

go 1.25.6

require (
	github.com/rcarmo/go-busybox v0.0.0
	mvdan.cc/sh/v3 v3.12.0
)

require (
	github.com/benhoyt/goawk v1.25.0 // indirect
	golang.org/x/sys v0.40.0 // indirect
	golang.org/x/term v0.39.0 // indirect
)

// Both upstreams are pinned and prepared by build.sh into build/upstream/.
replace github.com/rcarmo/go-busybox => ./build/upstream/go-busybox

replace mvdan.cc/sh/v3 => ./build/upstream/mvdan-sh
