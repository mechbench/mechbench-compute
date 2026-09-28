module mechbench.ai/mbshell

go 1.25.6

require (
	github.com/itchyny/gojq v0.12.19
	github.com/rcarmo/go-busybox v0.0.0
	mvdan.cc/sh/v3 v3.12.0
)

require (
	github.com/benhoyt/goawk v1.25.0 // indirect
	github.com/clipperhouse/stringish v0.1.1 // indirect
	github.com/clipperhouse/uax29/v2 v2.3.0 // indirect
	github.com/itchyny/go-yaml v0.0.0-20251001235044-fca9a0999f15 // indirect
	github.com/itchyny/timefmt-go v0.1.8 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mattn/go-runewidth v0.0.19 // indirect
	golang.org/x/sys v0.40.0 // indirect
	golang.org/x/term v0.39.0 // indirect
)

// All three upstreams are pinned and prepared by build.sh into build/upstream/.
replace github.com/rcarmo/go-busybox => ./build/upstream/go-busybox

replace mvdan.cc/sh/v3 => ./build/upstream/mvdan-sh

replace github.com/itchyny/gojq => ./build/upstream/gojq
