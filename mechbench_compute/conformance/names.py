from __future__ import annotations

import re

VERBS = {
    "ablate", "add", "aggregate", "align", "apply", "assert", "attach", "attribute", "average",
    "bin", "bind", "bootstrap", "build", "bump", "cache", "calculate", "call", "cap",
    "capture", "chat", "check", "choose", "classify", "clamp", "clear",
    "close", "coerce", "collect", "compare", "compile", "compute", "contrast",
    "convert", "copy", "correlate", "count", "cross", "decode", "decompose", "delete", "derive", "evaluate",
    "describe", "detach", "diff", "digest", "dispatch", "drop", "dump", "edit", "emit",
    "encode", "ensure", "estimate", "expand", "expect", "extend", "extract",
    "fetch", "fill", "filter", "find", "finish", "fit", "flatten",
    "fold", "force", "format", "freeze", "fuse", "gather", "generate", "get",
    "group", "grow", "guess", "hash", "index", "infer", "iter", "join",
    "judge", "keep", "label", "list", "load", "log", "lookup", "make", "map",
    "mark", "match", "materialize", "measure", "merge", "move", "name",
    "normalize", "note", "open", "orthogonalize", "pack", "parse", "patch",
    "pick", "place", "plan", "plot", "pop", "prepare", "project", "prune",
    "publish", "push", "put", "quote", "raise", "rank", "read", "record",
    "reduce", "refuse", "register", "regress", "reject", "relabel", "release", "remove",
    "rename", "render", "repair", "replace", "report", "require", "reshape", "resolve",
    "replay", "resample", "restore", "reverse", "rewrite", "round", "run", "sample", "save", "say",
    "scale", "scan", "score", "seed", "select", "send", "serialize", "set",
    "shape", "shift", "show", "skip", "slice", "sort", "span", "split",
    "stack", "start", "steer", "stop", "store", "strip", "subtract", "sum",
    "summarize", "sweep", "tabulate", "take", "test", "time", "tokenize",
    "total", "trace", "track", "train", "transform", "trim", "truncate", "try", "unembed",
    "union", "unnest", "unpack", "update", "use", "validate", "verify", "walk", "wrap",
    "write", "yield", "zip",
}

PREDICATES = {"is", "has", "can", "should", "must", "needs", "wants", "are",
              "fuses", "satisfies", "matches", "holds"}

KEEP_NAMES = {"run", "OP", "MONOID", "main"}


def normalize_name(name: str) -> str:
    bare = name.lstrip("_")
    if bare.isupper():
        return bare.lower()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", bare).lower()


def read_head(name: str) -> str:
    return normalize_name(name).split("_")[0]


def is_verb_first(name: str) -> bool:
    head = read_head(name)
    return head in VERBS or head in PREDICATES
