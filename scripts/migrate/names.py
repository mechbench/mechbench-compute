"""Every name in the package, who else says it, and whether it is a verb.

Analysis only; renames nothing. Re-runnable: the inventory is read out of
the source every time, so after another refactor this says what the new
tree looks like.

Two questions per definition, and a name is a candidate for renaming if
either answer is bad:

  (a) does it cross files?  A leading underscore says "mine alone". On a
      name eight files import, it says something false. Cross-file
      references are counted through `from X import name`, through an
      aliased module plus an attribute (`interp._last_logp`), and
      through a string a test patches with
      (`monkeypatch.setattr(mod, "name", ...)`, `patch("a.b.name")`).

  (b) is it a verb?  A function's name should say what it does, not what
      it returns, so that a call site reads as an action. The first
      token is looked up in VERBS below; `is_`/`has_` predicates are
      counted separately, since they already read as a question.

      python scripts/migrate/names.py            # the TSV, every definition
      python scripts/migrate/names.py --cand     # candidates only
      python scripts/migrate/names.py --summary  # counts, and unknown first tokens

The candidate set is (a) every cross-file name with a leading
underscore, plus (b) every top-level function under `ops/` or in a
one-definition-per-file helper module (`<topic>/<name>.py`, the file
named for what it holds) whose name is not a verb.
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "mechbench_compute"
SOURCES = ("mechbench_compute", "tests", "scripts")

#: Verbs the codebase already uses well, plus the ones the rename is
#: meant to move names towards. First token of a function name only.
VERBS = {
    "ablate", "add", "aggregate", "apply", "assert", "attribute", "average",
    "bin", "bootstrap", "build", "bump", "cache", "calculate", "call", "cap",
    "capture",
    "chat", "check", "choose", "classify", "clamp", "clear", "close",
    "coerce", "collect", "compare", "compile", "compute", "contrast",
    "convert", "copy", "count", "cross", "decode", "decompose", "delete",
    "describe", "digest", "dispatch", "drop", "dump", "edit", "emit",
    "encode", "ensure", "estimate", "expand", "expect", "extend", "extract",
    "fetch", "fill", "filter", "find", "finish", "fit", "flatten",
    "fold", "force", "format", "freeze", "fuse", "gather", "generate", "get",
    "group", "grow",
    "guess", "hash", "index", "infer", "iter", "join", "judge", "keep",
    "label", "list", "load", "log", "lookup", "make", "map", "mark", "match",
    "materialize", "measure", "merge", "move", "name", "normalize", "note",
    "open", "orthogonalize", "pack", "parse", "patch", "pick", "place",
    "plan", "plot", "pop", "prepare", "project", "prune", "publish", "push",
    "put", "quote",
    "raise", "rank", "read", "record", "reduce", "refuse", "register",
    "regress", "reject", "release", "remove", "rename", "render", "repair",
    "replace", "report", "require", "resolve", "restore", "reverse",
    "rewrite", "round", "run", "sample", "save", "say", "scale", "scan",
    "score", "seed", "select", "send", "serialize", "set", "shape", "shift",
    "show",
    "skip", "slice", "sort", "span", "split", "stack", "start", "steer",
    "stop", "store", "strip", "subtract", "sum", "summarize", "sweep",
    "tabulate", "take", "test", "time", "tokenize", "total", "trace",
    "track", "train", "trim", "truncate", "try", "unembed", "union",
    "unpack", "update", "use", "validate", "verify", "walk", "wrap", "write",
    "yield", "zip",
}

#: Not verbs, but they already read as a question at a call site. Kept
#: out of the candidate set and reported on their own line.
PREDICATES = {"is", "has", "can", "should", "must", "needs", "wants", "are",
              "fuses", "satisfies", "matches", "holds"}

#: Left alone by the rules of the task: the executor's entry point, the
#: two declarations a file's shape is read from, a script's entry point,
#: and the public declaration vocabulary in `lexicon/`. The executor's
#: methods are exempt for free — they are not top-level definitions.
KEEP_NAMES = {"run", "OP", "MONOID", "main"}
KEEP_DIRS = ("lexicon/",)

PATCHY = re.compile(r"setattr|getattr|patch|delattr|hasattr")


def snake(name: str) -> str:
    bare = name.lstrip("_")
    if bare.isupper():
        return bare.lower()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", bare).lower()


def rel_of_module(dotted: str) -> str | None:
    """`mechbench_compute.interp.last_logp` -> `interp/last_logp.py`, and
    a package -> its `__init__.py`."""
    if dotted == "mechbench_compute":
        return "__init__.py"
    if not dotted.startswith("mechbench_compute."):
        return None
    stem = dotted[len("mechbench_compute."):].replace(".", "/")
    if (PKG / f"{stem}.py").is_file():
        return f"{stem}.py"
    if (PKG / stem / "__init__.py").is_file():
        return f"{stem}/__init__.py"
    return None


@dataclass
class Def:
    file: str
    name: str
    kind: str                   # function | class | constant
    line: int
    refs: set[str] = field(default_factory=set)      # other package files that say it
    outside: set[str] = field(default_factory=set)   # tests and scripts that say it
    reexports: set[str] = field(default_factory=set)  # __init__.py files that only pass it on

    @property
    def crosses(self) -> bool:
        """Another module of the package names it. A test naming it does
        not count: reaching into a module's privates is what a test is
        allowed to do, and the underscore is still true."""
        return bool(self.refs)


class Mod:
    """One package module: what it defines, and what it takes from the
    rest of the package (which is also how it re-exports)."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.tree = ast.parse((PKG / rel).read_text())
        self.defs: dict[str, Def] = {}
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = Def(rel, node.name, "function", node.lineno)
            elif isinstance(node, ast.ClassDef):
                self.defs[node.name] = Def(rel, node.name, "class", node.lineno)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.defs[t.id] = Def(rel, t.id, "constant", node.lineno)
        self.imports, self.modules = bindings(self.tree)


def bindings(tree: ast.AST) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """What a file binds from the package, wherever the import sits — some
    are lazy, inside a function. Returns (name -> (module, original
    name)) and (name -> module) for a module bound as a whole."""
    imports: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            if not (n.module == "mechbench_compute" or n.module.startswith("mechbench_compute.")):
                continue
            for a in n.names:
                bound = a.asname or a.name
                sub = f"{n.module}.{a.name}"
                if rel_of_module(sub):
                    modules[bound] = sub          # `from mechbench_compute import points as P`
                # Both, and the name is tried first: in this layout a
                # helper's module and the name the package imports back
                # from it are spelled the same
                # (`from ...tools import build_toolbox`), and Python
                # gives the package's own binding, not the submodule.
                imports[bound] = (n.module, a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if not a.name.startswith("mechbench_compute"):
                    continue
                if a.asname:
                    modules[a.asname] = a.name
                else:
                    modules[a.name] = a.name      # `import mechbench_compute.interp` -> dotted use
    return imports, modules


def attr_chain(node: ast.Attribute) -> list[str] | None:
    """`mechbench_compute.interp.last_logp._last_logp` -> the parts."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return parts


class Inventory:
    def __init__(self) -> None:
        self.mods: dict[str, Mod] = {}
        for path in sorted(PKG.rglob("*.py")):
            rel = str(path.relative_to(PKG))
            try:
                self.mods[rel] = Mod(rel)
            except SyntaxError:
                print(f"!! cannot parse {rel}", file=sys.stderr)
        self.defs = {(m.rel, name): d for m in self.mods.values() for name, d in m.defs.items()}
        self.notes: list[str] = []
        for base in SOURCES:
            for path in sorted((ROOT / base).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                self.scan(path)

    def resolve(self, dotted: str, name: str, seen: tuple = ()) -> tuple[str, str] | None:
        """Where the thing that module calls `name` is really defined —
        through however many `__init__.py` re-exports it takes."""
        rel = rel_of_module(dotted)
        if rel is None or rel not in self.mods:
            return None
        mod = self.mods[rel]
        if name in mod.defs:
            return (rel, name)
        if name in mod.imports and (dotted, name) not in seen:
            nxt, orig = mod.imports[name]
            return self.resolve(nxt, orig, seen + ((dotted, name),))
        if name in mod.modules:
            return None                        # a module, not a definition
        return None

    def module_at(self, dotted: str, name: str) -> str | None:
        """The module `dotted.name` names: a submodule, or a module that
        module binds under that name — `judge.chat_mod` is
        `mechbench_compute.chat`, and a rename has to see through it."""
        sub = f"{dotted}.{name}"
        if rel_of_module(sub):
            return sub
        rel = rel_of_module(dotted)
        if rel is None or rel not in self.mods:
            return None
        return self.mods[rel].modules.get(name)

    def walk_modules(self, start: str, parts: list[str]) -> str | None:
        cur: str | None = start
        for p in parts:
            cur = self.module_at(cur, p) if cur else None
        return cur

    def hit(self, target: tuple[str, str] | None, source: str) -> None:
        if target is None:
            return
        d = self.defs.get(target)
        if d is None or d.file == source:
            return
        # The package `__init__.py` a helper left imports it back so that
        # nothing naming it through the module breaks (OPS_LAYOUT.md).
        # That is not a second file saying the name: every file that
        # names it through the module is already counted, at the far end
        # of the chain. Counting the re-export would make a helper
        # nobody uses look as though it crossed a file boundary.
        home = d.file.rsplit("/", 1)[0] + "/__init__.py" if "/" in d.file else "__init__.py"
        if source == home:
            d.reexports.add(source)
        elif source.startswith(("tests/", "scripts/")):
            d.outside.add(source)
        else:
            d.refs.add(source)

    def scan(self, path: Path) -> None:
        rel_root = str(path.relative_to(ROOT))
        source = str(path.relative_to(PKG)) if rel_root.startswith("mechbench_compute/") else rel_root
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            return
        imports, modules = bindings(tree)
        for module, orig in imports.values():
            self.hit(self.resolve(module, orig), source)
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                parts = attr_chain(n)
                if not parts or len(parts) < 2:
                    continue
                # `alias.NAME`, and `alias.mid.NAME` where each step is
                # a module — a submodule, or one a module binds.
                base = modules.get(parts[0])
                if base is None:
                    continue
                dotted = self.walk_modules(base, parts[1:-1])
                if dotted:
                    self.hit(self.resolve(dotted, parts[-1]), source)
            elif isinstance(n, ast.Call) and PATCHY.search(ast.unparse(n.func)):
                for arg in n.args:
                    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                        continue
                    text = arg.value
                    if "." in text and text.startswith("mechbench_compute"):
                        dotted, _, name = text.rpartition(".")
                        got = self.resolve(dotted, name)
                        if got:
                            self.hit(got, source)
                            self.notes.append(f"{rel_root}:{n.lineno} string patch {text!r}")
                        continue
                    # `monkeypatch.setattr(mod, "name", ...)`: the module
                    # is the argument before the string.
                    for other in n.args:
                        base = other.id if isinstance(other, ast.Name) else None
                        chain = attr_chain(other) if isinstance(other, ast.Attribute) else None
                        dotted = modules.get(base) if base else None
                        if dotted is None and chain and chain[0] in modules:
                            dotted = ".".join([modules[chain[0]]] + chain[1:])
                        if dotted is None:
                            continue
                        got = self.resolve(dotted, text)
                        if got:
                            self.hit(got, source)
                            self.notes.append(f"{rel_root}:{n.lineno} string patch {dotted}.{text}")


def verbiness(d: Def) -> str:
    if d.kind != "function":
        return "-"
    head = snake(d.name).split("_")[0]
    if head in VERBS:
        return "verb"
    if head in PREDICATES:
        return "predicate"
    return "NOT-A-VERB"


def helper_file(inv: Inventory, rel: str) -> bool:
    """A one-definition-per-file helper module, in the sense
    docs/OPS_LAYOUT.md gives it: in a topic package, not an
    `__init__.py`, named for what it holds, and holding that one thing —
    or two, where a class and the function that builds it share a file
    (`Plan` and `plan`). The bound is what tells the move's helper files
    apart from an older module that happens to carry its own name."""
    if rel.endswith("__init__.py") or "/" not in rel or rel.startswith("ops/"):
        return False
    stem = rel[:-3].rsplit("/", 1)[1]
    made = [d for d in inv.mods[rel].defs.values() if d.kind in ("function", "class")]
    return len(made) <= 2 and any(snake(d.name) == stem for d in made)


def exempt(rel: str, name: str) -> bool:
    return name in KEEP_NAMES or name.startswith("__") or rel.startswith(KEEP_DIRS)


def rows(inv: Inventory) -> list[dict]:
    out: list[dict] = []
    for (rel, name), d in sorted(inv.defs.items()):
        verb = verbiness(d)
        reasons = []
        if not exempt(rel, name):
            if name.startswith("_") and d.crosses:
                reasons.append("underscore-crosses-files")
            if verb == "NOT-A-VERB" and (rel.startswith("ops/") or helper_file(inv, rel)):
                reasons.append("not-a-verb")
        out.append({
            "file": rel, "name": name, "kind": d.kind, "line": d.line,
            "refs": len(d.refs), "outside": len(d.outside),
            "reexported": "yes" if d.reexports else "no",
            "verb": verb,
            "reason": "+".join(reasons) or "-",
            "ref_files": ",".join(sorted(d.refs | d.outside)),
        })
    return out


COLUMNS = ["file", "name", "kind", "line", "refs", "outside", "reexported",
           "verb", "reason", "ref_files"]


def main() -> None:
    inv = Inventory()
    table = rows(inv)
    if "--summary" in sys.argv:
        cand = [r for r in table if r["reason"] != "-"]
        print(f"{len(inv.mods)} modules, {len(table)} top-level definitions")
        print(f"{sum(1 for r in table if r['refs'])} cross files")
        by_kind = Counter(r["kind"] for r in cand)
        by_reason = Counter(r["reason"] for r in cand)
        print(f"{len(cand)} candidates   by kind {dict(by_kind)}   by reason {dict(by_reason)}")
        heads = Counter(snake(r["name"]).split("_")[0] for r in table
                        if r["kind"] == "function" and r["verb"] == "NOT-A-VERB")
        print("\nfirst tokens that are not in VERBS (review these):")
        for head, n in heads.most_common(60):
            print(f"  {n:4}  {head}")
        per_file = defaultdict(int)
        for r in cand:
            per_file[r["file"]] += 1
        print(f"\n{len(per_file)} files hold a candidate")
        if inv.notes:
            print("\nnames reached through a string, which a rename must follow by hand:")
            for note in sorted(set(inv.notes)):
                print("  ", note)
        return
    only = "--cand" in sys.argv
    print("\t".join(COLUMNS))
    for r in table:
        if only and r["reason"] == "-":
            continue
        print("\t".join(str(r[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
