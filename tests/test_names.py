"""What a name says, held where the layout applies (docs/NAMES.md).

Two rules, and both are about the reader of a call site rather than the
writer of a definition:

**A leading underscore means "mine alone".** On a name another module of
the package imports, reads as an attribute, or patches by string, it
says something false, and the reader who believes it is the one who
gets hurt. So a private that crosses a file fails here: give it a name
without the underscore, or move it where it is private again.

**A function's name starts with a verb.** It should say what it does,
not what it returns, so that a call site reads as an action.
`read_last_logp(logits)` is what happens; `last_logp(logits)` is a noun
pretending to be a call. `is_`/`has_` predicates already read as a
question and are counted apart.

The verb rule is held where the layout is: an operation's file under
`ops/`, and a helper file named for the one thing it holds
(docs/OPS_LAYOUT.md). Those are the files whose whole claim is that the
path and the name tell you what is inside. The older topic modules are
not held to it here.

Two judgement calls this gate makes, both about what counts as another
file naming something:

- **A package `__init__.py` importing a name back is a re-export, not a
  second file.** Every file that reaches the name through the package
  is already counted at the far end of the chain, so counting the
  re-export too would make a helper nobody uses look as though it
  crossed a file boundary.
- **A test naming a private does not make the underscore false.**
  Reaching into a module's privates is what a test (and a script) is
  allowed to do; the name is still private to the package.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "mechbench_compute"

#: The trees read for references to a package name.
SOURCES = ("mechbench_compute", "tests", "scripts")

#: First token of a function name. What the codebase already says well,
#: and what a new name is expected to reach for.
VERBS = {
    "ablate", "add", "aggregate", "apply", "assert", "attribute", "average",
    "bin", "bootstrap", "build", "bump", "cache", "calculate", "call", "cap",
    "capture", "chat", "check", "choose", "classify", "clamp", "clear",
    "close", "coerce", "collect", "compare", "compile", "compute", "contrast",
    "convert", "copy", "count", "cross", "decode", "decompose", "delete",
    "describe", "digest", "dispatch", "drop", "dump", "edit", "emit",
    "encode", "ensure", "estimate", "expand", "expect", "extend", "extract",
    "fetch", "fill", "filter", "find", "finish", "fit", "flatten",
    "fold", "force", "format", "freeze", "fuse", "gather", "generate", "get",
    "group", "grow", "guess", "hash", "index", "infer", "iter", "join",
    "judge", "keep", "label", "list", "load", "log", "lookup", "make", "map",
    "mark", "match", "materialize", "measure", "merge", "move", "name",
    "normalize", "note", "open", "orthogonalize", "pack", "parse", "patch",
    "pick", "place", "plan", "plot", "pop", "prepare", "project", "prune",
    "publish", "push", "put", "quote", "raise", "rank", "read", "record",
    "reduce", "refuse", "register", "regress", "reject", "release", "remove",
    "rename", "render", "repair", "replace", "report", "require", "resolve",
    "restore", "reverse", "rewrite", "round", "run", "sample", "save", "say",
    "scale", "scan", "score", "seed", "select", "send", "serialize", "set",
    "shape", "shift", "show", "skip", "slice", "sort", "span", "split",
    "stack", "start", "steer", "stop", "store", "strip", "subtract", "sum",
    "summarize", "sweep", "tabulate", "take", "test", "time", "tokenize",
    "total", "trace", "track", "train", "trim", "truncate", "try", "unembed",
    "union", "unpack", "update", "use", "validate", "verify", "walk", "wrap",
    "write", "yield", "zip",
}

#: Not verbs, but they already read as a question at a call site.
PREDICATES = {"is", "has", "can", "should", "must", "needs", "wants", "are",
              "fuses", "satisfies", "matches", "holds"}

#: The executor's entry point, the two declarations a file's shape is
#: read from, and a script's entry point. `lexicon/` is the public
#: declaration vocabulary, whose names are nouns on purpose — `Op`, `In`,
#: `Output` are what a declaration is written in.
KEEP_NAMES = {"run", "OP", "MONOID", "main"}
KEEP_DIRS = ("lexicon/",)

#: A call that can name a definition in a string.
PATCHY = re.compile(r"setattr|getattr|patch|delattr|hasattr")


def read_tree(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        # A file's own string literals may carry escapes the compiler
        # warns about; this gate reads definitions, not literals.
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(path.read_text())


def normalize_name(name: str) -> str:
    bare = name.lstrip("_")
    if bare.isupper():
        return bare.lower()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", bare).lower()


def find_module(dotted: str) -> str | None:
    """`mechbench_compute.interp.read_pair` -> `interp/read_pair.py`, and
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


def read_bindings(tree: ast.AST) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """What a file binds from the package, wherever the import sits —
    some are lazy, inside a function. Returns (name -> (module, original
    name)) and (name -> module) for a module bound as a whole."""
    imports: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            if not (n.module == "mechbench_compute"
                    or n.module.startswith("mechbench_compute.")):
                continue
            for a in n.names:
                bound = a.asname or a.name
                if find_module(f"{n.module}.{a.name}"):
                    modules[bound] = f"{n.module}.{a.name}"
                # Both, and the name is tried first: a helper's module
                # and the name a package imports back from it are spelled
                # the same, and Python gives the package's own binding.
                imports[bound] = (n.module, a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if not a.name.startswith("mechbench_compute"):
                    continue
                modules[a.asname or a.name] = a.name
    return imports, modules


def read_attr_chain(node: ast.Attribute) -> list[str] | None:
    """`mechbench_compute.interp.read_pair.read_pair` -> the parts."""
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


@dataclass
class Def:
    file: str
    name: str
    kind: str                                        # function | class | constant
    line: int
    refs: set[str] = field(default_factory=set)      # other package files that say it
    outside: set[str] = field(default_factory=set)   # tests and scripts that say it

    @property
    def crosses(self) -> bool:
        return bool(self.refs)


class Mod:
    """One package module: what it defines, and what it takes from the
    rest of the package (which is also how it re-exports)."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.tree = read_tree(PKG / rel)
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
        self.imports, self.modules = read_bindings(self.tree)


class Inventory:
    """Every top-level definition in the package, and every other file
    that says its name."""

    def __init__(self) -> None:
        self.mods: dict[str, Mod] = {}
        for path in sorted(PKG.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            self.mods[str(path.relative_to(PKG))] = Mod(str(path.relative_to(PKG)))
        self.defs = {(m.rel, name): d
                     for m in self.mods.values() for name, d in m.defs.items()}
        for base in SOURCES:
            for path in sorted((ROOT / base).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                self.scan(path)

    def resolve(self, dotted: str, name: str, seen: tuple = ()) -> tuple[str, str] | None:
        """Where the thing a module calls `name` is really defined —
        through however many `__init__.py` re-exports it takes."""
        rel = find_module(dotted)
        if rel is None or rel not in self.mods:
            return None
        mod = self.mods[rel]
        if name in mod.defs:
            return (rel, name)
        if name in mod.imports and (dotted, name) not in seen:
            nxt, orig = mod.imports[name]
            return self.resolve(nxt, orig, seen + ((dotted, name),))
        return None

    def find_module_at(self, dotted: str, name: str) -> str | None:
        """The module `dotted.name` names: a submodule, or a module that
        module binds under that name — `judge.chat_mod` is
        `mechbench_compute.chat`, and a reference has to see through it."""
        if find_module(f"{dotted}.{name}"):
            return f"{dotted}.{name}"
        rel = find_module(dotted)
        if rel is None or rel not in self.mods:
            return None
        return self.mods[rel].modules.get(name)

    def walk_modules(self, start: str, parts: list[str]) -> str | None:
        cur: str | None = start
        for p in parts:
            cur = self.find_module_at(cur, p) if cur else None
        return cur

    def hit(self, target: tuple[str, str] | None, source: str) -> None:
        if target is None:
            return
        d = self.defs.get(target)
        if d is None or d.file == source:
            return
        home = (d.file.rsplit("/", 1)[0] + "/__init__.py"
                if "/" in d.file else "__init__.py")
        if source == home:
            return                                   # a re-export, not a second file
        if source.startswith(("tests/", "scripts/")):
            d.outside.add(source)
        else:
            d.refs.add(source)

    def scan(self, path: Path) -> None:
        rel_root = str(path.relative_to(ROOT))
        source = (str(path.relative_to(PKG))
                  if rel_root.startswith("mechbench_compute/") else rel_root)
        tree = read_tree(path)
        imports, modules = read_bindings(tree)
        for module, orig in imports.values():
            self.hit(self.resolve(module, orig), source)
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                parts = read_attr_chain(n)
                if not parts or len(parts) < 2:
                    continue
                # `alias.NAME`, and `alias.mid.NAME` where each step is a
                # module — a submodule, or one a module binds.
                base = modules.get(parts[0])
                if base is None:
                    continue
                dotted = self.walk_modules(base, parts[1:-1])
                if dotted:
                    self.hit(self.resolve(dotted, parts[-1]), source)
            elif isinstance(n, ast.Call) and PATCHY.search(ast.unparse(n.func)):
                self.scan_patch(n, modules, source)

    def scan_patch(self, call: ast.Call, modules: dict[str, str], source: str) -> None:
        """A definition named in a string: `patch("a.b.name")`, and
        `monkeypatch.setattr(mod, "name", ...)`, where the module is the
        argument before the string."""
        for arg in call.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            text = arg.value
            if "." in text and text.startswith("mechbench_compute"):
                dotted, _, name = text.rpartition(".")
                self.hit(self.resolve(dotted, name), source)
                continue
            for other in call.args:
                base = other.id if isinstance(other, ast.Name) else None
                chain = read_attr_chain(other) if isinstance(other, ast.Attribute) else None
                dotted = modules.get(base) if base else None
                if dotted is None and chain and chain[0] in modules:
                    dotted = ".".join([modules[chain[0]]] + chain[1:])
                if dotted is None:
                    continue
                self.hit(self.resolve(dotted, text), source)


@cache
def take_inventory() -> Inventory:
    return Inventory()


def classify_name(d: Def) -> str:
    if d.kind != "function":
        return "-"
    head = normalize_name(d.name).split("_")[0]
    if head in VERBS:
        return "verb"
    if head in PREDICATES:
        return "predicate"
    return "not-a-verb"


def is_helper_file(inv: Inventory, rel: str) -> bool:
    """A one-definition-per-file helper module, in the sense
    docs/OPS_LAYOUT.md gives it: in a topic package, not an
    `__init__.py`, named for what it holds, and holding that one thing —
    or two, where a class and the function that builds it share a file
    (`Plan` and `plan`). The bound is what tells a helper file apart
    from a topic module that happens to carry its own name."""
    if rel.endswith("__init__.py") or "/" not in rel or rel.startswith("ops/"):
        return False
    stem = rel[:-3].rsplit("/", 1)[1]
    made = [d for d in inv.mods[rel].defs.values() if d.kind in ("function", "class")]
    return len(made) <= 2 and any(normalize_name(d.name) == stem for d in made)


def is_exempt(rel: str, name: str) -> bool:
    return name in KEEP_NAMES or name.startswith("__") or rel.startswith(KEEP_DIRS)


class TestAnUnderscoreMeansMineAlone:
    def test_no_private_name_crosses_a_file(self):
        offences = []
        for (rel, name), d in sorted(take_inventory().defs.items()):
            if is_exempt(rel, name) or not name.startswith("_") or not d.crosses:
                continue
            offences.append(f"{rel}:{d.line} {name} — read by "
                            + ", ".join(sorted(d.refs)))
        assert not offences, (
            f"{len(offences)} name(s) say they are private and are not "
            f"(docs/NAMES.md):\n" + "\n".join(offences)
            + "\nDrop the underscore, or move the definition to the one "
            "file that uses it.")


class TestAFunctionSaysWhatItDoes:
    def test_every_operation_and_helper_function_starts_with_a_verb(self):
        inv = take_inventory()
        offences = []
        for (rel, name), d in sorted(inv.defs.items()):
            if is_exempt(rel, name) or classify_name(d) != "not-a-verb":
                continue
            if not (rel.startswith("ops/") or is_helper_file(inv, rel)):
                continue
            head = normalize_name(name).split("_")[0]
            offences.append(f"{rel}:{d.line} {name} — {head!r} is not a verb")
        assert not offences, (
            f"{len(offences)} function name(s) do not start with a verb "
            f"(docs/NAMES.md):\n" + "\n".join(offences)
            + "\nName it for what it does. If the first word is a verb this "
            "gate has not met, add it to VERBS above.")
