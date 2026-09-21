"""Where every definition goes under docs/OPS_LAYOUT.md. Analysis only.

Reads the source, moves nothing. For each operation it gathers the
pieces it is made of today — its declaration, its registry entry, the
executor method or module function that runs it, its monoid — follows
each to the definitions it needs, in its own module or another, and
places every one:

    used by one operation             -> ops/<family>/<name>.py
    used by two or more               -> <topic>/<its own name>.py
    a constant one helper uses        -> that helper's file
    a constant several things use     -> <topic>/constants.py
    a fragment several declarations share, the executor's own
    definitions, anything nothing reaches          -> stays where it is

    python scripts/migrate/plan.py            # the summary
    python scripts/migrate/plan.py --json     # the whole plan
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "mechbench_compute"
sys.path.insert(0, str(ROOT / "tests"))
from test_block_params import SITES  # noqa: E402

REGISTRY_NAMES = {"PURE_BLOCKS", "PURE", "PURE_DIRECTION_BLOCKS", "PURE_REDUCE_BLOCKS",
                  "PURE_TOOL_BLOCKS", "PURE_TREE_BLOCKS", "MONOIDS"}
LEXICON_FILES = ["lexicon/model.py", "lexicon/records.py", "lexicon/external.py",
                 "lexicon/direction.py", "lexicon/trajectory.py"]


def locate(name: str) -> str | None:
    """`interp.py` or `interp` -> the file that module is today: a plain
    module, or a package's `__init__.py` once a helper has left it."""
    stem = name[:-3] if name.endswith(".py") else name
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    if (PKG / f"{stem}.py").is_file():
        return f"{stem}.py"
    if (PKG / stem / "__init__.py").is_file():
        return f"{stem}/__init__.py"
    return None


def topic_of(rel: str) -> str:
    stem = rel[:-3]
    return stem[: -len("/__init__")] if stem.endswith("/__init__") else stem


def snake(name: str) -> str:
    bare = name.lstrip("_")
    if bare.isupper():
        return bare.lower()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", bare).lower()


def op_path(op: str) -> str:
    family, name = op.split("/", 1)
    return f"ops/{family}/{name.replace('-', '_')}.py"


@dataclass
class Def:
    file: str
    name: str
    start: int          # first line, leading comments and decorators included
    end: int
    kind: str           # function | class | assign | declaration | method | registry
    refs: set[str] = field(default_factory=set)
    attr_refs: set[tuple[str, str]] = field(default_factory=set)   # (alias, attribute)
    users: set[str] = field(default_factory=set)   # operations that reach it

    @property
    def lines(self) -> int:
        return self.end - self.start + 1


class Module:
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.src = (PKG / rel).read_text()
        self.text = self.src.splitlines()
        self.tree = ast.parse(self.src)
        self.defs: dict[str, Def] = {}
        self.methods: dict[str, Def] = {}
        self.registry: dict[str, tuple[str, ast.expr]] = {}   # op -> (table, value node)
        self.loose: set[str] = set()      # names the module's own top-level statements use
        self._bindings()
        self._index()

    def _lead(self, lineno: int) -> int:
        """The first line of the comment run sitting directly on top of
        `lineno` — a comment block belongs to the statement under it."""
        k = lineno
        while k - 2 >= 0 and self.text[k - 2].lstrip().startswith("#"):
            k -= 1
        return k

    def _names(self, node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def _attrs(self, node: ast.AST) -> set[tuple[str, str]]:
        return {(n.value.id, n.attr) for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}

    def _bindings(self) -> None:
        """What this file takes from the package's other modules: a name
        imported from one and an alias for one, wherever in the file the
        import sits — the executor imports lazily, inside the method."""
        self.imported: dict[str, tuple[str, str]] = {}
        self.aliases: dict[str, str] = {}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                if n.module == "mechbench_compute":
                    for a in n.names:
                        rel = locate(a.name)
                        if rel:
                            self.aliases[a.asname or a.name] = rel
                elif n.module.startswith("mechbench_compute."):
                    rel = locate(n.module.split(".", 1)[1].replace(".", "/"))
                    if rel:
                        for a in n.names:
                            self.imported[a.asname or a.name] = (rel, a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("mechbench_compute.") and a.asname:
                        rel = locate(a.name.split(".", 1)[1].replace(".", "/"))
                        if rel:
                            self.aliases[a.asname] = rel

    def _index(self) -> None:
        for node in self.tree.body:
            first = node.lineno
            for d in getattr(node, "decorator_list", []):
                first = min(first, d.lineno)
            first = self._lead(first)
            end = node.end_lineno or node.lineno
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = Def(self.rel, node.name, first, end, "function",
                                           self._names(node), self._attrs(node))
            elif isinstance(node, ast.ClassDef):
                self.defs[node.name] = Def(self.rel, node.name, first, end, "class",
                                           self._names(node), self._attrs(node))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        s = self._lead(min([sub.lineno] + [d.lineno for d in sub.decorator_list]))
                        self.methods[sub.name] = Def(self.rel, sub.name, s, sub.end_lineno or sub.lineno,
                                                     "method", self._names(sub), self._attrs(sub))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                placed = False
                for t in targets:
                    if not isinstance(t, ast.Name):
                        continue
                    value = node.value
                    if t.id in REGISTRY_NAMES and isinstance(value, ast.Dict):
                        for k, v in zip(value.keys, value.values):
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                self.registry[k.value] = (t.id, v)
                        placed = True
                        continue
                    kind = "assign"
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                            and value.func.id == "Op"):
                        kind = "declaration"
                    self.defs[t.id] = Def(self.rel, t.id, first, end, kind,
                                          self._names(value) if value is not None else set(),
                                          self._attrs(value) if value is not None else set())
                    placed = True
                if not placed:
                    self.loose |= self._names(node)
            elif not isinstance(node, (ast.Import, ast.ImportFrom)):
                self.loose |= self._names(node)

    @property
    def hosts(self) -> set[str]:
        """Classes whose methods are operations' entry points: the
        executor. It runs operations and is not part of one."""
        return {node.name for node in self.tree.body if isinstance(node, ast.ClassDef)
                and any(isinstance(sub, ast.FunctionDef) and sub.name.startswith("_block_")
                        for sub in node.body)}


def declared_name(mod: Module, d: Def) -> str | None:
    for node in mod.tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == d.name for t in node.targets):
            for kw in node.value.keywords:  # type: ignore[union-attr]
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


@dataclass
class Analysis:
    mods: dict[str, Module]
    pieces: dict[str, list[Def]]           # op -> its roots
    placement: dict[str, list[Def]]        # destination -> what lands there
    stays: list[Def]                       # reached, and deliberately left where it is
    problems: list[str]

    def dest_of(self) -> dict[tuple[str, str], str]:
        return {(d.file, d.name): dest for dest, ds in self.placement.items() for d in ds}


def analyse() -> Analysis:
    named = {f for sites in SITES.values() for f, _ in sites} | set(LEXICON_FILES) | {"blocks.py", "reduce.py"}
    files = sorted({rel for rel in (locate(f) for f in named) if rel})
    mods = {f: Module(f) for f in files}

    declarations: dict[str, Def] = {}
    for f in LEXICON_FILES:
        for d in mods[f].defs.values():
            if d.kind == "declaration":
                name = declared_name(mods[f], d)
                if name:
                    declarations[name] = d

    pieces: dict[str, list[Def]] = defaultdict(list)
    problems: list[str] = []
    for op in sorted(SITES):
        if (PKG / op_path(op)).exists():
            continue                        # already moved
        if op in declarations:
            pieces[op].append(declarations[op])
        else:
            problems.append(f"{op}: no declaration found")
        for file, fn in SITES[op]:
            m = mods.get(locate(file) or "")
            if m is None:
                continue
            if fn in m.methods:
                pieces[op].append(m.methods[fn])
            elif fn in m.defs:
                pieces[op].append(m.defs[fn])
            else:
                problems.append(f"{op}: site {file}:{fn} not found")
        for m in mods.values():
            if op in m.registry:
                table, value = m.registry[op]
                pieces[op].append(Def(m.rel, f"<{table}[{op!r}]>", value.lineno, value.end_lineno or value.lineno,
                                      "registry", {n.id for n in ast.walk(value) if isinstance(n, ast.Name)},
                                      {(n.value.id, n.attr) for n in ast.walk(value)
                                       if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}))

    def edges(d: Def) -> list[Def]:
        """The definitions `d` needs, in its own file or in another of
        the files being taken apart."""
        m = mods[d.file]
        out: list[Def] = []
        for r in d.refs:
            if r in m.defs and r not in m.hosts:
                out.append(m.defs[r])
            elif r in m.imported:
                rel, orig = m.imported[r]
                if rel in mods and orig in mods[rel].defs and orig not in mods[rel].hosts:
                    out.append(mods[rel].defs[orig])
        for alias, attr in d.attr_refs:
            rel = m.aliases.get(alias)
            if rel in mods and attr in mods[rel].defs and attr not in mods[rel].hosts:
                out.append(mods[rel].defs[attr])
        return out

    for op, roots in pieces.items():
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if cur.kind not in ("method", "registry"):
                cur.users.add(op)
            stack.extend(edges(cur))

    # What code that is not going anywhere still needs: definitions
    # nothing reaches, the executor's own methods, and a module's
    # top-level statements. A definition they use cannot go into one
    # operation's file, because nothing may import from there back.
    op_methods = {(r.file, r.name) for roots in pieces.values() for r in roots if r.kind == "method"}
    needed_by_stayers: dict[str, set[str]] = {}
    for f, m in mods.items():
        need = set(m.loose)
        for d in m.defs.values():
            if not d.users:
                need |= d.refs
        for name, meth in m.methods.items():
            if (f, name) not in op_methods:
                need |= meth.refs
        needed_by_stayers[f] = need

    placement: dict[str, list[Def]] = defaultdict(list)
    stays: list[Def] = []
    for op, roots in pieces.items():
        for root in roots:
            if root.kind in ("method", "registry"):
                placement[op_path(op)].append(root)

    helper_home: dict[tuple[str, str], str] = {}
    for f, m in mods.items():
        executor = bool(m.hosts)
        for d in m.defs.values():
            if not d.users:
                continue
            pinned = d.name in needed_by_stayers[f]
            alone = len(d.users) == 1 and not pinned
            if d.kind == "declaration":
                placement[op_path(next(iter(d.users)))].append(d)
            elif f.startswith("lexicon/") or executor:
                # A fragment several declarations share is inlined later
                # (000631); the executor's own definitions are borrowed.
                if alone:
                    placement[op_path(next(iter(d.users)))].append(d)
                else:
                    stays.append(d)
            elif alone:
                placement[op_path(next(iter(d.users)))].append(d)
            elif d.kind != "assign":
                helper_home[(f, d.name)] = f"{topic_of(f)}/{snake(d.name)}.py"
    for f, m in mods.items():
        if f.startswith("lexicon/") or m.hosts:
            continue
        for d in m.defs.values():
            if not d.users or d.kind != "assign":
                continue
            if len(d.users) == 1 and d.name not in needed_by_stayers[f]:
                continue                    # already placed with its one operation
            # A constant rides with the one helper that uses it.
            riders = [o for o in m.defs.values() if o is not d and d.name in o.refs]
            if len(riders) == 1 and (f, riders[0].name) in helper_home:
                helper_home[(f, d.name)] = helper_home[(f, riders[0].name)]
            else:
                helper_home[(f, d.name)] = f"{topic_of(f)}/constants.py"
    for (f, name), dest in helper_home.items():
        placement[dest].append(mods[f].defs[name])

    taken: dict[str, list[str]] = defaultdict(list)
    for (f, name), dest in helper_home.items():
        if mods[f].defs[name].kind != "assign":
            taken[dest].append(name)
    for dest, names in taken.items():
        # `Plan` and `plan` both come to `plan.py`: a class and the
        # function that builds it, which belong in one file anyway. Two
        # unrelated names colliding would be a problem; say which.
        if len(names) > 1 and len({n.lower().lstrip("_") for n in names}) > 1:
            problems.append(f"{dest}: two definitions want this path — {sorted(names)}")
    return Analysis(mods, pieces, placement, stays, problems)


def main() -> None:
    a = analyse()
    mods, placement, problems = a.mods, a.placement, a.problems
    if "--json" in sys.argv:
        print(json.dumps({dest: [{"file": d.file, "name": d.name, "start": d.start, "end": d.end, "kind": d.kind,
                                  "users": sorted(d.users)} for d in sorted(ds, key=lambda d: (d.file, d.start))]
                          for dest, ds in sorted(placement.items())}, indent=1))
        return

    sizes = {dest: sum(d.lines for d in ds) for dest, ds in placement.items()}
    op_files = sorted(v for k, v in sizes.items() if k.startswith("ops/"))
    helpers = sorted(v for k, v in sizes.items() if not k.startswith("ops/"))
    print(f"{len(op_files):3} operation files   median {op_files[len(op_files)//2]:4} lines   largest {op_files[-1]}")
    if helpers:
        print(f"{len(helpers):3} helper files      median {helpers[len(helpers)//2]:4} lines   largest {helpers[-1]}")
    print("\nlargest files the move would make:")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {v:5}  {k}")
    print(f"\nstays where it is, though operations use it: {len(a.stays)} definitions, {sum(d.lines for d in a.stays)} lines")
    by_file = defaultdict(list)
    for d in a.stays:
        by_file[d.file].append(d.name)
    for f, names in sorted(by_file.items()):
        print(f"   {f:24} {', '.join(sorted(names)[:8])}{' …' if len(names) > 8 else ''}")
    print("\nwhat each source file shrinks to:")
    for f, m in sorted(mods.items()):
        gone = sum(d.lines for ds in placement.values() for d in ds if d.file == f)
        print(f"   {f:26} {len(m.text):5} -> {len(m.text) - gone:5}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  ", p)


if __name__ == "__main__":
    main()
