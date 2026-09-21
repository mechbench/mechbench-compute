"""Where every definition goes under docs/OPS_LAYOUT.md. Analysis only.

Reads the source, moves nothing. For each operation it gathers the
pieces it is made of today — its declaration, its registry entry, the
executor method or module function that runs it, its monoid — follows
each to the same-module definitions it needs, and applies the placement
rule to the result:

    used by one operation            -> that operation's file
    shared within one family         -> ops/<family>/_common.py
    shared across families           -> ops/_common.py

    python scripts/migrate/plan.py            # the summary
    python scripts/migrate/plan.py --json     # the whole plan, for move.py
"""
from __future__ import annotations

import ast
import json
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


def op_path(op: str) -> str:
    family, name = op.split("/", 1)
    return f"ops/{family}/{name.replace('-', '_')}.py"


@dataclass
class Def:
    file: str
    name: str
    start: int          # first line, leading comments and decorators included
    end: int
    kind: str           # function | class | assign | method | registry | declaration
    refs: set[str] = field(default_factory=set)
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

    def _index(self) -> None:
        for node in self.tree.body:
            first = node.lineno
            for d in getattr(node, "decorator_list", []):
                first = min(first, d.lineno)
            first = self._lead(first)
            end = node.end_lineno or node.lineno
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = Def(self.rel, node.name, first, end, "function", self._names(node))
            elif isinstance(node, ast.ClassDef):
                self.defs[node.name] = Def(self.rel, node.name, first, end, "class", self._names(node))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        s = self._lead(min([sub.lineno] + [d.lineno for d in sub.decorator_list]))
                        self.methods[sub.name] = Def(self.rel, sub.name, s, sub.end_lineno or sub.lineno,
                                                     "method", self._names(sub))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if not isinstance(t, ast.Name):
                        continue
                    value = node.value
                    if t.id in REGISTRY_NAMES and isinstance(value, ast.Dict):
                        for k, v in zip(value.keys, value.values):
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                self.registry[k.value] = (t.id, v)
                        continue
                    kind = "assign"
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                            and value.func.id == "Op"):
                        kind = "declaration"
                    self.defs[t.id] = Def(self.rel, t.id, first, end, kind,
                                          self._names(value) if value is not None else set())

    @property
    def hosts(self) -> set[str]:
        """Classes whose methods are operations' entry points: the
        executor. It is what runs operations, not part of one, so a
        reference to it is a reference to something that stays."""
        return {node.name for node in self.tree.body if isinstance(node, ast.ClassDef)
                and any(isinstance(sub, ast.FunctionDef) and sub.name in self.methods
                        and sub.name.startswith("_block_") for sub in node.body)}

    def closure(self, seeds: set[str]) -> set[str]:
        seen: set[str] = set()
        hosts = self.hosts
        seeds = {s for s in seeds if s not in hosts}
        stack = [s for s in seeds if s in self.defs]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(r for r in self.defs[cur].refs
                         if r in self.defs and r not in seen and r not in hosts)
        return seen


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
    problems: list[str]

    def dest_of(self) -> dict[tuple[str, str], str]:
        return {(d.file, d.name): dest for dest, ds in self.placement.items() for d in ds}


def analyse() -> Analysis:
    files = sorted({f for sites in SITES.values() for f, _ in sites if (PKG / f).exists()}
                   | set(LEXICON_FILES) | {"blocks.py", "reduce.py"})
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
            m = mods.get(file)
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
                names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
                pieces[op].append(Def(m.rel, f"<{table}[{op!r}]>", value.lineno, value.end_lineno or value.lineno,
                                      "registry", names))

    for op, roots in pieces.items():
        for root in roots:
            m = mods[root.file]
            seeds = {root.name} if root.name in m.defs else set()
            seeds |= {r for r in root.refs if r in m.defs}
            for name in m.closure(seeds):
                m.defs[name].users.add(op)

    placement: dict[str, list[Def]] = defaultdict(list)
    for op, roots in pieces.items():
        for root in roots:
            if root.kind in ("method", "registry"):
                placement[op_path(op)].append(root)
    for m in mods.values():
        for d in m.defs.values():
            if not d.users:
                continue
            families = {u.split("/")[0] for u in d.users}
            # What several declarations share and what several mechanisms
            # share have different readers, so they are different files.
            shared = "_params.py" if d.file.startswith("lexicon/") else "_common.py"
            if len(d.users) == 1:
                dest = op_path(next(iter(d.users)))
            elif len(families) == 1:
                dest = f"ops/{next(iter(families))}/{shared}"
            else:
                dest = f"ops/{shared}"
            placement[dest].append(d)
    return Analysis(mods, pieces, placement, problems)


def main() -> None:
    a = analyse()
    mods, placement, problems = a.mods, a.placement, a.problems

    if "--json" in sys.argv:
        print(json.dumps({dest: [{"file": d.file, "name": d.name, "start": d.start, "end": d.end, "kind": d.kind,
                                  "users": sorted(d.users)} for d in sorted(ds, key=lambda d: (d.file, d.start))]
                          for dest, ds in sorted(placement.items())}, indent=1))
        return

    sizes = {dest: sum(d.lines for d in ds) for dest, ds in placement.items()}
    op_files = {k: v for k, v in sizes.items() if not k.rsplit("/", 1)[-1].startswith("_")}
    commons = {k: v for k, v in sizes.items() if k.rsplit("/", 1)[-1].startswith("_")}
    ordered = sorted(op_files.values())
    print(f"{len(op_files)} operation files   median {ordered[len(ordered)//2]} lines   "
          f"largest {ordered[-1]}   total {sum(ordered)}")
    print("\nlargest operation files:")
    for k, v in sorted(op_files.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {v:5}  {k}")
    print("\nshared modules:")
    for k, v in sorted(commons.items(), key=lambda kv: -kv[1]):
        print(f"   {v:5}  {k:32} {len(placement[k]):3} definitions")
    print("\nwhat stays behind in each source file (definitions no operation reaches):")
    for f, m in sorted(mods.items()):
        moved = sum(d.lines for d in m.defs.values() if d.users)
        moved += sum(d.lines for ds in placement.values() for d in ds if d.file == f and d.kind in ("method", "registry"))
        stay = [d for d in m.defs.values() if not d.users]
        print(f"   {f:26} {len(m.text):5} lines   {moved:5} move   {len(stay):3} definitions stay "
              f"({sum(d.lines for d in stay)} lines)")
    print("\nmoved definitions that code staying behind also uses (it will import them from their new home):")
    total = 0
    for f, m in sorted(mods.items()):
        op_methods = {d.name for ds in placement.values() for d in ds if d.file == f and d.kind == "method"}
        stay_refs: set[str] = set()
        for d in m.defs.values():
            if not d.users:
                stay_refs |= d.refs
        for name, meth in m.methods.items():
            if name not in op_methods:
                stay_refs |= meth.refs
        both = sorted(d.name for d in m.defs.values() if d.users and d.name in stay_refs)
        total += len(both)
        if both:
            print(f"   {f:22} {len(both):3}  {', '.join(both[:7])}{' …' if len(both) > 7 else ''}")
    print(f"   ({total} in all)")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  ", p)


if __name__ == "__main__":
    main()
