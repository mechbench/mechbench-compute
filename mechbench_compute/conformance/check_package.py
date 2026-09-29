from __future__ import annotations

import ast
import importlib
import inspect
import re
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from mechbench_compute.conformance.check_types import walk_params
from mechbench_compute.conformance.finding import WARNING, Finding
from mechbench_compute.conformance.manifest import Manifest, read_core, read_manifest
from mechbench_compute.conformance.names import KEEP_NAMES, is_verb_first, read_head
from mechbench_compute.lexicon.extension import Extension

NOT_READ_THROUGH_CTX = ("model.sample", "model.backward", "provider.", "network:",
                        "runtime.mlx", "objects.write")

CARRIED_BY_A_MEMBER = {"provider": frozenset({"secrets"})}

KIND_LITERAL = re.compile(r'"kind":\s*"([^"]+)"')


def read_package(target: Any) -> tuple[dict[str, ModuleType], Path, Manifest]:
    if isinstance(target, Extension):
        root = Path(importlib.import_module(target.module).__file__).parent
        return target.modules(), root, read_manifest(target)
    import mechbench_compute

    return target.modules(), Path(mechbench_compute.__file__).parent, read_manifest(read_core())


def read_tree(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        return ast.parse(path.read_text())


def check_module(name: str, mod: ModuleType) -> Iterator[Finding]:
    from mechbench_compute.ops import MEMBER_NEEDS, check_member, read_context_uses

    if not callable(getattr(mod, "run", None)):
        yield Finding("NO_RUN", name, f"{mod.__name__} defines no `run(ctx, inputs, params)`")
        return
    for node in ast.parse(inspect.getsource(mod)).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in KEEP_NAMES or node.name.startswith("__") or is_verb_first(node.name):
            continue
        yield Finding("NAME_NOT_VERB", f"{name}#{node.name}",
                      f"{read_head(node.name)!r} is not a verb; name a function for what it does")
    op = mod.OP
    uses = read_context_uses(mod)
    for member in sorted(uses):
        refused = check_member(op, member)
        if refused is not None:
            yield Finding("NEED_UNDECLARED", name, f"its run reaches for ctx.{member}, and {refused}")
    for need in sorted(op.needs):
        if need.startswith(NOT_READ_THROUGH_CTX):
            continue
        if not any(need in MEMBER_NEEDS[m] | CARRIED_BY_A_MEMBER.get(m, frozenset())
                   for m in uses if m in MEMBER_NEEDS):
            yield Finding("NEED_UNUSED", name, f"declares {need!r} and its run never reaches for it", WARNING)


def list_sources(root: Path, skip: Iterable[str]) -> list[Path]:
    skip = frozenset(skip)
    return [p for p in sorted(root.rglob("*.py"))
            if "__pycache__" not in p.parts and p.name not in skip
            and not skip & set(p.relative_to(root).parts)]


def find_kind_literals(root: Path, m: Manifest, skip: Iterable[str]) -> Iterator[Finding]:
    for path in list_sources(root, skip):
        for hit in KIND_LITERAL.finditer(path.read_text()):
            if m.find_kind(hit.group(1)) is None:
                yield Finding("KIND_LITERAL_UNKNOWN", str(path.relative_to(root)),
                              f"{hit.group(1)!r} is not a declared kind")


def read_compared(test: ast.expr) -> tuple[str, str] | None:
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)):
        return test.left.id, test.comparators[0].value
    return None


def find_refusing_dispatch(body: list[ast.stmt]) -> Iterator[frozenset[str]]:
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.If):
            continue
        chain, node, name = [], stmt, None
        while isinstance(node, ast.If) and (hit := read_compared(node.test)) and (name is None or hit[0] == name):
            name = hit[0]
            chain.append(hit[1])
            nxt = node.orelse
            node = nxt[0] if len(nxt) == 1 and isinstance(nxt[0], ast.If) else nxt  # type: ignore[assignment]
        if len(chain) >= 2 and isinstance(node, list) and any(isinstance(n, ast.Raise) for n in node):
            yield frozenset(chain)
        run, name = [], None
        for later in body[i:]:
            hit = read_compared(later.test) if isinstance(later, ast.If) and not later.orelse else None
            if hit and (name is None or hit[0] == name):
                name = hit[0]
                run.append(hit[1])
                continue
            if len(run) >= 2 and isinstance(later, ast.Raise):
                yield frozenset(run)
            break


def read_string_sets(root: Path, skip: Iterable[str] = ()) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for path in list_sources(root, skip):
        for node in ast.walk(read_tree(path)):
            for field in ("body", "orelse"):
                if isinstance(getattr(node, field, None), list):
                    out.update(find_refusing_dispatch(getattr(node, field)))
            elts: list[ast.expr] = []
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                elts = list(node.elts)
            elif isinstance(node, ast.Dict):
                elts = [k for k in node.keys if k is not None]
            if len(elts) >= 2 and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts):
                out.add(frozenset(e.value for e in elts))  # type: ignore[attr-defined]
    return out


def is_enforced(choices: Iterable[str], sets: set[frozenset[str]]) -> bool:
    c = frozenset(choices)
    if c in sets:
        return True
    parts = [s for s in sets if len(s) >= 3 and s <= c]
    return bool(parts) and frozenset().union(*parts) == c


def check_closed_sets(declared: Iterable[tuple[str, Iterable[str]]],
                      sets: set[frozenset[str]]) -> Iterator[Finding]:
    for at, choices in declared:
        choices = tuple(choices)
        if not is_enforced(choices, sets):
            yield Finding("CLOSED_SET_UNENFORCED", at, f"declares the closed set {choices}, and no code "
                          "enforces it as one")


def check_package(target: Any, *, skip_literals: Iterable[str] = (), skip_sets: Iterable[str] = (),
                  closed_sets: Iterable[tuple[str, Iterable[str]]] = ()) -> list[Finding]:
    modules, root, m = read_package(target)
    findings: list[Finding] = []
    for name, mod in sorted(modules.items()):
        findings += check_module(name, mod)
    findings += find_kind_literals(root, m, skip_literals)
    declared = [(at, p["choices"]) for op in m.ops for at, p in walk_params(op.get("params"), str(op.get("name")))
                if p.get("choices")]
    declared += list(closed_sets)
    findings += check_closed_sets(declared, read_string_sets(root, skip_sets))
    return findings
