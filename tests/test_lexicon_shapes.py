"""A parameter's type is a declaration an editor can be built from.

The protocol composer renders every node's editor from the lexicon: a
number field for an `int`, a menu for a closed set, a form for an
object. What it cannot render from the declaration it can only show as
raw JSON, and before these declarations it did that for twenty-six of
the fifty-eight operations — `trajectory/capture` among them, whose
`pool` is two fields and whose `axis` is one of two words.

So this proves, for every parameter of every operation:

* its `type` is in the grammar (`lexicon._base.parse_type`);
* every `object` in it is declared — by the param's own `fields`, or by
  the shared `value` it names. An open structure says so in its type,
  as `json`;
* every closed set (`choices`) is one the code itself spells out, so the
  menu an editor offers is the set the executor accepts — no more,
  which would pass a value the run refuses, and no fewer, which would
  forbid one it takes;
* every example conforms to the declared types, which is the check that
  the declarations describe what protocols really write.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator
from typing import Any

import pytest

from mechbench_compute import lexicon
from mechbench_compute.lexicon import OPS, Op
from mechbench_compute.lexicon._base import REQUIRED, Param, TypeNode, parse_type, type_words
from mechbench_compute.lexicon.values import BY_VALUE

SRC = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute"


def _walk(params: tuple[Param, ...], where: str) -> Iterator[tuple[str, Param]]:
    for p in params:
        yield f"{where}.{p.name}", p
        yield from _walk(p.fields, f"{where}.{p.name}")


def _all_params() -> list[tuple[str, Param]]:
    out = list(_walk(lexicon.COMMON, "common"))
    for op in OPS:
        out += list(_walk(op.params, op.name))
    return out


ALL = _all_params()


@pytest.mark.parametrize(("where", "p"), ALL, ids=[w for w, _ in ALL])
def test_the_type_is_in_the_grammar(where: str, p: Param) -> None:
    parse_type(p.type)


@pytest.mark.parametrize(("where", "p"), ALL, ids=[w for w, _ in ALL])
def test_every_object_is_declared(where: str, p: Param) -> None:
    words = type_words(p.type)
    shape = BY_VALUE[p.value] if p.value else None
    if "object" in words:
        assert p.fields or (shape and shape.fields), (
            f"{where} is an `object` with no declared fields: declare them "
            f"(`fields=`), name the shared value it is (`value=`), or — if it "
            f"is open on purpose — type it `json` so a reader can see that")
    else:
        assert not p.fields, f"{where} declares fields but its type has no `object`"
    if p.fields:
        names = [f.name for f in p.fields]
        assert len(names) == len(set(names)), f"{where}: a field is declared twice"


@pytest.mark.parametrize(("where", "p"), ALL, ids=[w for w, _ in ALL])
def test_value_and_choices_are_well_formed(where: str, p: Param) -> None:
    words = type_words(p.type)
    if p.value:
        assert p.value in BY_VALUE, f"{where}: value {p.value!r} is not a declared value"
        v = BY_VALUE[p.value]
        assert v.grammar, f"{where}: {p.value} is a stored field's shape, not a parameter grammar"
        if v.choices:
            assert "string" in words, f"{where}: {p.value} is a vocabulary; the type has no string"
            assert set(p.choices) <= set(v.choices), (
                f"{where}: choices {sorted(set(p.choices) - set(v.choices))} "
                f"are not {p.value} words")
    if p.choices:
        assert "string" in words, f"{where}: choices on a type with no `string`"
        assert len(p.choices) == len(set(p.choices)), f"{where}: a choice is listed twice"
        allowed = set(p.choices)
        d = p.default
        if isinstance(d, str):
            assert d in allowed, f"{where}: default {d!r} is not one of the choices"
        if isinstance(d, list):
            assert all(x in allowed for x in d if isinstance(x, str)), (
                f"{where}: default {d!r} names a word outside the choices")


def _compared(test: ast.expr) -> tuple[str, str] | None:
    """`kind == "noise"` as `("kind", "noise")`."""
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)):
        return test.left.id, test.comparators[0].value
    return None


def _refusing_dispatch(body: list[ast.stmt]) -> Iterator[frozenset[str]]:
    """The sets a dispatch on one name refuses the rest of: an
    `if x == "a" … elif x == "b" … else: raise`, or a run of
    `if x == "a": return …` statements followed by a `raise`."""
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.If):
            continue
        chain, node, name = [], stmt, None
        while isinstance(node, ast.If) and (hit := _compared(node.test)) and (name is None or hit[0] == name):
            name = hit[0]
            chain.append(hit[1])
            nxt = node.orelse
            node = nxt[0] if len(nxt) == 1 and isinstance(nxt[0], ast.If) else nxt  # type: ignore[assignment]
        if len(chain) >= 2 and isinstance(node, list) and any(isinstance(n, ast.Raise) for n in node):
            yield frozenset(chain)
        run, name = [], None
        for later in body[i:]:
            hit = _compared(later.test) if isinstance(later, ast.If) and not later.orelse else None
            if hit and (name is None or hit[0] == name):
                name = hit[0]
                run.append(hit[1])
                continue
            if len(run) >= 2 and isinstance(later, ast.Raise):
                yield frozenset(run)
            break


def _string_sets() -> set[frozenset[str]]:
    """Every closed set of strings the code spells out: a tuple, list or
    set literal of string constants, a dict literal's keys, or the
    branches of a dispatch that refuses anything else."""
    out: set[frozenset[str]] = set()
    for path in SRC.rglob("*.py"):
        if "lexicon" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            for field in ("body", "orelse"):
                if isinstance(getattr(node, field, None), list):
                    out.update(_refusing_dispatch(getattr(node, field)))
            elts: list[ast.expr] = []
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                elts = list(node.elts)
            elif isinstance(node, ast.Dict):
                elts = [k for k in node.keys if k is not None]
            if len(elts) >= 2 and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                                      for e in elts):
                out.add(frozenset(e.value for e in elts))  # type: ignore[attr-defined]
    return out


def test_every_closed_set_is_the_codes_own() -> None:
    """A declared set must appear in the executor, verbatim as a set —
    the refusal (`if mode not in ("annotate", "corpus")`), or the
    constant it reads (`AXES`, `REDUCES`, `LEVELS`). Found by scanning
    the source rather than listed by hand, so a new closed param proves
    itself without anyone updating this file; a set the code never
    enforces fails here, which is the prompt to add the refusal.

    One field may take the union of two closed sets the code keeps apart:
    an intervention item's `op` is an activation op or a weight op,
    checked by different code. A union counts only of sets of three or
    more, so a small declared set still has to be spelled out whole."""
    sets = _string_sets()

    def enforced(c: frozenset[str]) -> bool:
        if c in sets:
            return True
        parts = [s for s in sets if len(s) >= 3 and s <= c]
        return bool(parts) and frozenset().union(*parts) == c
    declared: list[tuple[str, tuple[str, ...]]] = [(w, p.choices) for w, p in ALL if p.choices]
    for v in lexicon.VALUES:
        for name, spec in v.fields.items():
            if spec.get("choices"):
                declared.append((f"value {v.name}.{name}", tuple(spec["choices"])))
    missing = [(w, c) for w, c in declared if not enforced(frozenset(c))]
    assert not missing, (
        "declared closed sets the code does not enforce as such:\n"
        + "\n".join(f"  {w}: {c}" for w, c in missing))


# --- examples conform to the declared types ---------------------------------------


def _binding(v: Any) -> bool:
    """A value that resolves at run time: `"$model"`, `{"$fetch": …}`."""
    if isinstance(v, str):
        return v.startswith("$")
    return isinstance(v, dict) and len(v) == 1 and str(next(iter(v))).startswith("$")


def _fields(p: Param) -> list[Param]:
    if p.fields:
        return list(p.fields)
    if p.value and BY_VALUE[p.value].fields:
        v = BY_VALUE[p.value]
        return [Param(n, s["type"], s["description"],
                      REQUIRED if n in v.required else s.get("default"),
                      tuple(s.get("choices", ())))
                for n, s in v.fields.items()]
    return []


def _choices(p: Param) -> tuple[str, ...]:
    if p.choices:
        return p.choices
    return BY_VALUE[p.value].choices if p.value else ()


def _selector(v: Any) -> bool:
    if isinstance(v, str):
        return True
    if isinstance(v, list):
        return all(isinstance(i, int) and not isinstance(i, bool) for i in v)
    return isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("tokens", "range", "after")


def _conforms(v: Any, alts: tuple[TypeNode, ...], p: Param, where: str) -> list[str]:
    """Why `v` is not a `p.type`, or [] when it is (one alternative fits)."""
    if _binding(v):
        return []
    reasons: list[str] = []
    for a in alts:
        why = _fits(v, a, p, where)
        if not why:
            return []
        reasons += why
    return reasons or [f"{where}: {v!r} is not {p.type}"]


def _fits(v: Any, a: TypeNode, p: Param, where: str) -> list[str]:
    no = [f"{where}: {v!r} is not {p.type}"]
    if a.form == "literal":
        return [] if v == a.word else no
    if a.form == "list":
        if not isinstance(v, list):
            return no
        return [r for i, x in enumerate(v) for r in _conforms(x, a.of, p, f"{where}[{i}]")]
    if a.form == "map":
        if not isinstance(v, dict):
            return no
        return [r for k, x in v.items() for r in _conforms(x, a.of, p, f"{where}.{k}")]
    w = a.word
    if w == "string":
        if not isinstance(v, str):
            return no
        c = _choices(p)
        return [] if not c or v in c else [f"{where}: {v!r} is not one of {c}"]
    if w == "int":
        return [] if isinstance(v, int) and not isinstance(v, bool) else no
    if w == "float":
        return [] if isinstance(v, (int, float)) and not isinstance(v, bool) else no
    if w == "bool":
        return [] if isinstance(v, bool) else no
    if w == "null":
        return [] if v is None else no
    if w == "json":
        return []
    if w == "model":
        return [] if isinstance(v, (str, dict)) else no
    if w == "selector":
        return [] if _selector(v) else no
    if w == "object":
        if not isinstance(v, dict):
            return no
        fs = {f.name: f for f in _fields(p)}
        out = [f"{where}: {k!r} is not a declared field" for k in v if k not in fs]
        out += [f"{where}: required field {n!r} missing" for n, f in fs.items()
                if f.required and n not in v]
        for k, x in v.items():
            if k in fs:
                out += _conforms(x, parse_type(fs[k].type), fs[k], f"{where}.{k}")
        return out
    return no


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_the_example_conforms_to_the_declared_types(op: Op) -> None:
    if op.example is None:
        pytest.skip("no example written")
    by = {p.name: p for p in (*op.params, *lexicon.COMMON)}
    problems = [r for k, v in op.example.items()
                for r in _conforms(v, parse_type(by[k].type), by[k], f"{op.name}.{k}")]
    assert not problems, "\n".join(problems)


def test_shapes_reach_the_published_dict() -> None:
    import json

    capture = lexicon.BY_NAME["trajectory/capture"].to_dict()
    by = {p["name"]: p for p in capture["params"]}
    assert by["axis"]["choices"] == ["layers", "positions"]
    assert by["pool"]["value"] == "pool"
    lora = {p["name"]: p for p in lexicon.BY_NAME["adapter/train"].to_dict()["params"]}["lora"]
    assert [f["name"] for f in lora["fields"]][:2] == ["rank", "alpha"]
    json.dumps([op.to_dict() for op in OPS])


@pytest.mark.parametrize("op", [op for op in OPS if op.output and op.output.otherwise], ids=lambda op: op.name)
def test_what_an_op_emits_instead_is_declared_against_the_node(op: Op) -> None:
    """An `otherwise` names a declared kind and a condition on the node a
    composer can read — a param it has, with a value that param takes,
    or a port it has — and the emitted record's prose says so, so the
    page and the declaration cannot disagree."""
    from mechbench_compute.lexicon.kinds import BY_KIND

    assert op.output is not None
    for o in op.output.otherwise:
        assert o.kind in BY_KIND, f"{op.name}: {o.kind} is not a declared kind"
        assert (o.param is None) != (o.port is None), f"{op.name}: name a param or a port, not both"
        if o.param is not None:
            p = next((p for p in op.params if p.name == o.param), None)
            assert p is not None, f"{op.name}: no param {o.param!r}"
            if p.choices:
                assert o.equals in p.choices, f"{op.name}: {o.equals!r} is not a {o.param} choice"
        else:
            assert op.port(o.port or "") is not None and o.port in op.port_names, (
                f"{op.name}: no port {o.port!r}")
        assert f"`{o.kind}`" in op.output.doc, f"{op.name}: the output prose does not name `{o.kind}`"
