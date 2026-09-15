"""The lexicon is the operation reference: every canonical op described,
every parameter typed and explained, in language a stranger can read.

`test_block_params.py` proves the declared NAMES equal what the code
reads. This file proves the rest of the declaration is fit to publish:
nothing is blank, every example would pass `check_params`, and no entry
leans on a house idiom — a task number, an experiment step, an epic —
that resolves to nothing outside this repository.
"""

from __future__ import annotations

import re

import pytest

from mechbench_compute import lexicon
from mechbench_compute.block_params import ACCEPTED, COMMON, check_inputs, check_params
from mechbench_compute.lexicon import BY_NAME, OPS, Op

# What must never appear in published text. A bare six-digit id, "task
# 000123", "epic 000364", "step 07", "experiment 014", "000258 am. 4":
# every one of these is an argument to us and an opaque string to a
# reader of the documentation site.
INTERNAL = [
    re.compile(r"\b0\d{5}\b"),
    re.compile(r"\b(?:task|tasks|epic|epics)\s+\d", re.I),
    re.compile(r"\bstep\s+\d{2}\b", re.I),
    re.compile(r"\bexperiment\s+0\d{2}\b", re.I),
    re.compile(r"\bmechbench-experiments\b", re.I),
]


def _texts(op: Op) -> list[tuple[str, str]]:
    out = [("summary", op.summary), ("description", op.description),
           ("emits", op.emits.doc if op.emits else "")]
    out += [(f"port {p.name}.doc", p.doc) for p in op.inputs]
    out += [(f"param {p.name}.doc", p.doc) for p in op.params]
    out += [(f"param {p.name}.type", p.type) for p in op.params]
    return out


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_entry_is_complete(op: Op) -> None:
    assert op.summary.strip(), f"{op.name}: no summary"
    assert op.description.strip(), f"{op.name}: no description"
    if op.emits is not None:
        assert op.emits.doc.strip(), f"{op.name}: says nothing about what it emits"
    else:
        assert op.family == "tools", f"{op.name}: emits nothing and is not a tool"
    for p in op.params:
        assert p.type.strip(), f"{op.name}.{p.name}: no type"
        assert p.doc.strip(), f"{op.name}.{p.name}: no description"
    names = [p.name for p in op.params]
    assert len(names) == len(set(names)), f"{op.name}: duplicate params"
    ports = [p.name for p in op.inputs]
    assert len(ports) == len(set(ports)), f"{op.name}: duplicate ports"
    for p in op.inputs:
        assert p.doc.strip(), f"{op.name} port {p.name}: no description"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_every_port_names_a_declared_kind(op: Op) -> None:
    """A port's kind is a name in the kinds registry, so the docs can link
    it and `check_inputs` can walk its ancestry."""
    for p in op.inputs:
        for k in p.kinds:
            assert k in lexicon.BY_KIND, f"{op.name} port {p.name}: unknown kind {k!r}"
        assert p.name == lexicon.WILDCARD or re.fullmatch(r"[a-z][a-z0-9_]*", p.name), (
            f"{op.name}: port name {p.name!r}")


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_no_param_names_a_port(op: Op) -> None:
    """An input is a port, never a param: the two would be the same value
    spelled twice, and `check_params` could not tell an unwired port from
    a typo. The sentence "by edge, or the param" is the smell."""
    assert not (op.param_names & op.port_names), (
        f"{op.name}: {sorted(op.param_names & op.port_names)} declared as both")
    for where, text in _texts(op):
        assert not re.search(r"by edge,? or (the|by) (the )?param", text, re.I), (
            f"{op.name} {where}: an input is described as a param")
    for retired in ("user_field", "system_field", "prefill_field", "answer_field",
                    "prediction_field", "reference_field", "messages_field",
                    "label_field", "label_coord", "pairwise_fields", "collection_path"):
        assert retired not in op.param_names, f"{op.name} still declares {retired}"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_summary_is_one_sentence_for_a_stranger(op: Op) -> None:
    s = op.summary.strip()
    # One sentence: ends once, and does not restate the block ref the
    # page already shows.
    assert not s.startswith("~canonical/"), f"{op.name}: summary restates the ref"
    assert s[-1] in ".?!", f"{op.name}: summary should end as a sentence does"
    assert len(s) < 400, f"{op.name}: summary is a paragraph, not a sentence"


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_no_house_idioms_in_published_text(op: Op) -> None:
    for where, text in _texts(op):
        for pat in INTERNAL:
            m = pat.search(text)
            assert m is None, (
                f"{op.name} {where}: {m.group(0)!r} is a reference nobody "
                f"outside this repo can follow — rewrite it as what it means")


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_example_would_be_accepted(op: Op) -> None:
    if op.example is None:
        pytest.skip("no example written")
    # The example must not name a param the block refuses — the exact
    # mistake the documentation exists to prevent — and its inputs must
    # land on ports the block has. A `{"$fetch": …}` carries no kind
    # until it resolves, so the example's inputs are checked by port
    # name, with every required port present.
    check_params(op.name, op.example)
    inputs = dict(op.example_inputs or {})
    for p in op.inputs:
        if p.required and not p.wildcard and p.name not in inputs:
            assert p.name in inputs, f"{op.name}: example wires no {p.name!r}, which is required"
    check_inputs(op.name, inputs)


def test_common_params_are_documented() -> None:
    for p in lexicon.COMMON:
        assert p.type.strip() and p.doc.strip(), f"common param {p.name} undocumented"
        for pat in INTERNAL:
            assert not pat.search(p.doc), f"common param {p.name}: house idiom in doc"
    assert COMMON == frozenset(p.name for p in lexicon.COMMON)


def test_block_params_is_derived_from_the_lexicon() -> None:
    """One source. `block_params` must not carry its own table."""
    assert set(ACCEPTED) == set(BY_NAME)
    for name, op in BY_NAME.items():
        assert ACCEPTED[name] == op.param_names
    # A param cannot be both common and op-specific: the table would
    # accept it twice and the docs would describe it twice.
    for op in OPS:
        assert not (op.param_names & COMMON), f"{op.name} redeclares a common param"


def test_names_are_two_level_bare_and_unique() -> None:
    """docs/LEXICON.md §1: `family/op`, exactly two levels, no root, no
    version segment; §2: every family has at least two members."""
    names = [op.name for op in OPS]
    assert len(names) == len(set(names)), "two ops share a name"
    for name in names:
        assert re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", name), name
        assert not name.endswith(tuple(f"/{d}" for d in "0123456789")), name
    families: dict[str, int] = {}
    for op in OPS:
        families[op.family] = families.get(op.family, 0) + 1
    lonely = sorted(f for f, n in families.items() if n < 2)
    assert not lonely, f"families of one: {lonely}"
    assert set(families) == {
        "records", "text", "eval", "logits", "activations", "geometry",
        "intervene", "direction", "trajectory", "adapter", "tools",
    }


def test_to_dict_is_json_shaped() -> None:
    import json

    for op in OPS:
        json.dumps(op.to_dict())  # REQUIRED never leaks; defaults are JSON values
        for p in op.params:
            d = p.to_dict()
            assert ("default" in d) != d["required"]


def _publishable(where: str, text: str) -> None:
    for pat in INTERNAL:
        m = pat.search(text)
        assert m is None, f"{where}: {m.group(0)!r} is a reference nobody outside this repo can follow"


def test_every_family_is_declared_and_described() -> None:
    """The families are the namespaces of both vocabularies: each one an
    op or a kind belongs to is declared with a summary and a doc, and
    nothing is declared that has no members."""
    from mechbench_compute.lexicon import kinds as K

    used = {op.family for op in OPS} | {k.family for k in K.KINDS if k.name != K.COLLECTION}
    declared = {f.name for f in lexicon.FAMILIES}
    assert used == declared, f"undeclared: {sorted(used - declared)}; empty: {sorted(declared - used)}"
    for f in lexicon.FAMILIES:
        s = f.summary.strip()
        assert s and s[-1] in ".?!" and len(s) < 400, f"family {f.name}: summary"
        assert f.doc.strip(), f"family {f.name}: no doc"
        _publishable(f"family {f.name}", f.summary + "\n" + f.doc)
        assert lexicon.BY_FAMILY[f.name] is f
        import json
        json.dumps(f.to_dict())


def test_every_value_type_is_described() -> None:
    """A value type is declared with its summary, its doc, and typed,
    described fields; the point vocabulary's page lists exactly the
    points the code accepts."""
    import json

    from mechbench_compute import points
    from mechbench_compute.lexicon import values as V

    for v in lexicon.VALUES:
        s = v.summary.strip()
        assert s and s[-1] in ".?!" and len(s) < 400, f"value {v.name}: summary"
        assert v.doc.strip(), f"value {v.name}: no doc"
        assert re.fullmatch(r"[a-z]+", v.name), v.name
        for f, spec in v.fields.items():
            assert spec.get("type") and spec.get("description"), f"value {v.name}.{f}: type and description"
        for r in v.required:
            assert r in v.fields, f"value {v.name}: required {r!r} is not a field"
        _publishable(f"value {v.name}", "\n".join([v.summary, v.doc, *(f["description"] for f in v.fields.values())]))
        json.dumps(v.to_dict())
    assert V.DOCUMENTED_POINTS == frozenset(points.POINTS)
    for name in points.POINTS:
        assert f"`{name}`" in V.BY_VALUE["point"].doc, f"point {name} is not on the page"
    assert {v.name for v in lexicon.VALUES if v.grammar} == {"position", "pool", "point"}
