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
from mechbench_compute.block_params import ACCEPTED, COMMON, check_params
from mechbench_compute.lexicon import BY_REF, OPS, Op

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
           ("inputs", op.inputs), ("emits", op.emits)]
    out += [(f"param {p.name}.doc", p.doc) for p in op.params]
    out += [(f"param {p.name}.type", p.type) for p in op.params]
    return out


@pytest.mark.parametrize("op", OPS, ids=lambda op: op.name)
def test_entry_is_complete(op: Op) -> None:
    assert op.summary.strip(), f"{op.name}: no summary"
    assert op.description.strip(), f"{op.name}: no description"
    assert op.emits.strip(), f"{op.name}: says nothing about what it emits"
    for p in op.params:
        assert p.type.strip(), f"{op.name}.{p.name}: no type"
        assert p.doc.strip(), f"{op.name}.{p.name}: no description"
    names = [p.name for p in op.params]
    assert len(names) == len(set(names)), f"{op.name}: duplicate params"


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
    # mistake the documentation exists to prevent.
    check_params(op.ref, op.example)


def test_common_params_are_documented() -> None:
    for p in lexicon.COMMON:
        assert p.type.strip() and p.doc.strip(), f"common param {p.name} undocumented"
        for pat in INTERNAL:
            assert not pat.search(p.doc), f"common param {p.name}: house idiom in doc"
    assert COMMON == frozenset(p.name for p in lexicon.COMMON)


def test_block_params_is_derived_from_the_lexicon() -> None:
    """One source. `block_params` must not carry its own table."""
    assert set(ACCEPTED) == set(BY_REF)
    for ref, op in BY_REF.items():
        assert ACCEPTED[ref] == op.param_names
    # A param cannot be both common and op-specific: the table would
    # accept it twice and the docs would describe it twice.
    for op in OPS:
        assert not (op.param_names & COMMON), f"{op.name} redeclares a common param"


def test_refs_are_well_formed_and_unique() -> None:
    refs = [op.ref for op in OPS]
    assert len(refs) == len(set(refs))
    for ref in refs:
        assert re.fullmatch(r"~canonical/ops/[a-z0-9-]+(?:/[a-z0-9-]+)*/\d+", ref), ref


def test_to_dict_is_json_shaped() -> None:
    import json

    for op in OPS:
        json.dumps(op.to_dict())  # REQUIRED never leaks; defaults are JSON values
        for p in op.params:
            d = p.to_dict()
            assert ("default" in d) != d["required"]
