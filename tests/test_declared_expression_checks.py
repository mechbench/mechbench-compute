from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mechbench_compute import dataflow
from mechbench_compute.registry import CORE

ROOT = Path(__file__).resolve().parents[1]


def _kind(name: str):
    return next(k for k in CORE.kinds() if k.name == name)


def test_a_kind_declares_closed_sub_fields_by_the_prefix_the_engine_spells():
    assert _kind("logits/distribution").subfields == {"top[]": ("token", "p", "logp")}
    assert _kind("activations/vector").subfields["space"] == ("model", "layer", "point", "head", "d")
    assert _kind("logits/lens").subfields == {
        "target": ("id", "text"), "variants[]": ("token", "p", "logp"),
        "cells[]": ("id", "coords", "parent", "address", "point", "layer", "position", "token", "logprob", "rank")}


def test_an_open_map_declares_no_sub_fields():
    subs = _kind("records/record").subfields
    assert "coords" not in subs
    assert "tracked" not in _kind("logits/distribution").subfields


def test_the_generated_kind_table_carries_the_sub_fields():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "dump_kinds_ts.py")],
                         capture_output=True, text=True, check=True).stdout
    row = next(line for line in out.splitlines() if line.startswith('  "logits/distribution":'))
    assert 'subfields: {"top[]": ["token", "p", "logp"]}' in row
    record = next(line for line in out.splitlines() if line.startswith('  "records/record":'))
    assert "subfields" not in record


def _mapped(inner_params: dict) -> dict:
    body = {"nodes": [{"id": "keep", "block": "records/filter", "params": inner_params}], "edges": []}
    return {"each": {"block": "records/map", "params": {"over": [1, 2], "as": "layer", "body": body}}}


def test_a_declared_expression_inside_a_map_body_is_syntax_checked():
    with pytest.raises(ValueError, match=r"each\.body\.nodes\.0\.params\.where: .*`p >`"):
        dataflow.check_refs(_mapped({"where": "p >"}), {})


def test_a_declared_expression_inside_a_map_body_reads_only_params_and_bound_names():
    dataflow.check_refs(_mapped({"where": "layer > params.layer"}), {})
    dataflow.check_refs(_mapped({"where": "p > params.cut"}), {"cut": 0.5})
    with pytest.raises(ValueError, match=r"`p > params\.cut` reads params\.cut, which is not a param"):
        dataflow.check_refs(_mapped({"where": "p > params.cut"}), {})


def test_a_declared_expression_param_bound_through_a_param_is_checked():
    with pytest.raises(ValueError, match=r"each\.body\.nodes\.0\.params\.where: .*`p >`"):
        dataflow.check_refs(_mapped({"where": {"$param": "w"}}), {"w": "p >"})


def test_a_top_level_expression_map_and_list_are_checked_value_by_value():
    ok = {"n": {"block": "records/sort", "params": {"by": ["-p", "id"]}},
          "d": {"block": "records/derive", "params": {"fields": {"q": "p * 2"}, "templates": {"t": "{id}"}}}}
    dataflow.check_refs(ok, {})
    bad = {"d": {"block": "records/derive", "params": {"fields": {"q": "p *"}}}}
    with pytest.raises(ValueError, match=r"d\.fields\.q: .*`p \*`"):
        dataflow.check_refs(bad, {})


def test_a_string_where_no_expression_is_declared_is_left_alone():
    dataflow.check_refs({"n": {"block": "records/filter", "params": {"where": "p > 0", "label": "p >"}}}, {})
