from __future__ import annotations

import json
import pathlib

import pytest

from mechbench_compute.ops.records import plot

HERE = pathlib.Path(__file__).resolve().parent.parent
VENDORED = HERE / "mechbench_compute" / "ops" / "records" / "marks.generated.json"
VIZ = HERE.parent / "mechbench-viz" / "marks.generated.json"


def test_the_vendored_marks_are_vizs():
    if not VIZ.exists():
        pytest.skip("no mechbench-viz checkout beside this one")
    assert json.loads(VENDORED.read_text()) == json.loads(VIZ.read_text()), (
        "ops/records/marks.generated.json is stale: run\n"
        "  cp ../mechbench-viz/marks.generated.json mechbench_compute/ops/records/")


def test_the_declaration_names_what_the_file_declares():
    names = [m["name"] for m in plot.DECLARED["marks"]]
    assert list(plot.MARKS) == names
    params = {p.name: p for p in plot.OP.params}
    assert list(params["mark"].choices) == names
    assert sorted(f.name for f in params["encoding"].fields) == sorted(plot.CHANNELS)
    assert set(plot.SETTINGS) <= set(params)
    assert list(params["scale"].choices) == plot.BY_NAME["heat"]["options"]["scale"]["choices"] == list(plot.SCALES)


def test_the_package_ships_the_file():
    text = (HERE / "pyproject.toml").read_text()
    assert '"mechbench_compute.ops.records" = ["marks.generated.json"]' in text
