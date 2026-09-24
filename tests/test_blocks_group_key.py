from __future__ import annotations

import pytest

from mechbench_compute import blocks
from mechbench_compute.ops.records.summarize import GroupStats
from mechbench_compute.ops.records.summarize import group_stats
from mechbench_compute.ops.records.plot import build_chart


ROWS = [
    {"id": "a", "layer": 0, "delta": -8.0},
    {"id": "b", "layer": 0, "delta": -6.0},
    {"id": "c", "layer": 1, "delta": -2.0},
]
COORD_ROWS = [
    {"id": "a", "coords": {"layer": 0}, "delta": -8.0},
    {"id": "b", "coords": {"layer": 1}, "delta": -2.0},
]


def _by_layer(rows):
    out = group_stats({"kind": "collection", "items": rows},
                      {"value": "delta", "by": ["layer"]})
    return {r["layer"]: r["n"] for r in out["rows"]}


def test_a_top_level_field_groups():
    assert _by_layer(ROWS) == {0: 2, 1: 1}


def test_a_coordinate_still_groups():
    assert _by_layer(COORD_ROWS) == {0: 1, 1: 1}


def test_a_coordinate_wins_over_a_top_level_field_of_the_same_name():
    rows = [{"id": "a", "layer": 9, "coords": {"layer": 0}, "delta": -1.0}]
    assert _by_layer(rows) == {0: 1}


def test_the_monoid_agrees_with_the_flat_block():
    m = GroupStats()
    params = {"value": "delta", "by": ["layer"]}
    part = m.merge(m.partial(ROWS[:2], params), m.partial(ROWS[2:], params))
    flat = group_stats({"kind": "collection", "items": ROWS}, params)
    assert m.finalize(part, params)["rows"] == flat["rows"]


class TestChartMarks:
    ROWS = {"kind": "collection", "item_kind": "records/record", "items": [
        {"id": "a", "coords": {"layer": 0, "position": 1}, "recovery": -0.5, "mean": 1.0,
         "lo": 0.5, "hi": 1.5, "tokens": ["the", " cat"], "surprisal": [1.0, 4.0]},
    ]}

    def test_a_heat_mark_takes_three_fields(self):
        spec = build_chart(self.ROWS, {"mark": "heat", "x": "position", "y": "layer",
                                           "value": "recovery", "scale": "diverging"})
        assert spec["mark"] == "heat" and spec["scale"] == "diverging"
        assert spec["encoding"] == {"x": "position", "y": "layer", "value": "recovery"}
        assert spec["data"]["rows"][0]["layer"] == 0
        with pytest.raises(ValueError, match="heat mark needs"):
            build_chart(self.ROWS, {"mark": "heat", "x": "position", "y": "layer"})

    def test_a_token_strip_takes_the_tokens_and_the_number(self):
        spec = build_chart(self.ROWS, {"mark": "tokens", "text": "tokens",
                                           "value": "surprisal"})
        assert spec["mark"] == "tokens"
        assert spec["encoding"] == {"value": "surprisal", "text": "tokens"}
        with pytest.raises(ValueError, match="tokens mark needs"):
            build_chart(self.ROWS, {"mark": "tokens", "text": "tokens"})

    def test_an_interval_rides_beside_the_point_it_belongs_to(self):
        spec = build_chart(self.ROWS, {"mark": "line", "x": "layer", "y": "mean",
                                           "encoding": {"x": "layer", "y": "mean",
                                                        "lo": "lo", "hi": "hi"}})
        assert spec["encoding"] == {"x": "layer", "y": "mean", "lo": "lo", "hi": "hi"}

    def test_the_older_marks_are_unchanged(self):
        spec = build_chart(self.ROWS, {"mark": "bar", "x": "layer", "y": "mean"})
        assert spec["mark"] == "bar" and spec["encoding"] == {"x": "layer", "y": "mean"}
        assert "scale" not in spec
        with pytest.raises(ValueError, match="one of bar, line, point, heat, tokens"):
            build_chart(self.ROWS, {"mark": "sparkline", "x": "layer", "y": "mean"})

    GRID = {"kind": "collection", "item_kind": "intervene/trace", "items": [
        {"id": "eiffel", "coords": {"topic": "landmark"}, "axes": ["layer", "position"],
         "tokens": ["The", " Tower", " is"],
         "measures": {"recovery": [[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]}},
    ]}

    def test_a_grid_becomes_one_row_per_cell(self):
        spec = build_chart(self.GRID, {"mark": "heat", "x": "position", "y": "layer",
                                           "value": "recovery"})
        rows = spec["data"]["rows"]
        assert len(rows) == 6
        assert rows[0] == {"id": "eiffel", "topic": "landmark", "layer": 0, "position": 0,
                           "token": "The", "recovery": 0.1}
        assert rows[4]["layer"] == 1 and rows[4]["position"] == 1 and rows[4]["recovery"] == 2.0

    def test_what_is_not_a_grid_is_left_alone(self):
        assert blocks.expand_grid({"id": "x", "value": 1}) is None
        assert blocks.expand_grid({"id": "x", "axes": ["layer"], "measures": {}}) is None
        spec = build_chart(self.ROWS, {"mark": "bar", "x": "layer", "y": "mean"})
        assert len(spec["data"]["rows"]) == 1
