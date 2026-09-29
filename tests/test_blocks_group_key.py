from __future__ import annotations

import pytest

from mechbench_compute import blocks
from mechbench_compute.ops.records.plot import build_chart


class TestChartMarks:
    ROWS = {"kind": "collection", "item_kind": "records/record", "items": [
        {"id": "a", "coords": {"layer": 0, "position": 1}, "recovery": -0.5, "mean": 1.0,
         "lo": 0.5, "hi": 1.5, "tokens": ["the", " cat"], "surprisal": [1.0, 4.0]},
    ]}

    def test_a_heat_mark_takes_three_fields(self):
        spec = build_chart(self.ROWS, {"mark": "heat", "x": "position", "y": "layer",
                                           "value": "recovery", "scale": "diverging"})
        assert spec["mark"] == "heat@1" and spec["scale"] == "diverging"
        assert spec["encoding"] == {"x": "position", "y": "layer", "value": "recovery"}
        assert spec["data"]["rows"][0]["layer"] == 0
        with pytest.raises(ValueError, match="heat mark needs"):
            build_chart(self.ROWS, {"mark": "heat", "x": "position", "y": "layer"})

    def test_a_token_strip_takes_the_tokens_and_the_number(self):
        spec = build_chart(self.ROWS, {"mark": "tokens", "text": "tokens",
                                           "value": "surprisal"})
        assert spec["mark"] == "tokens@1"
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
        assert spec["mark"] == "bar@1" and spec["encoding"] == {"x": "layer", "y": "mean"}
        assert "scale" not in spec
        with pytest.raises(ValueError, match="bar@1, line@1, point@1, heat@1 and tokens@1"):
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
