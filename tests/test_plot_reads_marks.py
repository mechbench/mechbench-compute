from __future__ import annotations

import pytest

from mechbench_compute.ops.records.plot import build_chart

ROWS = [{"id": f"L{i}", "layer": i, "position": i % 2, "v": i / 10, "mean": -i / 10,
         "lo": -i / 10 - 0.1, "hi": -i / 10 + 0.1} for i in range(4)]
ARCH = {"n_layers": 4, "global_layers": [1, 3], "first_kv_shared_layer": 2}


def chart(**params):
    return build_chart(ROWS, params)


class TestRefusals:
    def test_a_missing_channel_names_the_mark_and_every_channel_it_needs(self):
        with pytest.raises(ValueError, match=r"^a heat mark needs encoding\.y and encoding\.value\.$"):
            chart(mark="heat", encoding={"x": "layer"})
        with pytest.raises(ValueError, match=r"^a tokens mark needs encoding\.text and encoding\.value\.$"):
            chart(mark="tokens", encoding={})

    def test_a_channel_the_mark_does_not_draw_is_refused(self):
        with pytest.raises(ValueError) as e:
            chart(mark="heat", encoding={"x": "layer", "y": "position", "value": "v", "series": "s", "lo": "lo"})
        assert str(e.value) == ("a heat mark does not draw encoding.series and encoding.lo; "
                                "it reads encoding.x, encoding.y and encoding.value.")
        with pytest.raises(ValueError, match="a tokens mark does not draw encoding.color"):
            chart(mark="tokens", encoding={"text": "t", "value": "v", "color": "kind"})

    def test_a_setting_the_mark_does_not_draw_is_refused(self):
        with pytest.raises(ValueError, match=r"a heat mark does not draw annotate and reference\.$"):
            chart(mark="heat", encoding={"x": "layer", "y": "position", "value": "v"},
                  annotate=[{"at": {"layer": 1}, "text": "t"}], reference=[{"y": 0}])
        with pytest.raises(ValueError, match=r"a bar mark does not draw scale\."):
            chart(mark="bar", encoding={"x": "layer", "y": "mean"}, scale="diverging")
        with pytest.raises(ValueError, match=r"a line mark does not draw bin\."):
            chart(mark="line", encoding={"x": "layer", "y": "mean"}, bin=5)

    def test_an_unknown_mark_or_version_is_refused(self):
        with pytest.raises(ValueError, match="'pie' is not a mark"):
            chart(mark="pie", encoding={"x": "layer", "y": "mean"})
        with pytest.raises(ValueError, match="'heat@2' is not a mark"):
            chart(mark="heat@2", encoding={"x": "layer", "y": "position", "value": "v"})


class TestTheAddress:
    def test_the_mark_is_written_as_name_at_version(self):
        assert chart(mark="point", encoding={"x": "layer", "y": "mean"})["mark"] == "point@1"
        assert chart(mark="point@1", encoding={"x": "layer", "y": "mean"})["mark"] == "point@1"

    def test_the_legacy_spellings_are_rewritten(self):
        assert chart(mark="scatter", encoding={"x": "layer", "y": "mean"})["mark"] == "point@1"
        spec = chart(mark="histogram", encoding={"x": "v"})
        assert (spec["mark"], spec["bin"], spec["encoding"]) == ("bar@1", 20, {"x": "v"})
        assert chart(mark="histogram", encoding={"x": "v"}, bin=8)["bin"] == 8


class TestBin:
    def test_a_binned_bar_needs_no_y_and_leaves_the_counting_to_the_renderer(self):
        spec = chart(mark="bar", bin=5, encoding={"x": "v"})
        assert spec["bin"] == 5 and spec["encoding"] == {"x": "v"}
        assert spec["data"]["rows"] == ROWS

    def test_a_y_beside_bin_is_refused(self):
        with pytest.raises(ValueError, match="a bar mark with bin draws encoding.y itself"):
            chart(mark="bar", bin=5, encoding={"x": "v", "y": "mean"})

    def test_bin_is_a_whole_number(self):
        for bad in (0, 2.5, "ten", True):
            with pytest.raises(ValueError, match="bin is how many bins"):
                chart(mark="bar", bin=bad, encoding={"x": "v"})


class TestLevel:
    def test_a_given_level_rides_on_the_spec(self):
        spec = chart(mark="point", level=0.9, encoding={"x": "layer", "y": "mean", "lo": "lo", "hi": "hi"})
        assert spec["level"] == 0.9

    def test_the_level_is_read_from_the_inputs_interval(self):
        table = {"kind": "records/table", "rows": ROWS,
                 "interval": {"level": 0.95, "method": "percentile-bootstrap", "resamples": 2000, "seed": 0}}
        spec = build_chart(table, {"mark": "line", "encoding": {"x": "layer", "y": "mean", "lo": "lo", "hi": "hi"}})
        assert spec["level"] == 0.95
        bare = build_chart(table, {"mark": "line", "encoding": {"x": "layer", "y": "mean"}})
        assert "level" not in bare
        heat = build_chart(table, {"mark": "heat", "encoding": {"x": "layer", "y": "position", "value": "v"}})
        assert "level" not in heat

    def test_no_level_is_assumed(self):
        assert "level" not in chart(mark="point", encoding={"x": "layer", "y": "mean", "lo": "lo", "hi": "hi"})

    def test_a_level_outside_zero_and_one_is_refused(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            chart(mark="point", level=95, encoding={"x": "layer", "y": "mean", "lo": "lo", "hi": "hi"})


class TestDepth:
    def coll(self):
        return {"kind": "collection", "item_kind": "intervene/ablation", "items": ROWS, "arch": ARCH}

    def test_landmarks_are_stamped_when_layer_is_on_x(self):
        spec = build_chart(self.coll(), {"mark": "heat", "encoding": {"x": "layer", "y": "position", "value": "v"}})
        assert spec["axes"] == {"layer": {"n": 4, "global": [1, 3], "kv_shared_from": 2}}

    def test_landmarks_are_not_stamped_when_layer_is_on_y(self):
        spec = build_chart(self.coll(), {"mark": "heat", "encoding": {"x": "position", "y": "layer", "value": "v"}})
        assert "axes" not in spec

    def test_a_token_strip_has_no_depth_channel(self):
        rows = {**self.coll(), "items": [{"id": "s", "layer": 1, "tokens": ["a"], "v": [1.0]}]}
        spec = build_chart(rows, {"mark": "tokens", "encoding": {"text": "tokens", "value": "v", "x": "layer"}})
        assert "axes" not in spec
