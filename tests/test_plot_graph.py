from __future__ import annotations

import copy

import jsonschema
import pytest

from mechbench_compute import platform_kinds
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.plot import build_chart

LAYOUT = {0: ("attn.per_head_out", 3), 1: ("mlp_out", 2), 2: ("attn.per_head_out", 3)}


def component(layer: int, i: int) -> dict:
    point, _ = LAYOUT[layer]
    head = {"head": i} if point.startswith("attn") else {}
    address = f"L{layer}.{point}{f'.H{i}' if head else ''}@-1" if head else f"L{layer}.{point}.{i}@-1"
    return {"address": address, "point": point, "layer": layer, **head, "position": -1,
            "effect": round((-1) ** i * (0.5 + 0.25 * i + 0.1 * layer), 3)}


COMPONENTS = [component(layer, i) for layer, (_, n) in LAYOUT.items() for i in range(n)]
BY_LAYER = {layer: [c["address"] for c in COMPONENTS if c["layer"] == layer] for layer in LAYOUT}
PAIRS = ([(a, b) for a in BY_LAYER[0] for b in BY_LAYER[1]][:6]
         + [(a, b) for a in BY_LAYER[1] for b in BY_LAYER[2]][:4])
LINKS = [{"from": a, "to": b, "effect": round(0.4 - 0.1 * i, 3)} for i, (a, b) in enumerate(PAIRS)]
CIRCUIT = {
    "id": "ioi", "components": COMPONENTS, "links": LINKS, "metric": "logprob",
    "measure": "mean_delta", "ablation": "patch",
    "universe": {"points": ["attn.per_head_out", "mlp_out"], "layers": [0, 1, 2], "positions": [-1]},
    "source": {"kind": "hand"}, "task": {"ids": ["p0"], "targets": []},
    "derivation": {"method": "hand", "sign": "positive", "kept": 8, "total": 8, "kept_share": 1.0},
}
DRAW = K.BY_KIND["intervene/circuit"].draw


def graph(records=CIRCUIT, **params):
    return build_chart(records, {"mark": "graph", "encoding": dict(DRAW.encoding), **params})


class TestTheFixture:
    def test_three_layers_eight_components_ten_links_and_a_valid_circuit(self):
        assert len({c["layer"] for c in COMPONENTS}) == 3 and len(COMPONENTS) == 8 and len(LINKS) == 10
        addresses = {c["address"] for c in COMPONENTS}
        assert len(addresses) == 8 and all(e["from"] in addresses and e["to"] in addresses for e in LINKS)
        jsonschema.validate(CIRCUIT, platform_kinds.item_schema(K.BY_KIND["intervene/circuit"]))


class TestAGraph:
    def test_the_circuit_draws_through_its_kinds_draw(self):
        before = copy.deepcopy(CIRCUIT)
        spec = graph(threshold=0.15, layout="rows", order="weight", labels={"weight": "Δ log p"},
                     title="The circuit")
        assert spec["mark"] == "graph@1"
        assert spec["encoding"] == {"id": "address", "layer": "layer", "weight": "effect",
                                    "components": "components", "links": "links"}
        assert spec["threshold"] == 0.15 and spec["layout"] == "rows" and spec["order"] == "weight"
        assert spec["labels"] == {"weight": "Δ log p"}
        assert spec["data"]["rows"] == [before] and CIRCUIT == before
        assert "axes" not in spec

    def test_a_collection_of_circuits_is_one_graph_each(self):
        other = {**CIRCUIT, "id": "half", "components": COMPONENTS[:4], "links": LINKS[:3]}
        coll = K.collection("intervene/circuit", [CIRCUIT, other])
        spec = graph(coll, facet="id")
        assert [r["id"] for r in spec["data"]["rows"]] == ["ioi", "half"] and spec["facet"] == "id"

    def test_a_stored_source_rides_by_reference(self):
        spec = build_chart(CIRCUIT, {"mark": "graph", "encoding": dict(DRAW.encoding)}, source_label="you/lab/c")
        assert spec["source"] == "you/lab/c" and "data" not in spec


class TestTheRefusals:
    def test_a_missing_components_channel_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"^a graph mark needs encoding\.components\.$"):
            build_chart(CIRCUIT, {"mark": "graph", "encoding": {"id": "address"}})

    def test_a_channel_or_setting_the_graph_does_not_draw(self):
        with pytest.raises(ValueError, match=r"a graph mark does not draw encoding\.x;"):
            graph(encoding={**DRAW.encoding, "x": "layer"})
        with pytest.raises(ValueError, match=r"a graph mark does not draw annotate and scale\.$"):
            graph(scale="diverging", annotate=[{"at": {"layer": 1}, "text": "t"}])

    def test_the_graph_settings_are_refused_on_another_mark(self):
        with pytest.raises(ValueError, match=r"a bar mark does not draw threshold, layout and order\.$"):
            build_chart([{"a": 1, "b": 2}], {"mark": "bar", "encoding": {"x": "a", "y": "b"},
                                             "threshold": 0.1, "layout": "rows", "order": "id"})

    def test_a_bad_threshold_layout_or_order(self):
        with pytest.raises(ValueError, match="threshold is a |weight| at or above zero"):
            graph(threshold=-1)
        with pytest.raises(ValueError, match="layout must be one of columns, rows, not 'grid'"):
            graph(layout="grid")
        with pytest.raises(ValueError, match="order must be one of id, weight, not 'size'"):
            graph(order="size")

    def test_a_record_without_components_is_refused(self):
        with pytest.raises(ValueError, match=r"record 1 carries no list under `components`"):
            graph([CIRCUIT, {"id": "x"}])

    def test_an_unknown_label_is_still_refused(self):
        with pytest.raises(ValueError, match="labels names"):
            graph(labels={"sign": "s"})


ROWS = [{"layer": i, "position": i % 2, "v": i - 1.5} for i in range(4)]
HEAT = {"mark": "heat", "encoding": {"x": "layer", "y": "position", "value": "v"}}


class TestTheScale:
    def test_the_string_and_the_object(self):
        assert build_chart(ROWS, {**HEAT, "scale": "sequential"})["scale"] == "sequential"
        spec = build_chart(ROWS, {**HEAT, "scale": {"ramp": "diverging", "per": "row"}})
        assert spec["scale"] == {"ramp": "diverging", "per": "row"}
        assert build_chart(ROWS, {**HEAT, "scale": {"per": "figure"}})["scale"] == {"per": "figure"}

    def test_a_token_strip_takes_a_scale(self):
        rows = [{"tokens": ["a", "b"], "values": [0.1, -0.2]}]
        spec = build_chart(rows, {"mark": "tokens", "encoding": {"text": "tokens", "value": "values"},
                                  "scale": {"per": "row"}})
        assert spec["scale"] == {"per": "row"}

    def test_an_unknown_ramp_per_or_key_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"scale\.ramp must be one of diverging, sequential, not 'rainbow'"):
            build_chart(ROWS, {**HEAT, "scale": {"ramp": "rainbow"}})
        with pytest.raises(ValueError, match=r"scale\.per must be one of row, figure, not 'panel'"):
            build_chart(ROWS, {**HEAT, "scale": {"per": "panel"}})
        with pytest.raises(ValueError, match=r"scale names \['range'\]"):
            build_chart(ROWS, {**HEAT, "scale": {"range": [0, 1]}})
        with pytest.raises(ValueError, match="scale must be one of diverging, sequential"):
            build_chart(ROWS, {**HEAT, "scale": "rainbow"})


LINE = {"mark": "line", "encoding": {"x": "layer", "y": "v"}}


class TestTheAxes:
    def test_a_numeric_axis_passes_through(self):
        axes = {"x": {"scale": "log"},
                "y": {"scale": "signed-log", "constant": 0.5, "zero": True,
                      "quantiles": [0.99, {"q": 0.5, "value": 1.25}]}}
        spec = build_chart(ROWS, {**LINE, "axes": axes})
        assert spec["axes"] == {"x": {"scale": "log"},
                                "y": {"scale": "signed-log", "constant": 0.5,
                                      "quantiles": [0.99, {"q": 0.5, "value": 1.25}], "zero": True}}

    def test_beside_the_depth_landmarks(self):
        spec = build_chart(ROWS, {**LINE, "axes": {"layer": {"n": 4}, "y": {"scale": "linear"}}})
        assert spec["axes"] == {"layer": {"n": 4}, "y": {"scale": "linear"}}

    @pytest.mark.parametrize("axis, said", [
        ({"scale": "sqrt"}, r"axes\.y\.scale must be one of linear, log, signed-log, not 'sqrt'"),
        ({"constant": 0}, r"axes\.y\.constant is the signed-log axis's c, a number above zero, not 0"),
        ({"constant": -2.0}, r"axes\.y\.constant is .* not -2\.0"),
        ({"quantiles": [1.5]}, r"axes\.y\.quantiles\[0\] is a level in 0\.\.1"),
        ({"quantiles": [{"q": 0.5}]}, r"axes\.y\.quantiles\[0\]\.value is the number at that level"),
        ({"zero": "yes"}, r"axes\.y\.zero is true or false"),
        ({"ticks": 3}, r"axes\.y names \['ticks'\]"),
    ])
    def test_a_bad_axis_is_refused_by_name(self, axis, said):
        with pytest.raises(ValueError, match=said):
            build_chart(ROWS, {**LINE, "axes": {"y": axis}})
