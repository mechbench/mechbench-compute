from __future__ import annotations

from mechbench_compute.blocks.sort_group_key import sort_group_key
from mechbench_compute.ops.records.contrast import contrast
from mechbench_compute.ops.records.correlate import correlate
from mechbench_compute.ops.records.count import count
from mechbench_compute.ops.records.summarize import group_stats

LAYERS = [0, 1, 2, 10, 11, 20]


def _records() -> list[dict]:
    out = []
    for layer in reversed(LAYERS):
        for fact in range(4):
            for condition in ("honest", "lie"):
                p = 0.1 * fact + (0.5 if condition == "honest" else 0.0) + 0.01 * layer
                out.append({
                    "id": f"{condition}-{fact}-{layer}",
                    "coords": {"condition": condition, "fact": fact},
                    "layer": layer, "p": p, "q": p * p, "top1": p > 0.5,
                })
    return out


def _dtypes(table: dict) -> dict[str, str]:
    return {c["name"]: c["dtype"] for c in table["columns"]}


def test_numbers_come_before_text_and_in_numeric_order():
    keys = [(10,), (2,), ("b",), (1.5,), ("a",)]
    assert sorted(keys, key=sort_group_key) == [(1.5,), (2,), (10,), ("a",), ("b",)]


def test_contrast_rows_follow_the_layers_and_type_them_as_numbers():
    out = contrast(_records(), {"value": "p", "on": "condition", "a": "honest",
                                "b": "lie", "by": ["layer"], "paired": "fact"})
    assert [r["layer"] for r in out["rows"]] == LAYERS
    assert _dtypes(out)["layer"] == "number"
    assert _dtypes(out)["on"] == _dtypes(out)["a"] == "string"


def test_count_and_correlate_order_and_type_a_numeric_group():
    counted = count(_records(), {"field": "top1", "equals": True, "by": ["layer"]})
    assert [r["layer"] for r in counted["rows"]] == LAYERS
    assert _dtypes(counted)["layer"] == "number"
    rho = correlate(_records(), {"x": "p", "y": "q", "by": ["layer"]})
    assert [r["layer"] for r in rho["rows"]] == LAYERS
    assert _dtypes(rho)["layer"] == "number"


def test_summarize_types_a_numeric_group_and_keeps_text_as_text():
    out = group_stats(_records(), {"value": "p", "by": ["layer", "condition"]})
    assert _dtypes(out)["layer"] == "number"
    assert _dtypes(out)["condition"] == "string"
