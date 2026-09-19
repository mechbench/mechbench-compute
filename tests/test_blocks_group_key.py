"""Grouping reads a field wherever the record carries it (000590)."""

from __future__ import annotations

from mechbench_compute.blocks import group_stats
from mechbench_compute.reduce import GroupStats


ROWS = [
    # An ablation's rows: the varying thing is top-level, not a coord.
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
    # It produced ONE row keyed None before: forty-two layers of an
    # ablation collapsed into a single meaningless mean.
    assert _by_layer(ROWS) == {0: 2, 1: 1}


def test_a_coordinate_still_groups():
    assert _by_layer(COORD_ROWS) == {0: 1, 1: 1}


def test_a_coordinate_wins_over_a_top_level_field_of_the_same_name():
    # Coordinates are where a condition belongs; if a record says both,
    # the coordinate is the considered answer.
    rows = [{"id": "a", "layer": 9, "coords": {"layer": 0}, "delta": -1.0}]
    assert _by_layer(rows) == {0: 1}


def test_the_monoid_agrees_with_the_flat_block():
    # Resume replays through the monoid; the two must not diverge.
    m = GroupStats()
    params = {"value": "delta", "by": ["layer"]}
    part = m.merge(m.partial(ROWS[:2], params), m.partial(ROWS[2:], params))
    flat = group_stats({"kind": "collection", "items": ROWS}, params)
    assert m.finalize(part, params)["rows"] == flat["rows"]
