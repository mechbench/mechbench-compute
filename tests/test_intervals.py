"""Intervals on a summary and a contrast between two conditions: a
sweep's peak becomes a claim with a width."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mechbench_compute import blocks
from mechbench_compute.ops.records.summarize import group_stats
from mechbench_compute.ops.records.contrast import contrast


def _sweep(n_prompts=30, layers=(21, 22, 23, 24), seed=1):
    """An ablation-shaped result: one record per prompt per layer, the
    prompt's own difficulty shared across layers (so pairing matters),
    layer 23 costing 1.0 more than its neighbours."""
    rng = random.Random(seed)
    out = []
    for i in range(n_prompts):
        hard = rng.gauss(0, 2.0)  # the prompt's own level, same at every layer
        for layer in layers:
            effect = -3.0 if layer == 23 else -2.0
            out.append({"id": f"p{i}", "layer": layer,
                        "delta_logp": hard + effect + rng.gauss(0, 0.3)})
    return out


class TestSummarizeInterval:
    def test_without_interval_the_rows_are_what_they_were(self):
        out = group_stats(_sweep(), {"by": ["layer"], "value": "delta_logp"})
        assert [c["name"] for c in out["columns"]] == ["layer", "n", "median", "mean", "min", "max", "share_negative"]
        assert "interval" not in out and "lo" not in out["rows"][0]

    def test_an_interval_brackets_the_mean_and_says_how_it_was_made(self):
        out = group_stats(_sweep(), {"by": ["layer"], "value": "delta_logp",
                                            "interval": 0.95, "resamples": 500})
        for row in out["rows"]:
            assert row["lo"] <= row["mean"] <= row["hi"]
            assert 0.3 < row["hi"] - row["lo"] < 2.5   # sd≈2 over 30 prompts → ±0.7-ish
        assert out["interval"] == {"level": 0.95, "method": "percentile-bootstrap", "of": "mean",
                                   "resamples": 500, "seed": 0}
        assert [c["name"] for c in out["columns"]][-2:] == ["lo", "hi"]

    def test_the_interval_is_a_function_of_the_records_not_their_order(self):
        recs = _sweep()
        a = group_stats(recs, {"by": ["layer"], "value": "delta_logp", "interval": 0.9})
        b = group_stats(list(reversed(recs)), {"by": ["layer"], "value": "delta_logp", "interval": 0.9})
        assert {r["layer"]: (r["lo"], r["hi"]) for r in a["rows"]} == {r["layer"]: (r["lo"], r["hi"]) for r in b["rows"]}

    def test_more_records_narrow_it(self):
        narrow = group_stats(_sweep(n_prompts=200), {"value": "delta_logp", "interval": 0.95})["rows"][0]
        wide = group_stats(_sweep(n_prompts=20), {"value": "delta_logp", "interval": 0.95})["rows"][0]
        assert narrow["hi"] - narrow["lo"] < wide["hi"] - wide["lo"]

    def test_a_single_record_has_a_zero_width_interval(self):
        out = group_stats([{"id": "x", "delta_logp": 1.5}], {"value": "delta_logp", "interval": 0.95})
        assert (out["rows"][0]["lo"], out["rows"][0]["hi"]) == (1.5, 1.5)

    def test_a_bad_level_is_refused(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            group_stats(_sweep(), {"value": "delta_logp", "interval": 95})


class TestContrast:
    def test_paired_the_layer_23_cost_is_resolved_against_its_neighbour(self):
        # The prompts' own levels (sd 2) swamp a 1.0 effect unless the
        # records are paired; paired, the interval is tight around −1.
        out = contrast(_sweep(), {"value": "delta_logp", "on": "layer", "a": 23, "b": 22,
                                         "paired": "id", "resamples": 500})
        [row] = out["rows"]
        assert row["on"] == "layer" and row["a"] == 23 and row["b"] == 22 and row["n"] == 30
        assert -1.3 < row["diff"] < -0.7
        assert row["lo"] < row["diff"] < row["hi"]
        assert row["hi"] - row["lo"] < 0.5
        assert row["hi"] < 0 and row["share_positive"] == 0.0
        assert out["interval"]["paired"] == "id" and out["interval"]["of"] == "difference of means"

    def test_unpaired_the_same_contrast_is_wide_and_uncertain(self):
        out = contrast(_sweep(), {"value": "delta_logp", "on": "layer", "a": 23, "b": 22,
                                         "resamples": 500})
        [row] = out["rows"]
        assert row["hi"] - row["lo"] > 1.0
        assert out["interval"]["paired"] is None

    def test_a_null_contrast_straddles_zero(self):
        out = contrast(_sweep(), {"value": "delta_logp", "on": "layer", "a": 22, "b": 21,
                                         "paired": "id", "resamples": 500})
        [row] = out["rows"]
        assert row["lo"] < 0 < row["hi"]
        assert 0.1 < row["share_positive"] < 0.9

    def test_by_holds_other_coordinates_fixed(self):
        recs = [dict(r, coords={"point": p}) for r in _sweep(n_prompts=12) for p in ("attn", "mlp")]
        out = contrast(recs, {"value": "delta_logp", "on": "layer", "a": 23, "b": 22,
                                     "paired": "id", "by": ["point"], "resamples": 200})
        assert [r["point"] for r in out["rows"]] == ["attn", "mlp"]
        assert [c["name"] for c in out["columns"]][:4] == ["point", "on", "a", "b"]

    def test_a_side_that_is_absent_is_refused_by_name(self):
        with pytest.raises(ValueError, match="layer=99"):
            contrast(_sweep(), {"value": "delta_logp", "on": "layer", "a": 99, "b": 22})

    def test_pairs_that_never_meet_are_refused(self):
        recs = [{"id": f"a{i}", "layer": 23, "delta_logp": 1.0} for i in range(3)] + \
               [{"id": f"b{i}", "layer": 22, "delta_logp": 1.0} for i in range(3)]
        with pytest.raises(ValueError, match="appears on both sides"):
            contrast(recs, {"value": "delta_logp", "on": "layer", "a": 23, "b": 22, "paired": "id"})

    def test_reads_the_coordinate_from_coords_or_the_record(self):
        recs = [{"id": f"p{i}", "coords": {"arm": arm}, "v": (1.0 if arm == "t" else 0.0) + 0.01 * i}
                for i in range(10) for arm in ("t", "c")]
        out = contrast(recs, {"value": "v", "on": "arm", "a": "t", "b": "c", "paired": "id", "resamples": 100})
        assert abs(out["rows"][0]["diff"] - 1.0) < 1e-9
