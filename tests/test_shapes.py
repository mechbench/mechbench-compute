"""The shared shapes (docs/LEXICON.md §4–§5): one constructor per value
type and item, and readers that take the older spellings too."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mechbench_compute import shapes as S


class Tok:
    def decode(self, ids):
        return "".join(f"t{int(i)}" for i in ids)


SP = S.space(model="m", layer=3, point="resid_post", d=4)


class TestSpace:
    def test_every_space_has_the_same_five_fields(self):
        assert SP == {"model": "m", "layer": 3, "point": "resid_post", "head": None, "d": 4}
        assert S.space(model=None, layer=None, point="embed", d=2, head=1)["head"] == 1

    def test_same_space_compares_the_fields_that_matter(self):
        a = {"space": SP}
        S.same_space(a, {"space": dict(SP)})
        S.same_space(a, {"space": {**SP, "model": None}})  # an unknown model is not a disagreement
        with pytest.raises(ValueError, match="layer"):
            S.same_space(a, {"space": {**SP, "layer": 4}})
        with pytest.raises(ValueError, match="model"):
            S.same_space(a, {"space": {**SP, "model": "other"}})

    def test_space_of_assembles_the_older_spelling(self):
        # `layer` on the item, `point` / `d_model` / `model` on the header
        sp = S.space_of({"layer": 5, "vector": [0, 0, 0]}, header={"point": "post", "model": "hf/x"})
        assert sp == {"model": "hf/x", "layer": 5, "point": "resid_post", "head": None, "d": 3}
        # a direction's flattened fields
        sp = S.space_of({"layer": 2, "point": "resid_pre", "d": 8, "derivation": {"model": "hf/y"}})
        assert sp["model"] == "hf/y" and sp["point"] == "resid_pre" and sp["d"] == 8
        # a per-head source
        assert S.space_of({"layer": 1, "head": 3, "vector": [1.0]})["head"] == 3


class TestItems:
    def test_vector_carries_space_norm_and_coords(self):
        v = S.vector([3.0, 0.0, 4.0, 0.0], SP, id="r", coords={"genre": "noir"},
                     token={"id": 7, "text": "t7"}, n_pooled=None)
        assert v["space"] == SP and v["norm"] == 5.0 and v["coords"] == {"genre": "noir"}
        assert v["token"] == {"id": 7, "text": "t7"} and "n_pooled" not in v
        with pytest.raises(ValueError, match="dims"):
            S.vector([1.0, 2.0], SP)

    def test_distribution_ranks_and_tracks(self):
        lp = np.log(np.array([0.1, 0.6, 0.3]))
        d = S.distribution(lp, Tok(), top_k=2, tracked={"yes": 2, "no": 0})
        assert [t["token"]["text"] for t in d["top"]] == ["t1", "t2"]
        assert d["top"][0]["p"] == pytest.approx(0.6) and d["top"][0]["logp"] == pytest.approx(math.log(0.6), abs=1e-4)
        assert d["tracked"]["yes"] == {"token": {"id": 2, "text": "t2"}, "p": 0.3, "logp": round(math.log(0.3), 4)}
        assert d["entropy_bits"] == pytest.approx(1.2955, abs=1e-3)
        assert "tracked" not in S.distribution(lp, Tok(), top_k=1)

    def test_grid_and_coordinate(self):
        g = S.grid("r", ["layer", "position"], {"logprob": [[0.0, -1.0]]}, tokens=["a", "b"],
                   target={"id": 1, "text": "t1"})
        assert g["axes"] == ["layer", "position"] and g["measures"]["logprob"] == [[0.0, -1.0]]
        assert g["tokens"] == ["a", "b"] and g["target"]["text"] == "t1"
        d = {"kind": "direction/vector", "space": SP, "vector": [1, 0, 0, 0],
             "derivation": {"method": "diff_of_means", "axis": "genre", "positive": "noir", "negative": "fable"}}
        c = S.coordinate(0.25, SP, d, id="r", coords={"genre": "noir"}, step=3)
        assert c["coord"] == 0.25 and c["step"] == 3
        assert c["direction"] == {"space": SP, "method": "diff_of_means", "axis": "genre",
                                  "positive": "noir", "negative": "fable"}


class TestReaders:
    def test_label_of_reads_a_coordinate_or_the_retired_field(self):
        assert S.label_of({"coords": {"genre": "noir"}}, "genre") == "noir"
        assert S.label_of({"label": "x"}, "label") == "x"
        assert S.label_of({"coords": {"label": "y"}, "label": "x"}, None) == "y"
        assert S.label_of({"coords": {}}, "genre") is None
        assert S.coords_of({"label": "x", "coords": {"a": 1}}) == {"a": 1, "label": "x"}

    def test_distribution_of_reads_the_retired_spellings(self):
        old = {"id": "c", "entropy_bits": 1.0,
               "top_tokens": [{"token": "3", "p": 0.5}, {"token": "1", "p": 0.25}],
               "outcome_mass": {"3": 0.5, "1": 0.25}}
        d = S.distribution_of(old)
        assert d["top"][0]["token"] == {"id": None, "text": "3"} and d["top"][0]["p"] == 0.5
        assert d["top"][0]["logp"] == pytest.approx(math.log(0.5))
        assert d["tracked"]["1"]["p"] == 0.25 and "outcome_mass" not in d
        steer = {"top": [{"token": " x", "logp": -0.1}], "track_logp": -2.0, "tracks": {"yes": -0.5}}
        d = S.distribution_of(steer)
        assert d["top"][0]["p"] == pytest.approx(math.exp(-0.1))
        assert d["tracked"]["yes"]["logp"] == -0.5 and d["tracked"]["track"]["logp"] == -2.0
        # a current item comes back as it is
        cur = {"top": [{"token": {"id": 1, "text": "t1"}, "p": 0.5, "logp": -0.69}]}
        assert S.distribution_of(cur) == cur

    def test_measures_of_reads_a_grid_or_the_retired_fields(self):
        assert S.measures_of({"measures": {"rank": [[0]]}}, {"rank": "rank"}) == {"rank": [[0]]}
        assert S.measures_of({"logprob": [[1]], "rank": [[0]]}, {"logprob": "logprob", "rank": "rank"}) == {
            "logprob": [[1]], "rank": [[0]]}
