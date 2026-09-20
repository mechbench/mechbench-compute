"""`activations/examples` (task 000615): the windows that most excite a
direction, kept without holding the corpus."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mechbench_compute import interp
from mechbench_compute import shapes as S
from tests.test_interp_blocks import D_MODEL, StubModel


def _direction(dims, layer=1):
    v = np.zeros(D_MODEL, np.float32)
    for dim in dims:
        v[dim] = 1.0
    return {"kind": "direction/vector",
            **S.vector(v, S.space(model="stub", layer=layer, point="resid_post", d=D_MODEL)),
            "derivation": {"method": "test"}}


class TestExamples:
    RECORDS = [{"id": "r0", "user": "aa bbb aa"}, {"id": "r1", "user": "bbb bbb aa"}]

    def test_the_windows_come_back_ranked_with_the_token_marked(self):
        out = interp.examples(StubModel(), self.RECORDS, {"k": 3, "window": 1},
                              direction=_direction([1]))
        assert out["item_kind"] == "records/record"
        items = out["items"]
        assert [it["rank"] for it in items] == [0, 1, 2]
        assert [it["value"] for it in items] == sorted((it["value"] for it in items), reverse=True)
        first = items[0]
        assert first["coords"]["record"] in ("r0", "r1")
        assert first["tokens"][first["hit"]] == first["token"]
        assert first["text"] == "".join(first["tokens"])
        assert len(first["tokens"]) <= 3        # the hit and one either side

    def test_the_header_says_what_the_corpus_was_like(self):
        out = interp.examples(StubModel(), self.RECORDS, {"k": 2}, direction=_direction([1]))
        over = out["over"]
        assert over["n_tokens"] == sum(len(StubModel().tokenize(r["user"])[0]) for r in self.RECORDS)
        assert over["min"] <= over["mean"] <= over["max"] and over["sd"] >= 0
        assert out["layer"] == 1 and out["point"] == "resid_post" and out["window"] == 8

    def test_both_ends_are_marked_by_side(self):
        out = interp.examples(StubModel(), self.RECORDS, {"k": 2, "sign": "both"},
                              direction=_direction([1]))
        sides = [it["coords"]["side"] for it in out["items"]]
        assert sides == ["high", "high", "low", "low"]
        highs = [it["value"] for it in out["items"] if it["coords"]["side"] == "high"]
        lows = [it["value"] for it in out["items"] if it["coords"]["side"] == "low"]
        assert min(highs) >= max(lows)

    def test_a_neuron_is_the_other_way_to_name_what_to_excite(self):
        out = interp.examples(StubModel(), self.RECORDS,
                              {"k": 2, "neuron": {"layer": 1, "index": 1}, "point": "resid_post"})
        assert out["neuron"] == 1 and out["items"]
        # …and it is one or the other, never both or neither.
        with pytest.raises(ValueError, match="one of them, not both"):
            interp.examples(StubModel(), self.RECORDS, {"neuron": {"layer": 1, "index": 0}},
                            direction=_direction([1]))
        with pytest.raises(ValueError, match="one of them, not both"):
            interp.examples(StubModel(), self.RECORDS, {})

    def test_memory_does_not_grow_with_the_corpus(self):
        many = [{"id": f"r{i}", "user": "aa bbb aa"} for i in range(50)]
        out = interp.examples(StubModel(), many, {"k": 4}, direction=_direction([1]))
        assert len(out["items"]) == 4 and out["over"]["n_tokens"] > 100


E2B = "mlx-community/gemma-4-e2b-it-bf16"


@pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1"
    or not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--" + E2B.replace("/", "--"))),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached",
)
def test_on_gemma_a_probe_finds_the_words_it_was_fit_on():
    """A probe fit to separate colour words from animal words, pointed at
    a corpus, brings back colour words."""
    from mechbench_compute import Model
    from mechbench_compute import directions as dirs

    model = Model.load(E2B)
    layer = model.arch.n_layers // 2
    colours = ["red", "blue", "green", "yellow", "purple", "orange"]
    animals = ["otter", "badger", "heron", "marten", "ferret", "vole"]
    rows = []
    for label, words in (("colour", colours), ("animal", animals)):
        for w in words:
            res = model.run(model.tokenize(f"the {w}", chat_template=False),
                            capture=[f"blocks.{layer}.resid_post"])
            import mlx.core as mx
            v = np.array(res.cache[f"blocks.{layer}.resid_post"][0, -1].astype(mx.float32))
            rows.append({**S.vector(v, S.space(model="e2b", layer=layer, point="resid_post", d=v.size),
                                    id=w, coords={"kind": label}), "kind": "activations/vector"})
    probe = dirs.from_classification({"kind": "collection", "item_kind": "activations/vector",
                                      "items": rows}, axis="kind", seed=0, holdout=0.34)
    [axis] = [it for it in probe["items"]]
    if axis["derivation"]["positive"] == "animal":      # point it at colours either way
        axis = {**axis, "vector": [-x for x in axis["vector"]]}
    corpus = [{"id": "c0", "text": "The otter swam past a red buoy and a heron stood in the green reeds."},
              {"id": "c1", "text": "A badger crossed the yellow field where the blue van was parked."}]
    out = interp.examples(model, corpus, {"k": 4, "window": 3}, direction=axis)
    found = " ".join(it["token"].strip().lower() for it in out["items"])
    assert sum(c in found for c in ("red", "green", "yellow", "blue")) >= 2, found
