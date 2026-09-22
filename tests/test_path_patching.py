"""Path patching: a sender's effect through one receiver,
everything between them held clean."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mechbench_compute import paths
from mechbench_compute.ops.intervene.path import RECEIVER_POINTS
from mechbench_compute.ops.intervene.path import SENDER_POINTS
from mechbench_compute.ops.intervene.path import SpecError
from mechbench_compute.ops.intervene.path import _freeze_off_path
from mechbench_compute.ops.intervene.path import _parse_end
from mechbench_compute.ops.intervene.path import _collect_senders
from mechbench_compute.ops.intervene.path import run_path_patch as run

E2B = "mlx-community/gemma-4-e2b-it-bf16"
_real = pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1"
    or not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--" + E2B.replace("/", "--"))),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached",
)

PAIR = {"id": "capital", "template": "raw",
        "a": "The Eiffel Tower is in the city of",
        "b": "The Space Needle is in the city of",
        "tracked": {"answer": " Paris"}}


class _Arch:
    n_layers, n_heads = 6, 4


class _Model:
    arch = _Arch()


class TestWhatItRefuses:
    def test_a_sender_must_be_earlier_than_its_receiver(self):
        with pytest.raises(SpecError, match="must be earlier"):
            _collect_senders({"senders": {"layer": 4, "head": 1}},
                           {"point": "attn.q", "layer": 3, "head": 0}, 6, 4)

    def test_the_points_are_named(self):
        with pytest.raises(SpecError, match="receiver.point"):
            _parse_end({"point": "mlp_out", "layer": 2}, what="receiver",
                             points=RECEIVER_POINTS, n_layers=6, default_point="logits")
        with pytest.raises(SpecError, match="sender.point"):
            _parse_end({"point": "attn.q", "layer": 2}, what="sender",
                             points=SENDER_POINTS, n_layers=6, default_point="attn_out")
        with pytest.raises(SpecError, match="has no heads"):
            _parse_end({"point": "mlp_out", "layer": 2, "head": 1}, what="sender",
                             points=SENDER_POINTS, n_layers=6, default_point="attn_out")
        with pytest.raises(SpecError, match="names no layer"):
            _parse_end({"point": "attn.q"}, what="receiver",
                             points=RECEIVER_POINTS, n_layers=6, default_point="logits")

    def test_the_sweeps_reach_every_earlier_component(self):
        heads = _collect_senders({"senders": "all-heads"},
                               {"point": "attn.q", "layer": 3, "head": 0}, 6, 4)
        assert len(heads) == 3 * 4
        assert {s["point"] for s in heads} == {"attn.per_head_out"}
        assert max(s["layer"] for s in heads) == 2
        layers = _collect_senders({"senders": "all-layers"},
                                {"point": "logits", "layer": None}, 6, 4)
        assert len(layers) == 6 * 2 and {s["point"] for s in layers} == {"attn_out", "mlp_out"}

    def test_what_is_frozen_is_what_lies_between(self):
        cache = {f"blocks.{i}.{p}": i for i in range(6) for p in ("attn_out", "mlp_out")}
        hooks = _freeze_off_path(cache, {"point": "attn.per_head_out", "layer": 1, "head": 0},
                              {"point": "attn.q", "layer": 4, "head": 0}, 6)
        # The sender's own MLP (it is not on the path), then both
        # branches of every layer strictly between.
        assert sorted(hooks) == ["blocks.1.mlp_out", "blocks.2.attn_out", "blocks.2.mlp_out",
                                 "blocks.3.attn_out", "blocks.3.mlp_out"]
        # An MLP sender does not freeze itself.
        hooks = _freeze_off_path(cache, {"point": "mlp_out", "layer": 1, "head": None},
                              {"point": "logits", "layer": None}, 3)
        assert sorted(hooks) == ["blocks.2.attn_out", "blocks.2.mlp_out"]


@_real
class TestOnGemma:
    @pytest.fixture(scope="class")
    def model(self):
        from mechbench_compute import Model
        return Model.load(E2B)

    SENDERS = [{"point": "attn.per_head_out", "layer": 3, "head": 1},
               {"point": "mlp_out", "layer": 11},
               {"point": "attn_out", "layer": 19}]
    RECEIVER = {"point": "attn.q", "layer": 20, "head": 2}

    def test_a_pair_that_does_not_differ_moves_nothing(self, model):
        """The decisive check on the freezing: when the corrupt prompt IS
        the clean one, every patch writes the value that was already
        there, so nothing should move — however many components were
        frozen along the way. Two of the three land on exactly 0.0; the
        third is 0.125, which on a logit of ~16 is two bf16 ulps and the
        floor of what this arithmetic can say. A freeze of the wrong
        layer moves the answer by whole logits, which is what this
        catches."""
        same = {**PAIR, "b": PAIR["a"]}
        out = run(model, [same], {"receiver": self.RECEIVER,
                                        "senders": self.SENDERS, "metric": "logit"})
        assert max(abs(r["delta"]) for r in out["items"]) < 0.2

    def test_a_direct_path_to_the_answer_is_one_component_and_not_the_rest(self, model):
        """What path patching is for: with every other component frozen
        clean, almost nothing reaches the answer directly — the median
        sender moves it by exactly zero — and one does."""
        out = run(model, [PAIR], {"receiver": {"point": "logits"},
                                        "senders": "all-layers", "metric": "logit"})
        deltas = np.array([abs(r["delta"]) for r in out["items"]])
        assert float(np.median(deltas)) == 0.0
        assert deltas.max() > 1.0
        biggest = max(out["items"], key=lambda r: abs(r["delta"]))
        assert biggest["coords"]["layer"] >= model.arch.n_layers - 2

    def test_the_rows_name_both_ends(self, model):
        out = run(model, [PAIR],
                        {"receiver": {"point": "attn.k", "layer": 5, "head": 0},
                         "senders": "all-heads", "metric": "logit"})
        assert out["item_kind"] == "intervene/readout"
        assert out["n_senders"] == 5 * model.arch.n_heads == len(out["items"])
        r = out["items"][0]
        assert r["coords"]["into_layer"] == 5 and r["coords"]["into_point"] == "attn.k"
        assert r["coords"]["into_head"] == 0 and r["coords"]["point"] == "attn.per_head_out"
        assert r["cell"] == f"L{r['coords']['layer']}H{r['coords']['head']}:attn.per_head_out"
        assert out["target"]["text"] == " Paris"

    def test_a_direct_path_to_the_logits_is_the_plain_patch(self, model):
        """With nothing between the sender and the end, path patching and
        ordinary patching are the same intervention — which is what says
        the freezing is the only difference between them."""
        from mechbench_compute import intervene as iv

        last = model.arch.n_layers - 1
        out = run(model, [PAIR], {"receiver": {"point": "logits"},
                                        "senders": [{"point": "mlp_out", "layer": last}],
                                        "metric": "logit"})
        [row] = out["items"]

        # The same thing by hand: capture the corrupt mlp_out, patch it
        # into the clean run, read the target's logit.
        from mechbench_compute.interp import render_text
        from mechbench_compute.interventions import Capture
        import mlx.core as mx
        name = f"blocks.{last}.mlp_out"
        ids_a = render_text(model, PAIR, PAIR["a"])
        ids_b = render_text(model, PAIR, PAIR["b"])
        corrupt = model.run(ids_b, interventions=[Capture.at([name])]).cache[name]
        tok = int(mx.argmax(model.run(ids_a).logits[0, -1]))  # unused; the pair tracks its own
        target = out["target"]["id"]
        clean_logit = float(np.array(model.run(ids_a).logits[0, -1].astype(mx.float32))[target])
        patched = model.run(ids_a, hooks={name: lambda act, info: corrupt.astype(act.dtype)})
        want = float(np.array(patched.logits[0, -1].astype(mx.float32))[target]) - clean_logit
        assert abs(row["delta"] - want) < 0.02, (row["delta"], want)

    def test_most_heads_reach_one_heads_query_not_at_all(self, model):
        out = run(model, [PAIR], {"receiver": self.RECEIVER,
                                        "senders": "all-heads", "metric": "logit"})
        deltas = np.array([abs(r["delta"]) for r in out["items"]])
        assert len(deltas) == 20 * model.arch.n_heads
        assert float(np.median(deltas)) == 0.0
