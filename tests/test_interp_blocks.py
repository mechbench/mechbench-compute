"""The interp primitives (mechbench-experiments port), on a stub model.

The stub's residuals are ONE-HOT in the token id (scaled by layer+1),
so every geometric assertion is exact: identical tokens → cosine 1,
different tokens → cosine 0. Ablation is simulated by damping the
favored token's logit by (layer+1), so deltas are computable in the
test with the same softmax arithmetic.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute.ops.activations import capture as capture_op
from mechbench_compute.ops.activations import capture_attention as capture_attention_op
from mechbench_compute.ops.activations import capture_tokens as capture_tokens_op
from mechbench_compute import interp
from mechbench_compute.ops.intervene.patch import patch_trace
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.ops.activations.capture_attention import capture_attention_patterns
from mechbench_compute.ops.activations.capture_tokens import capture_tokens
from mechbench_compute.ops.activations.contrast import measure_residual_divergence
from mechbench_compute.ops.intervene.ablate_heads import ablate_heads
from mechbench_compute.ops.intervene.ablate_layers import ablate_layers
from mechbench_compute.ops.intervene.steer import steer_inject
from mechbench_compute.ops.logits.attribute import attribute_logits
from mechbench_compute.ops.logits.scan import scan_positions

N_LAYERS = 4
D_MODEL = 8
VOCAB = 12
FAV = 7  # the token the stub's baseline favors


class StubArch:
    n_layers = N_LAYERS
    d_model = D_MODEL
    n_heads = 2


class StubTokenizer:
    """Word-per-token: each word maps to id 1 + (len(word) % 7), after a
    BOS of 0. The chat template is the identity, so a condition and a
    raw record tokenize alike and the arithmetic below holds for both."""

    all_special_ids = (0,)

    def encode(self, text, add_special_tokens=True):
        return [0] + [1 + (len(w) % 7) for w in text.split()]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages[-1]["content"]

    def decode(self, ids):
        return " ".join(f"t{int(i)}" for i in ids)


class StubModel:
    """Word-per-token: each word maps to id 1 + (len(word) % 7)."""

    arch = StubArch()
    tokenizer = StubTokenizer()

    def __init__(self):
        self.runs = 0
        #: When set, a patch at position 1 answers as if the second
        #: token were this id — the stub's model of causal tracing.
        self.clean_second: int | None = None

    def tokenize(self, prompt: str, chat_template: bool = True):
        ids = [0] + [1 + (len(w) % 7) for w in prompt.split()]
        return mx.array([ids])

    def run(self, ids, interventions=None):
        self.runs += 1
        arr = np.array(ids)[0]
        seq = len(arr)
        # ablation penalty: sum of (layer+1) across ablated layers,
        # detected from the intervention objects' own fields
        penalty = 0.0
        patched_positions = []
        for iv in interventions or []:
            if hasattr(iv, "position") and hasattr(iv, "value"):
                patched_positions.append(int(iv.position))
                continue
            if hasattr(iv, "head"):
                penalty += (iv.layer_idx + 1) + iv.head / 10.0
                continue
            layer_idx = getattr(iv, "layer_idx", None)
            if layer_idx is not None:
                penalty += layer_idx + 1
            for name in getattr(iv, "names", ()) or ():
                if name.endswith(("attn_out", "mlp_out", "gate_out")):
                    penalty += int(name.split(".")[1]) + 1

        logits = np.zeros((1, seq, VOCAB), dtype=np.float32)
        fav = FAV
        if seq > 1:
            second = int(arr[1])
            if 1 in patched_positions and self.clean_second is not None:
                second = self.clean_second
            fav = second % VOCAB
        logits[0, -1, fav] = 5.0 - penalty

        cache = {}
        n_heads = self.arch.n_heads
        for layer in range(N_LAYERS):
            w = np.zeros((1, n_heads, seq, seq), dtype=np.float32)
            for h in range(n_heads):
                for i2 in range(seq):
                    w[0, h, i2, : i2 + 1] = 1.0 / (i2 + 1)
            cache[f"blocks.{layer}.attn.weights"] = mx.array(w)
        for layer in range(N_LAYERS):
            resid = np.zeros((1, seq, D_MODEL), dtype=np.float32)
            for pos, tok in enumerate(arr):
                resid[0, pos, int(tok) % D_MODEL] = float(layer + 1)
            cache[f"blocks.{layer}.resid_post"] = mx.array(resid)
        pre0 = np.zeros((1, seq, D_MODEL), dtype=np.float32)
        for pos, tok in enumerate(arr):
            pre0[0, pos, int(tok) % D_MODEL] = 0.5
        cache["blocks.0.resid_pre"] = mx.array(pre0)
        cache["final_norm.scale"] = mx.array(
            np.ones((1, seq), dtype=np.float32))

        class R:
            pass

        r = R()
        r.logits = mx.array(logits)
        r.cache = cache
        return r


def _expected_delta(layer: int) -> float:
    def logp(fav_logit: float) -> float:
        z = math.log(math.exp(fav_logit) + (VOCAB - 1) * math.exp(0.0))
        return fav_logit - z

    return logp(5.0 - (layer + 1)) - logp(5.0)


class TestAblateLayers:
    def test_deltas_match_the_softmax_arithmetic_exactly(self):
        model = StubModel()
        out = ablate_layers(
            model, [{"id": "c1", "user": "the tower is in"}], {})
        assert out["item_kind"] == "intervene/ablation"
        assert out["layers"] == [0, 1, 2, 3]
        deltas = {r["layer"]: r["delta_logp"] for r in out["items"]}
        assert all(r["layer"] is not None for r in out["items"])
        assert out["conditions"][0]["id"] == "c1" and "baseline_logp" in out["conditions"][0]
        for layer in range(N_LAYERS):
            assert deltas[layer] == pytest.approx(
                _expected_delta(layer), abs=1e-3)
        # deeper stub layers are damped harder — the sweep must say so
        assert deltas[3] < deltas[0] < 0

    def test_progress_covers_every_forward(self):
        model = StubModel()
        ticks = []
        ablate_layers(
            model, [{"id": "a", "user": "x y"}, {"id": "b", "user": "p q"}],
            {"layers": [1, 2]},
            on_item=lambda: ticks.append(1), on_start=lambda n: ticks.append(n))
        assert ticks[0] == 2 * 3  # (baseline + 2 layers) per condition
        assert sum(t for t in ticks[1:]) == 6
        assert model.runs == 6

    def test_a_named_target_wins_over_top1(self):
        model = StubModel()
        out = ablate_layers(
            model, [{"id": "c", "user": "a b", "target": "word"}], {"layers": [0]})
        meta = out["conditions"][0]
        assert meta["target"]["id"] == 1 + (len("word") % 7)
        assert meta["target"]["text"] == f"t{1 + (len('word') % 7)}"

    def test_sublayer_components_route_to_their_hooks(self):
        model = StubModel()
        out = ablate_layers(
            model, [{"id": "c", "user": "a b"}],
            {"component": "mlp", "layers": [2]})
        row = next(r for r in out["items"] if r.get("layer") == 2)
        assert row["delta_logp"] == pytest.approx(_expected_delta(2), abs=1e-3)

    def test_a_point_that_is_not_a_sublayer_output_refuses(self):
        with pytest.raises(ValueError, match="unknown point"):
            ablate_layers(StubModel(), [{"id": "c", "user": "a"}],
                                 {"point": "norm"})
        with pytest.raises(ValueError, match="sub-layer output"):
            ablate_layers(StubModel(), [{"id": "c", "user": "a"}],
                                 {"point": "resid_post"})

    def test_the_default_zeroes_the_whole_layer(self):
        out = ablate_layers(StubModel(), [{"id": "c", "user": "a"}], {"layers": [0]})
        assert out["points"] == ["attn_out", "mlp_out"]

    def test_a_sweep_at_a_prefilled_decision_point_reads_where_a_decision_read_does(self):
        """One rendering, two ops, one number: the sweep's baseline log-prob
        for a prefilled condition is the last-position log-prob of the same
        rendering — what `logits/read` reports for that condition."""
        from mechbench_compute.distill import render
        from mechbench_compute.interp import read_last_logp

        model = StubModel()
        cond = {"id": "c", "system": "s", "user": "roll the die", "prefill": '{ "roll": ',
                "tracked": {"three": "ccc"}}
        out = ablate_layers(model, [cond], {"layers": [0]})
        r = render(model, cond)
        assert r.chat and r.text.endswith('{ "roll": ')
        lp = read_last_logp(model.run(r.array).logits)
        tok = 1 + (len("ccc") % 7)
        assert out["conditions"][0]["target"]["id"] == tok
        assert out["conditions"][0]["baseline_logp"] == pytest.approx(float(lp[tok]), abs=1e-4)


class TestRenderingIsOnTheResult:
    """A result says how its prompts reached the model.

    A record with only `text` renders RAW, with no chat template, and
    an instruct model completing raw text answers with function words.
    The only trace of that in the numbers is a target token reading
    " the", so the condition says `template: "raw"` in as many words.
    """

    def test_ablation_records_raw_and_chat(self):
        model = StubModel()
        out = ablate_layers(
            model,
            [{"id": "doc", "text": "aa bbb"},
             {"id": "cond", "user": "aa bbb"}],
            {"layers": [0]})
        by_id = {c["id"]: c["template"] for c in out["conditions"]}
        assert by_id == {"doc": "raw", "cond": "chat"}

    def test_template_chat_on_a_text_record_renders_as_chat(self):
        model = StubModel()
        out = ablate_layers(
            model, [{"id": "doc", "text": "aa bbb", "template": "chat"}],
            {"layers": [0]})
        assert out["conditions"][0]["template"] == "chat"


class TestOwnTop1BesideATarget:
    """A tracked target that is not the model's answer is reported
    beside the model's answer."""

    def test_a_target_the_model_would_not_say_is_flagged(self):
        model = StubModel()
        # The stub's top-1 for "aa bbb" is id 3; track "q" (id 2) instead.
        out = ablate_layers(
            model, [{"id": "c", "user": "aa bbb", "tracked": {"x": "q"}}],
            {"layers": [0]})
        c = out["conditions"][0]
        assert "own_top1" in c and c["own_top1"]["id"] != c["target"]["id"]
        assert "logp" in c["own_top1"]
        # …and the header counts them, so a sweep over the wrong spelling
        # announces itself at the top.
        assert out["n_off_top1"] == 1

    def test_the_models_own_answer_is_not_flagged_against_itself(self):
        model = StubModel()
        out = ablate_layers(model, [{"id": "c", "user": "aa bbb"}], {"layers": [0]})
        assert "own_top1" not in out["conditions"][0]
        assert out["n_off_top1"] == 0


class TestResidualVectors:
    def test_vectors_are_the_positions_residual(self):
        model = StubModel()
        out = capture_residual_vectors(
            model, [{"id": "c", "user": "aa bbb", "label": "en"}],
            {"layers": [1], "position": "final"})
        row = out["items"][0]
        # A top-level `label` field becomes the `label` coordinate.
        assert row["coords"] == {"label": "en"} and "label" not in row
        assert row["space"] == {"model": None, "layer": 1, "point": "resid_post", "head": None, "d": D_MODEL}
        # final token of "aa bbb" is id 1+(3%7)=4; layer 1 scale = 2
        assert row["token"] == {"id": 4, "text": "t4"}
        v = np.array(row["vector"])
        assert v[4] == pytest.approx(2.0)
        assert np.count_nonzero(v) == 1
        assert row["norm"] == pytest.approx(2.0)

    def test_every_item_carries_the_records_coords(self):
        # A grouping is a coordinate; the grouping ops name it by `axis`.
        model = StubModel()
        out = capture_residual_vectors(
            model, [{"id": "c", "user": "a", "coords": {"language": "fr"}}],
            {"layers": [0]})
        assert out["items"][0]["coords"] == {"language": "fr"}

    def test_the_float_cap_refuses_a_runaway_capture(self, monkeypatch):
        monkeypatch.setattr(capture_op, "MAX_VECTOR_FLOATS", 10)
        with pytest.raises(ValueError, match="cap"):
            capture_residual_vectors(
                StubModel(), [{"id": "c", "user": "a"}], {"layers": "all"})


class TestCaptureTokens:
    """One vector per token, each carrying its own surprisal."""

    def test_a_row_per_position_per_layer(self):
        model = StubModel()
        out = capture_tokens(
            model, [{"id": "c", "user": "aa bbb"}],
            {"layers": [0, 1], "positions": "all"})
        items = out["items"]
        # "aa bbb" renders to 3 tokens (a leading 0), × 2 layers.
        assert len(items) == 6
        assert sorted({i["coords"]["position"] for i in items}) == [0, 1, 2]
        assert sorted({i["space"]["layer"] for i in items}) == [0, 1]

    def test_surprisal_rides_on_the_vector_and_position_zero_has_none(self):
        model = StubModel()
        out = capture_tokens(
            model, [{"id": "c", "user": "aa bbb"}], {"layers": [0]})
        by_pos = {i["coords"]["position"]: i["coords"] for i in out["items"]}
        # A join by (record, position) afterwards is where an off-by-one
        # would creep in; the coordinate is carried, not matched later.
        assert "surprisal" not in by_pos[0]
        assert isinstance(by_pos[1]["surprisal"], float)

    def test_positions_narrow_and_every_subsamples(self):
        model = StubModel()
        after = capture_tokens(
            model, [{"id": "c", "user": "aa bbb ccc dddd"}],
            {"layers": [0], "positions": {"after": 2}})
        assert sorted({i["coords"]["position"] for i in after["items"]}) == [2, 3, 4]
        every = capture_tokens(
            model, [{"id": "c", "user": "aa bbb ccc dddd"}],
            {"layers": [0], "every": 2})
        assert sorted({i["coords"]["position"] for i in every["items"]}) == [0, 2, 4]

    def test_the_ceiling_refuses_json_and_names_the_levers(self, monkeypatch):
        monkeypatch.setattr(capture_tokens_op, "MAX_TOKEN_VECTOR_FLOATS", 10)
        with pytest.raises(ValueError, match="cap .* capture fewer layers"):
            capture_tokens(StubModel(), [{"id": "c", "user": "a b c"}],
                                  {"layers": "all", "storage": "json"})

    def test_above_the_ceiling_auto_writes_shards(self, monkeypatch):
        # The rows go to shards beside the object: the result
        # is the header, its items empty, and read back through
        # items_of they are the same rows the json form would carry.
        from mechbench_compute import tensors
        from mechbench_compute.lexicon import kinds as K

        monkeypatch.setattr(capture_tokens_op, "MAX_TOKEN_VECTOR_FLOATS", 10)
        out = capture_tokens(StubModel(), [{"id": "c", "user": "aa bbb"}], {"layers": [0, 1]})
        assert tensors.is_tensor(out) and out["items"] == [] and out["n_items"] == 6
        assert [s["rows"] for s in out["shards"]] == [6] and out["d"] == D_MODEL
        rows = list(K.items_of(out))
        monkeypatch.setattr(capture_tokens_op, "MAX_TOKEN_VECTOR_FLOATS", 10_000)
        plain = capture_tokens(StubModel(), [{"id": "c", "user": "aa bbb"}],
                                      {"layers": [0, 1], "storage": "json"})["items"]
        assert len(rows) == len(plain) == 6
        for a, b in zip(rows, plain):
            assert a["id"] == b["id"] and a["space"] == b["space"] and a["token"] == b["token"]
            assert a["coords"] == b["coords"]
            assert np.allclose(a["vector"], b["vector"])

    def test_one_forward_pass_per_record(self):
        # The logits come from the capture's own run. A second pass would
        # be slower and could disagree with the vectors it labels.
        model = StubModel()
        capture_tokens(model, [{"id": "c", "user": "aa bbb"}], {"layers": [0]})
        assert model.runs == 1


class TestPooledPositions:
    """Pooling over the sequence.

    The stub puts `layer + 1` at dimension `token_id % D_MODEL` for
    each position, so "aa bbb" at layer 1 is three one-hot rows of 2.0
    at dims 0, 3 and 4 — which makes every pooled value checkable by
    hand rather than by re-running the implementation."""

    RECORD = {"id": "c", "user": "aa bbb", "label": "en"}

    def _vec(self, **params):
        out = capture_residual_vectors(
            StubModel(), [dict(self.RECORD)], {"layers": [1], **params})
        return out, np.array(out["items"][0]["vector"])

    def test_mean_pools_the_whole_sequence(self):
        out, v = self._vec(pool={"reduce": "mean", "over": "all"})
        for dim in (0, 3, 4):
            assert v[dim] == pytest.approx(2.0 / 3.0, abs=1e-4)
        assert np.count_nonzero(v) == 3
        assert out["items"][0]["n_pooled"] == 3

    def test_a_window_after_the_first_position(self):
        out, v = self._vec(pool={"reduce": "mean", "over": {"after": 1}})
        assert v[0] == pytest.approx(0.0)
        for dim in (3, 4):
            assert v[dim] == pytest.approx(1.0, abs=1e-4)
        assert out["items"][0]["n_pooled"] == 2

    def test_pooling_the_last_position_reproduces_the_single_read(self):
        # The compatibility anchor: pooling one position must equal the
        # unpooled read, or the two paths have drifted apart.
        _, pooled = self._vec(pool={"reduce": "mean", "over": {"range": [-1, None]}})
        _, single = self._vec(position="last")
        assert pooled.tolist() == single.tolist()
        _, listed = self._vec(pool={"reduce": "mean", "over": [-1]})
        assert listed.tolist() == single.tolist()

    def test_max_takes_the_elementwise_maximum(self):
        _, v = self._vec(pool={"reduce": "max", "over": "all"})
        for dim in (0, 3, 4):
            assert v[dim] == pytest.approx(2.0)

    def test_the_record_says_how_it_was_made(self):
        out, _ = self._vec(pool={"reduce": "mean", "over": {"range": [1, 3]}})
        assert out["position"] == "pooled"
        assert out["pool"] == {"reduce": "mean", "over": {"range": [1, 3]}}
        assert out["items"][0]["n_pooled"] == 2

    def test_no_pool_is_untouched(self):
        # Adding the parameter must not change a single number in a
        # record made without it — published geometry depends on this.
        out, v = self._vec(position="last")
        assert out["position"] == "last"
        assert "pool" not in out
        assert "n_pooled" not in out["items"][0]
        assert v[4] == pytest.approx(2.0)
        assert np.count_nonzero(v) == 1

    def test_a_window_past_the_end_falls_back_to_the_last_position(self):
        # Zeros would look like a vector and mean nothing.
        out, v = self._vec(pool={"reduce": "mean", "over": {"after": 99}})
        assert out["items"][0]["n_pooled"] == 1
        assert v[4] == pytest.approx(2.0)

    def test_pooling_needs_no_resolvable_position(self):
        # `subject` would raise without a `subject` field; pooling
        # never resolves a single position, so it must not.
        out = capture_residual_vectors(
            StubModel(), [{"id": "c", "user": "aa bbb"}],
            {"layers": [1], "position": "subject", "pool": {"reduce": "mean", "over": "all"}})
        assert out["items"][0]["n_pooled"] == 3

    def test_a_bad_pool_refuses(self):
        with pytest.raises(ValueError, match="unknown pool reduce"):
            self._vec(pool={"reduce": "median", "over": "all"})

    def test_the_retired_pool_string_is_refused_with_the_new_form(self):
        with pytest.raises(ValueError, match="range"):
            self._vec(pool="last_k")

    def test_a_bad_selector_refuses(self):
        with pytest.raises(ValueError, match="unknown positions"):
            self._vec(pool={"reduce": "mean", "over": "middle"})


class TestResidualDivergence:
    def test_identical_pair_diverges_nowhere(self):
        model = StubModel()
        out = measure_residual_divergence(
            model, [{"id": "p", "a": "over the hill", "b": "over the hill"}],
            {"layers": [0, 1]})
        pair = out["items"][0]
        assert pair["axes"] == ["layer", "position"]
        flat = [x for row in pair["measures"]["divergence"] for x in row]
        assert all(x == pytest.approx(0.0, abs=1e-4) for x in flat)

    def test_a_one_word_swap_diverges_exactly_there(self):
        model = StubModel()
        # 'over'(4) vs 'under'(5) -> ids differ at position 1 only
        out = measure_residual_divergence(
            model, [{"id": "p", "a": "go over it", "b": "go under it"}],
            {"layers": [0]})
        div = out["items"][0]["measures"]["divergence"][0]
        assert div[0] == pytest.approx(0.0, abs=1e-4)  # BOS
        assert div[1] == pytest.approx(0.0, abs=1e-4)  # 'go'
        assert div[2] == pytest.approx(1.0, abs=1e-4)  # the swapped word
        assert div[3] == pytest.approx(0.0, abs=1e-4)  # 'it'

    def test_unequal_lengths_report_instead_of_lying(self):
        model = StubModel()
        out = measure_residual_divergence(
            model, [{"id": "p", "a": "one two", "b": "one two three"}],
            {"layers": [0]})
        assert "different lengths" in out["items"][0]["error"]


class TestVectorSimilarity:
    def _vectors_record(self):
        # two tight clusters along different axes
        rows = []
        for i, label in enumerate(["cat", "cat", "dog", "dog"]):
            v = [0.0] * 4
            v[0 if label == "cat" else 2] = 1.0
            v[1 if label == "cat" else 3] = 0.1 * i
            rows.append({"id": f"r{i}", "label": label, "layer": 5, "vector": v})
        from mechbench_compute.lexicon import kinds as K

        return K.collection("activations/vector", rows, layers=[5],
                            position="final", point="post")

    def test_matrix_and_separation(self):
        from mechbench_compute.ops.geometry.compare import compare_geometry

        out = compare_geometry({"items": self._vectors_record()}, {})
        assert out["item_kind"] == "geometry/similarity"
        assert out["metric"] == "cosine" and out["metric_kind"] == "similarity"
        layer = out["items"][0]
        assert layer["layer"] == 5 and layer["group"] == "layer=5"
        m = np.array(layer["matrix"])
        assert m.shape == (4, 4)
        assert m[0, 1] > m[0, 2]  # same-label closer than cross-label
        assert layer["separation"]["gap"] > 0
        assert layer["nn_purity"] == pytest.approx(1.0)

    def test_unlabeled_vectors_still_get_a_matrix(self):
        from mechbench_compute.ops.geometry.compare import compare_geometry

        rec = self._vectors_record()
        for r in rec["items"]:
            r["label"] = None
        out = compare_geometry({"items": rec}, {})
        assert "separation" not in out["items"][0]

    def test_wrong_input_kind_refuses(self):
        from mechbench_compute.ops.geometry.compare import compare_geometry

        with pytest.raises(ValueError, match="declares metrics"):
            compare_geometry({"items": {"kind": "word_list"}}, {})


class TestGateComponent:
    def test_gate_routes_to_the_side_channel_hook(self):
        model = StubModel()
        out = ablate_layers(
            model, [{"id": "c", "user": "a b"}],
            {"point": "gate_out", "layers": [1]})
        assert out["points"] == ["gate_out"]
        row = next(r for r in out["items"] if r.get("layer") == 1)
        assert row["delta_logp"] < 0  # the stub penalizes any named zero-hook


class TestLensPositions:
    def test_the_target_surfaces_where_its_token_sits(self, monkeypatch):
        model = StubModel()
        # give the stub the unembedding the lens needs: identity over
        # the one-hot residual dims
        def project(resid):
            arr = np.array(resid.astype(mx.float32))
            out = np.zeros((*arr.shape[:-1], VOCAB), dtype=np.float32)
            out[..., :D_MODEL] = arr
            return mx.array(out)

        model.project_to_logits = project
        out = scan_positions(
            model, [{"id": "c", "user": "aa bbb aa", "target": "bbb"}],
            {"layers": [0, 1]})
        row = out["items"][0]
        assert out["item_kind"] == "logits/lens"
        assert row["axes"] == ["layer", "position"] and row["target"]["id"] == 4
        # 'bbb' -> id 4; it sits at position 2 (BOS, aa, bbb, aa)
        ranks = np.array(row["measures"]["rank"])
        lps = np.array(row["measures"]["logprob"])
        assert ranks[0, 2] == 0  # top readout exactly where the token is
        assert ranks[0, 1] > 0  # and not where it is not
        assert lps[0, 2] > lps[0, 1]


class TestPatchTrace:
    def test_logprob_metric_registers_low_mass_targets(self):
        model = StubModel()
        model.clean_second = 1 + (len("over") % 7)
        out = patch_trace(
            model, [{"id": "p", "clean": "over the hill",
                     "corrupt": "under the hill"}],
            {"layers": [0], "metric": "logprob"})
        pair = out["items"][0]
        assert pair["metric"] == "logprob"
        rec = np.array(pair["measures"]["recovery"])
        assert rec[0, 1] > 1.0  # log-space recovery is loud
        assert abs(rec[0, 0]) < 1e-3

    def test_unknown_metric_refuses(self):
        with pytest.raises(ValueError, match="metric"):
            patch_trace(
                StubModel(), [{"id": "p", "clean": "a", "corrupt": "b"}],
                {"metric": "vibes"})

    def test_recovery_lands_exactly_on_the_differing_position(self):
        model = StubModel()
        # prompts whose difference sits exactly at position 1 (after
        # BOS), where the stub's flip logic looks
        clean, corrupt = "over the hill", "under the hill"
        model.clean_second = 1 + (len("over") % 7)
        out = patch_trace(
            model, [{"id": "p", "clean": clean, "corrupt": corrupt}],
            {"layers": [0, 1], "metric": "prob"})
        pair = out["items"][0]
        rec = np.array(pair["measures"]["recovery"])  # [layer][pos]
        assert rec.shape[1] == 4  # BOS + 3 words
        # patching the differing position recovers the clean answer fully
        assert rec[0, 1] > 0.5
        # patching agreeing positions recovers nothing
        assert abs(rec[0, 0]) < 1e-3
        assert abs(rec[0, 3]) < 1e-3
        assert pair["value_a"] > pair["value_b"]
        assert pair["target"]["text"] == "t5"

    def test_a_and_b_are_the_pair_fields(self):
        model = StubModel()
        model.clean_second = 1 + (len("over") % 7)
        out = patch_trace(
            model, [{"id": "p", "a": "over the hill", "b": "under the hill"}],
            {"layers": [0], "metric": "prob"})
        assert np.array(out["items"][0]["measures"]["recovery"])[0, 1] > 0.5

    def test_unequal_pairs_report(self):
        out = patch_trace(
            StubModel(), [{"id": "p", "clean": "a b", "corrupt": "a b c"}],
            {"layers": [0]})
        assert "different lengths" in out["items"][0]["error"]


class TestAttentionPatterns:
    def test_shapes_and_row_normalization(self):
        model = StubModel()
        out = capture_attention_patterns(
            model, [{"id": "c", "user": "a b c"}], {"layers": [1]})
        row = out["items"][0]
        assert row["axes"] == ["layer", "head", "query", "key"]
        heads = row["measures"]["weight"][0]
        assert len(heads) == 2  # stub n_heads
        m = np.array(heads[0])
        assert m.shape == (4, 4)
        assert np.allclose(m.sum(axis=1), 1.0, atol=1e-3)  # causal rows sum to 1

    def test_all_layers_refuses_loudly(self):
        with pytest.raises(ValueError, match="explicit layers"):
            capture_attention_patterns(
                StubModel(), [{"id": "c", "user": "a"}], {"layers": "all"})

    def test_the_float_cap_refuses(self, monkeypatch):
        monkeypatch.setattr(capture_attention_op, "MAX_ATTN_FLOATS", 3)
        with pytest.raises(ValueError, match="floats"):
            capture_attention_patterns(
                StubModel(), [{"id": "c", "user": "a b"}], {"layers": [0]})


class TestAblateHeads:
    def test_the_head_matrix_matches_the_stub_arithmetic(self):
        model = StubModel()
        out = ablate_heads(
            model, [{"id": "c", "user": "a"}], {"layers": [0, 2]})
        assert out["kind"] == "intervene/heads" and out["axes"] == ["layer", "head"]
        m = np.array(out["measures"]["mean_delta"])  # [2 layers][2 heads]
        assert out["conditions"][0]["target"]["text"] == "t2"
        assert m.shape == (2, 2)
        # stub penalty = (layer+1) + head/10: deeper layer and higher
        # head both hurt more
        assert m[1, 0] < m[0, 0] < 0
        assert m[0, 1] < m[0, 0]
        assert out["n_heads"] == 2


class TestSubjectPosition:
    def test_the_longest_hit_beats_a_stray_short_one(self):
        model = StubModel()

        class Tok(StubTokenizer):
            def decode(self, ids):
                # word-length-keyed vocabulary: id 5 -> 'casa', id 2 -> 'a'
                names = {5: "casa", 2: "a"}
                return " ".join(names.get(int(i), f"t{int(i)}") for i in ids)

        model.tokenizer = Tok()
        # prompt 'casa xx a': ids [0, 1+(4%7)=5, 1+(2%7)=3, 1+(1%7)=2]
        out = capture_residual_vectors(
            model, [{"id": "c", "user": "casa xx a", "subject": "casa"}],
            {"layers": [0], "position": "subject"})
        v = np.array(out["items"][0]["vector"])
        # position of 'casa' (pos 1, token id 5) — not the stray 'a' at pos 3
        assert v[5] == pytest.approx(1.0)


class TestLogitAttribution:
    def test_component_bookkeeping_and_the_honesty_number(self, monkeypatch):
        from mechbench_compute import attribution

        def fake_attrs(model, stack, targets, *, position=-1,
                       apply_ln=False, ln_scale=None):
            assert apply_ln and ln_scale is not None
            at = stack[..., position, :]
            return np.stack([at[..., t % at.shape[-1]] for t in targets],
                            axis=-1).astype(np.float32)

        monkeypatch.setattr(attribution, "logit_attrs", fake_attrs)
        model = StubModel()
        out = attribute_logits(
            model, [{"id": "c", "user": "a b", "target": "word"}], {})
        assert out["item_kind"] == "logits/attribution"
        row = out["items"][0]
        # embedding + one component per layer
        assert row["axes"] == ["component"]
        assert len(row["measures"]["contribution"]) == N_LAYERS + 1
        assert out["components"][0] == "embed"
        assert row["target"]["id"] == 1 + (len("word") % 7)
        add = row["additivity"]
        assert set(add) == {"summed", "true_logit", "residual"}
        assert add["residual"] == pytest.approx(
            add["summed"] - add["true_logit"], abs=1e-3)

    def test_partial_layers_refuse_because_additivity_would_lie(self):
        with pytest.raises(ValueError, match="all"):
            attribute_logits(
                StubModel(), [{"id": "c", "user": "a"}], {"layers": [1, 2]})


class TestSteerInject:
    def _vectors(self):
        rows = []
        for label, dim in (("city", 1), ("city", 1), ("money", 3), ("money", 3)):
            v = [0.0] * D_MODEL
            v[dim] = 2.0
            rows.append({"id": f"{label}{dim}", "label": label,
                         "layer": 2, "vector": v})
        return {"kind": "residual_vectors", "layers": [2], "rows": rows}

    def test_direction_comes_from_the_data_and_alpha_zero_is_control(self):
        model = StubModel()
        seen = []
        orig = model.run

        def spy(ids, interventions=None):
            seen.append(list(interventions or []))
            return orig(ids, interventions=interventions)

        model.run = spy
        out = steer_inject(
            model, [{"id": "e", "user": "the capital was"}],
            {"layer": 2, "alphas": [0.0, 4.0],
             "direction": {"positive": "city", "negative": "money"}},
            inputs={"vectors": self._vectors()})
        assert out["item_kind"] == "intervene/readout"
        assert out["direction"]["norm"] == pytest.approx(
            float(np.linalg.norm([0, 2, 0, -2] + [0] * (D_MODEL - 4))),
            abs=1e-3)
        assert seen[0] == []  # alpha 0 runs clean
        add = seen[1][0]
        assert getattr(add, "alpha", None) == 4.0
        assert getattr(add, "layer_idx", None) == 2
        rows = out["items"]
        assert [r["factor"] for r in rows] == [0.0, 4.0]
        assert out["sweep"] == {"strength": [0.0, 4.0]}
        assert out["direction"]["axis"] == "label"
        assert len(rows[0]["top"]) == 5
        assert set(rows[0]["top"][0]) == {"token", "p", "logp"}

    def test_missing_layer_in_vectors_refuses_with_directions(self):
        model = StubModel()
        with pytest.raises(ValueError, match="capture that layer"):
            steer_inject(
                model, [{"id": "e", "user": "x"}],
                {"layer": 3, "direction": {"positive": "city", "negative": "money"}},
                inputs={"vectors": self._vectors()})

    def test_no_vectors_port_refuses(self):
        with pytest.raises(ValueError, match="vectors"):
            steer_inject(
                StubModel(), [{"id": "e", "user": "x"}],
                {"layer": 2, "direction": {"positive": "a", "negative": "b"}},
                inputs={})


class TestPerHeadDla:
    def test_per_head_rows_appear_for_named_layers(self, monkeypatch):
        from mechbench_compute import attribution

        def fake_attrs(model, stack, targets, *, position=-1,
                       apply_ln=False, ln_scale=None):
            at = stack[..., position, :]
            return np.stack([at[..., t % at.shape[-1]] for t in targets],
                            axis=-1).astype(np.float32)

        def fake_heads(model, cache, layer):
            return np.ones((2, 3, D_MODEL), dtype=np.float32) * (layer + 1)

        monkeypatch.setattr(attribution, "logit_attrs", fake_attrs)
        monkeypatch.setattr(attribution, "head_results", fake_heads)
        out = attribute_logits(
            StubModel(), [{"id": "c", "user": "a b", "target": "word"}],
            {"per_head_layers": [1]})
        row = out["items"][0]
        assert row["per_head"][0]["layer"] == 1
        assert len(row["per_head"][0]["contributions"]) == 2


class TestSubjectCase:
    def test_sentence_initial_capitalization_still_matches(self):
        model = StubModel()

        class Tok(StubTokenizer):
            def decode(self, ids):
                names = {5: "Capi", 3: "xx"}
                return " ".join(names.get(int(i), f"t{int(i)}") for i in ids)

        model.tokenizer = Tok()
        out = capture_residual_vectors(
            model, [{"id": "c", "user": "Capi xx", "subject": "capital"}],
            {"layers": [0], "position": "subject"})
        v = np.array(out["items"][0]["vector"])
        assert v[5] == pytest.approx(1.0)


class TestQKSources:
    def test_queries_emit_per_head_rows(self):
        model = StubModel()
        # give the stub per-head q/k caches: one-hot in head index
        orig = model.run

        def with_qk(ids, interventions=None):
            r = orig(ids, interventions=interventions)
            arr = np.array(ids)[0]
            seq = len(arr)
            hd = D_MODEL // 2
            for layer in range(N_LAYERS):
                q = np.zeros((1, 2, seq, hd), dtype=np.float32)
                for h in range(2):
                    q[0, h, :, h] = float(layer + 1)
                r.cache[f"blocks.{layer}.attn.q"] = mx.array(q)
                r.cache[f"blocks.{layer}.attn.k"] = mx.array(q * 0.5)
            return r

        model.run = with_qk
        model.arch.n_kv_heads = 2
        out = capture_residual_vectors(
            model, [{"id": "c", "user": "a b", "label": "x"}],
            {"layers": [1], "source": "queries"})
        assert out["source"] == "queries"
        heads = sorted(r["space"]["head"] for r in out["items"])
        assert heads == [0, 1]
        assert out["items"][0]["space"]["point"] == "attn.q"
        v0 = np.array(out["items"][0]["vector"])
        assert v0[0] == pytest.approx(2.0)  # head 0, layer 1 scale

    def test_similarity_groups_by_layer_and_head(self):
        rows = []
        for head in (0, 1):
            for i, label in enumerate(["a", "a", "b", "b"]):
                v = [0.0] * 4
                v[(0 if label == "a" else 2) + head % 2] = 1.0
                rows.append({"id": f"h{head}r{i}", "label": label,
                             "layer": 7, "head": head, "vector": v})
        from mechbench_compute.ops.geometry.compare import compare_geometry

        out = compare_geometry(
            {"items": {"kind": "residual_vectors", "rows": rows}}, {})
        entries = out["items"]
        assert len(entries) == 2
        assert {e["head"] for e in entries} == {0, 1}
        assert {e["group"] for e in entries} == {"layer=7,head=0", "layer=7,head=1"}
        assert all(e["layer"] == 7 for e in entries)
        assert all(len(e["ids"]) == 4 for e in entries)

    def test_unknown_source_refuses(self):
        with pytest.raises(ValueError, match="source"):
            capture_residual_vectors(
                StubModel(), [{"id": "c", "user": "a"}],
                {"layers": [0], "source": "values"})


class TestSteerTracks:
    def test_named_tracks_report_per_alpha(self):
        model = StubModel()
        rows = []
        for label, dim in (("a", 1), ("b", 3)):
            v = [0.0] * D_MODEL
            v[dim] = 1.0
            rows.append({"id": label, "label": label, "layer": 2, "vector": v})
        out = steer_inject(
            model, [{"id": "e", "user": "x y",
                     "tracks": {"city": "word", "money": "cash"}}],
            {"layer": 2, "alphas": [0.0],
             "direction": {"positive": "a", "negative": "b"}},
            inputs={"vectors": {"kind": "residual_vectors", "rows": rows}})
        row = out["items"][0]
        assert set(row["tracked"]) == {"city", "money"}
        assert isinstance(row["tracked"]["city"]["logp"], float)
        assert row["tracked"]["city"]["token"]["text"] == "t5"  # "word" -> 1 + 4 % 7


class TestEmptyDocuments:
    """A model can answer with output tokens and no text — an adapter
    dropping content blocks it does not map. Embedding such a record is
    impossible, and skipping it must change n visibly."""

    def test_a_document_is_embedded_by_its_text(self):
        from mechbench_compute.distill import render

        r = render(StubModel(), {"id": "a", "text": "a story"})
        assert r.text == "a story" and not r.chat and r.ids == [0, 2, 6]

    def test_an_empty_record_is_refused_by_name(self):
        import pytest

        from mechbench_compute.distill import render

        with pytest.raises(ValueError, match="record 'a' has no prompt"):
            render(StubModel(), {"id": "a", "text": "   "})

    def test_unmapped_content_blocks_are_not_silently_empty(self, monkeypatch):
        from dataclasses import dataclass, field

        from mechbench_compute.providers import anthropic, http
        from mechbench_compute.providers.messages import request

        @dataclass
        class Resp:
            status: int = 200
            headers: dict = field(default_factory=dict)
            body: dict = field(default_factory=lambda: {
                "id": "m1", "model": "claude-sonnet-5", "stop_reason": "end_turn",
                "content": [{"type": "some_future_block", "data": "?"}],
                "usage": {"input_tokens": 10, "output_tokens": 250}})

        monkeypatch.setattr(http, "post_json", lambda *a, **k: Resp())
        t = anthropic.AnthropicTransport({"token": "sk-ant-x"})
        out = t.chat(request({"model": "claude-sonnet-5",
                              "messages": [{"role": "user", "content": "hi"}]}))
        assert out.empty is not None and out.empty.cause == "unmapped"
        assert "250 output tokens but no text" in out.empty.message
