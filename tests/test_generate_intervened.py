"""Generation under an intervention (task 000601): the spec is live at
every forward pass — the prompt's prefill and each decoding step — with
positions resolved over the whole sequence as it grows, a sweep giving
one set of samples per factor, and factor 0 the plain path."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import distill, generate
from mechbench_compute import intervene as iv
from mechbench_compute.hooks import HookInfo
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec


class _Arch:
    n_layers = 4


class _Tok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _Model:
    tokenizer = _Tok()
    arch = _Arch()


# --- positions across chunks -------------------------------------------------

def _hook(positions, tokens, prompt_len):
    spec = iv.Spec({"point": "resid_post", "layers": [2], "op": "scale", "strength": 0.0,
                    "positions": positions}, n_layers=4, seed=0)
    live = iv.SpecIntervention([spec], tokens, None, prompt_len=prompt_len, growing=True)
    return live, live.as_hooks()["blocks.2.resid_post"]


def _act(L):
    return mx.ones((1, L, 3))


def _zeroed(out, L):
    return [i for i in range(L) if float(out[0, i, 0]) == 0.0] if out is not None else []


class TestPositionsResolveOverTheWholeSequence:
    def test_last_is_the_prompts_end_then_each_new_token(self):
        live, fn = _hook("last", ["a", "b", "c"], 3)
        # The prefill: one chunk of three at offset 0; "last" is index 2.
        assert _zeroed(fn(_act(3), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=0)), 3) == [2]
        # A decoding step: a one-token chunk at offset 3; "last" is it.
        live.on_token("d")
        assert _zeroed(fn(_act(1), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=3)), 1) == [0]

    def test_a_token_selector_sees_the_words_as_they_arrive(self):
        live, fn = _hook({"tokens": ["d"]}, ["a", "b", "c"], 3)
        assert _zeroed(fn(_act(3), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=0)), 3) == []
        live.on_token("d")
        assert _zeroed(fn(_act(1), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=3)), 1) == [0]
        live.on_token("e")
        # "d" sits at position 3, outside the chunk at offset 4: untouched.
        assert _zeroed(fn(_act(1), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=4)), 1) == []

    def test_generated_is_nothing_in_the_prompt_and_everything_after(self):
        live, fn = _hook("generated", ["a", "b", "c"], 3)
        assert _zeroed(fn(_act(3), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=0)), 3) == []
        live.on_token("d")
        assert _zeroed(fn(_act(1), HookInfo("blocks.2.resid_post", 2, "resid_post", offset=3)), 1) == [0]

    def test_a_whole_sequence_pass_is_the_chunk_at_offset_zero(self):
        # The pre-000601 contract: no offset on the info, the tensor is
        # the sequence.
        _, fn = _hook({"range": [1, 3]}, ["a", "b", "c"], 3)
        assert _zeroed(fn(_act(3), HookInfo("blocks.2.resid_post", 2, "resid_post")), 3) == [1, 2]

    def test_a_one_shot_pass_still_refuses_a_token_that_is_not_there(self):
        spec = iv.Spec({"point": "resid_post", "layers": [2], "positions": {"tokens": ["zzz"]}},
                       n_layers=4, seed=0)
        fn = iv.SpecIntervention([spec], ["a", "b"], None).as_hooks()["blocks.2.resid_post"]
        with pytest.raises(iv.SpecError, match="none of"):
            fn(_act(2), HookInfo("blocks.2.resid_post", 2, "resid_post"))


# --- the block ---------------------------------------------------------------

RECORD = {"id": "p0", "user": "Say something.", "coords": {"cond": "x"}}


@pytest.fixture
def seen(monkeypatch):
    """A fake substrate that records, per call, whether the prefill and
    the sampler were handed live interventions, and at what strength."""
    calls: list[dict] = []

    monkeypatch.setattr(ProtocolExecutor, "_model_loaded", lambda self, model_id: _Model())
    monkeypatch.setattr(ProtocolExecutor, "_run_model_block",
                        lambda self, fn, inputs, params, *a, **k: fn(inputs, params, *a, **k))
    monkeypatch.setattr(distill, "render_chat", lambda tok, s, u, p: f"<{u}>")
    monkeypatch.setattr(distill, "encode", lambda tok, text: [ord(c) for c in text])

    def prefill(model, ids, interventions=None):
        calls.append({"at": "prefill", "ivs": interventions})
        return ("cache", ids)

    def sample(model, ids, *, max_tokens, temperature, top_p, rng, prefill,
               return_ids=False, stop_strings=(), interventions=None):
        calls.append({"at": "sample", "ivs": interventions})
        return "words", [ord("w")]

    monkeypatch.setattr(distill, "prefill_decision", prefill)
    monkeypatch.setattr(generate, "sample_completion_cached", sample)
    return calls


def _run(params, inputs=None):
    graph = {"nodes": [{"id": "gen", "block": "text/generate",
                        "params": {"model": "fake/m@rev", "n": 2, "seed": 7, **params},
                        "inputs": {"records": [RECORD], **(inputs or {})}}], "edges": []}
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                              extra={"graph": graph}))
    payload = out.payload if hasattr(out, "payload") else out
    return payload["outputs"]["gen"]


ITEM = {"point": "resid_post", "layers": [1], "op": "scale", "strength": 0.5}


def _strength(ivs):
    return None if not ivs else ivs[0]._hooks and next(iter(ivs[0]._hooks.values()))


class TestTheBlock:
    def test_without_an_intervention_the_calls_are_the_ones_they_always_were(self, seen):
        out = _run({})
        assert [c["ivs"] for c in seen] == [None, None, None]
        assert [i["id"] for i in out["items"]] == ["p0-s0", "p0-s1"]
        assert "spec" not in out and "factor" not in out["items"][0]["coords"]

    def test_a_spec_is_live_at_the_prefill_and_at_every_sample_per_factor(self, seen):
        out = _run({"spec": [ITEM], "sweep": {"strength": [1.0, 2.0]}})
        # Factor 0 (the control) first, then 1 and 2: one prefill and
        # two samples each; the control runs with no intervention.
        assert out["sweep"] == {"strength": [0.0, 1.0, 2.0]}
        by_factor = [seen[i:i + 3] for i in range(0, 9, 3)]
        assert [c["ivs"] for c in by_factor[0]] == [None, [], []] or all(not c["ivs"] for c in by_factor[0])
        for chunk in by_factor[1:]:
            assert [c["at"] for c in chunk] == ["prefill", "sample", "sample"]
            assert all(isinstance(c["ivs"][0], iv.SpecIntervention) for c in chunk)
            # Each sample gets its OWN live intervention, so the token
            # lists it grows do not leak between samples.
            assert chunk[1]["ivs"][0] is not chunk[2]["ivs"][0]
        ids = sorted(i["id"] for i in out["items"])
        assert ids == sorted(["p0-s0-f0", "p0-s1-f0", "p0-s0-f1", "p0-s1-f1", "p0-s0-f2", "p0-s1-f2"])
        assert sorted(i["coords"]["factor"] for i in out["items"]) == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        assert out["spec"][0]["op"] == "scale" and "weights" not in out  # no edits: absent, as apply does

    def test_the_spec_may_arrive_as_an_object_on_the_port(self, seen):
        spec = {"kind": "intervene/spec", "items": [ITEM]}
        out = _run({"control": False}, inputs={"intervention": spec})
        assert out["sweep"] == {"strength": [1.0]}
        assert all(c["ivs"] and isinstance(c["ivs"][0], iv.SpecIntervention) for c in seen)

    def test_the_scaled_strength_reaches_the_hook(self):
        spec = iv.Spec(ITEM, n_layers=4, seed=0)
        assert [s.strength for s in iv.scaled([spec], 2.0)] == [1.0]
        assert iv.scaled([spec], 1.0)[0] is spec


# --- the real forward, chunked --------------------------------------------------

E2B = "mlx-community/gemma-4-e2b-it-bf16"
_real = pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1"
    or not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--" + E2B.replace("/", "--"))),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached",
)


def _last_logp(logits):
    row = logits[0, -1, :].astype(mx.float32)
    return np.array(row - mx.logsumexp(row))


@_real
class TestTheHookedForwardRunsInChunks:
    """A sequence run as the prompt then one token per step through the
    hooked forward, with an external KV cache, is the SAME computation
    mlx's own cached decode performs — bit for bit with no hook — and
    under a hook whose positions span the chunks it reads the
    distribution the whole-sequence hooked pass reads, up to the bf16
    drift mlx itself has between a whole pass and a cached one. Without
    this, an intervention 'at every decoding step' would be a different
    model."""

    @pytest.fixture(scope="class")
    def model(self):
        from mechbench_compute import Model
        return Model.load(E2B)

    @staticmethod
    def _chunks(ids):
        k = ids.shape[1] - 2
        return [ids[:, :k], ids[:, k:k + 1], ids[:, k + 1:]]

    def _hooked_chunked(self, model, ids, interventions=None):
        kv = model.prompt_cache()
        for chunk in self._chunks(ids):
            res = model.run(chunk, interventions=interventions, kv_cache=kv)
        return _last_logp(res.logits)

    def _native_chunked(self, model, ids):
        c = model.prompt_cache()
        for chunk in self._chunks(ids):
            o = model.lm(chunk, cache=c)
        return _last_logp(o.logits if hasattr(o, "logits") else o)

    @staticmethod
    def _tv(a, b):
        return 0.5 * float(np.abs(np.exp(a) - np.exp(b)).sum())

    def test_plain_is_mlx_cached_decode_bit_for_bit(self, model):
        ids = model.tokenize("The old lighthouse keeper climbed the", chat_template=False)
        a, b = self._hooked_chunked(model, ids), self._native_chunked(model, ids)
        assert np.array_equal(a, b)

    def test_under_a_hook_spanning_both_chunks(self, model):
        ids = model.tokenize("The old lighthouse keeper climbed the", chat_template=False)
        toks = [model.tokenizer.decode([int(t)]) for t in np.array(ids).reshape(-1)]
        layer = model.arch.n_layers // 3
        spec = iv.Spec({"point": "resid_post", "layers": [layer], "op": "scale",
                        "strength": 0.0, "positions": {"range": [2, 6]}}, n_layers=model.arch.n_layers, seed=0)
        live = [iv.SpecIntervention([spec], toks, None)]
        whole = _last_logp(model.run(ids, interventions=live).logits)
        chunked = self._hooked_chunked(model, ids, interventions=live)
        plain = self._native_chunked(model, ids)
        # The hook changed the distribution, and both routes agree on
        # what it became — within mlx's own whole-vs-cached drift
        # (measured at TV ≈ 0.03 on this prompt with no hook at all).
        assert self._tv(chunked, plain) > 0.1
        assert self._tv(whole, chunked) < 0.06
        assert np.argmax(whole) == np.argmax(chunked)

    def test_a_hook_inside_attention_under_a_cache(self, model):
        # The manual attention path handles the cache offset for RoPE and
        # the K/V update; a hook at attn.q forces it.
        ids = model.tokenize("The old lighthouse keeper climbed the", chat_template=False)
        toks = [model.tokenizer.decode([int(t)]) for t in np.array(ids).reshape(-1)]
        layer = model.arch.last_fresh_kv_global
        spec = iv.Spec({"point": "attn.q", "layers": [layer], "op": "scale", "strength": 0.5,
                        "positions": "all"}, n_layers=model.arch.n_layers, seed=0)
        live = [iv.SpecIntervention([spec], toks, None)]
        whole = _last_logp(model.run(ids, interventions=live).logits)
        chunked = self._hooked_chunked(model, ids, interventions=live)
        assert self._tv(whole, chunked) < 0.06
        assert np.argmax(whole) == np.argmax(chunked)

    def test_a_generated_sample_under_a_zero_strength_spec_is_the_plain_sample(self, model):
        from mechbench_compute.distill import prefill_decision
        from mechbench_compute.generate import sample_completion_cached

        ids = [int(t) for t in np.array(model.tokenize("Write one sentence about the sea.")).reshape(-1)]
        toks = [model.tokenizer.decode([t]) for t in ids]
        spec = iv.Spec({"point": "resid_post", "layers": [4], "op": "scale", "strength": 1.0,
                        "positions": "last"}, n_layers=model.arch.n_layers, seed=0)
        live = [iv.SpecIntervention([spec], toks, None, growing=True)]
        plain = sample_completion_cached(model, ids, max_tokens=12, temperature=0.7, top_p=0.95,
                                         rng=np.random.default_rng(3), prefill=prefill_decision(model, ids))
        same = sample_completion_cached(model, ids, max_tokens=12, temperature=0.7, top_p=0.95,
                                        rng=np.random.default_rng(3),
                                        prefill=prefill_decision(model, ids, interventions=live),
                                        interventions=live)
        assert same == plain
        # And the intervention's token list grew with the sample.
        assert len(live[0].tokens) > len(toks)


class TestChatUnderAnIntervention:
    """`text/chat` on local weights takes an intervention too (000601),
    and sweeps its axes (000602) — the path `text/generate`'s tests did
    not cover, which is how it went a release without one."""

    def _run(self, params, monkeypatch):
        from mechbench_compute import chat as chat_mod
        from mechbench_compute import distill, generate
        from mechbench_compute import model_ref as mr

        seen = []

        class FakeTok:
            def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, **kw):
                return " | ".join(f"{t['role']}:{t['content']}" for t in turns)

            def decode(self, ids):
                return "".join(chr(i + 96) for i in ids)

        class FakeModel:
            tokenizer = FakeTok()
            arch = _Arch()

        monkeypatch.setattr(distill, "encode", lambda tok, text: [1, 2, 3])
        monkeypatch.setattr(distill, "prefill_decision",
                            lambda m, ids, interventions=None: seen.append(("prefill", interventions)) or None)
        monkeypatch.setattr(generate, "sample_completion_cached",
                            lambda *a, interventions=None, **k: seen.append(("sample", interventions)) or "said")
        out = chat_mod.run_local(FakeModel(), mr.parse("google/gemma-3-4b-it"),
                                 [{"id": "q0", "user": "a question", "coords": {"topic": "t"}}],
                                 {"n": 1, "seed": 3, **params})
        return out, seen

    def test_without_one_the_calls_are_unchanged(self, monkeypatch):
        out, seen = self._run({}, monkeypatch)
        assert [s[1] for s in seen] == [None, None]
        assert [i["id"] for i in out["items"]] == ["q0-s0"]
        assert "factor" not in out["items"][0]["coords"]

    def test_a_layer_sweep_gives_one_set_of_replies_per_cell(self, monkeypatch):
        out, seen = self._run({"spec": [{"point": "resid_post", "op": "zero"}],
                               "sweep": {"layers": [1, 2]}}, monkeypatch)
        assert [i["id"] for i in out["items"]] == ["q0-s0-control", "q0-s0-layer1", "q0-s0-layer2"]
        assert [i["coords"].get("layer") for i in out["items"]] == [None, 1, 2]
        assert [i["coords"]["factor"] for i in out["items"]] == [0.0, 1.0, 1.0]
        # The control asked for nothing; each cell had its own live spec.
        assert [bool(s[1]) for s in seen] == [False, False, True, True, True, True]
        assert out["sweep"] == {"strength": [0.0, 1.0], "layers": [1, 2]}
