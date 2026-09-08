"""Bit-identical resume (epic 000320, task 000322) — the acceptance
tests that define the semantics.

For an item-resumable block: run a pipeline to completion; run it
again with an interruption after item j, collecting what the runner
would have spooled; resume with that map; the result payload's
canonical bytes and the node object's content hash must be identical
to the uninterrupted run, and only the missing items are computed.
For training: interrupt at a checkpoint, resume, and the final
weights must be byte-identical. A fingerprint mismatch, or a consumer
requiring more than the block offers, restarts the node instead.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest
from mechbench_schema import dump_canonical
from mlx import nn

from mechbench_compute import resume as rm
from mechbench_compute.distill import Example
from mechbench_compute.finetune import train_soft_ce
from mechbench_compute.lora import apply_lora
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

# --- a fake generate substrate: deterministic per rng -------------------------


class _FakeTok:
    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


class _FakeModel:
    tokenizer = _FakeTok()


class _Calls:
    def __init__(self):
        self.n = 0


def _fake_generate_substrate(monkeypatch, calls: _Calls):
    from mechbench_compute import distill, generate

    monkeypatch.setattr(ProtocolExecutor, "_model_loaded",
                        lambda self, model_id: _FakeModel())
    monkeypatch.setattr(ProtocolExecutor, "_run_model_block",
                        lambda self, fn, inputs, params, *a, **k: fn(inputs, params, *a, **k))
    monkeypatch.setattr(distill, "render_chat", lambda tok, s, u, p: f"{s}|{u}|{p}")
    monkeypatch.setattr(distill, "encode", lambda tok, text: [len(text)])
    monkeypatch.setattr(distill, "prefill_decision", lambda model, ids: ("cache", ids))

    def sample(model, ids, *, max_tokens, temperature, top_p, rng, prefill,
               return_ids=False):
        calls.n += 1
        word = int(rng.integers(0, 10**9))
        return f"story-{word}", [1, 2, 3]

    monkeypatch.setattr(generate, "sample_completion_cached", sample)


def _gen_spec(n=3):
    graph = {
        "nodes": [
            {"id": "gen", "block": "~canonical/ops/generate/1",
             "params": {"model": "fake/m@rev", "n": n, "seed": 7,
                        "records": [
                            {"id": "flash", "user": "Write a story."},
                            {"id": "neutral", "user": "Write another."},
                        ]}},
            {"id": "stats", "block": "~canonical/ops/text/stats/1",
             "params": {"measures": [{"kind": "lexical", "name": "lex"}],
                        "mode": "annotate"}},
        ],
        "edges": [{"from": {"node": "gen", "port": "documents"},
                   "to": {"node": "stats", "port": "documents"},
                   "kind": "document_collection"}],
    }
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": graph})


class _Spool:
    """What the runner's second half will keep: fingerprints per node,
    items per node, done nodes."""

    def __init__(self, interrupt_after: int | None = None):
        self.fingerprints: dict[str, str] = {}
        self.items: dict[str, dict[str, dict]] = {}
        self.done: dict[str, tuple[str | None, str]] = {}
        self.interrupt_after = interrupt_after
        self.count = 0

    def on_node_start(self, nid, fp):
        self.fingerprints[nid] = fp

    def on_spool_item(self, nid, key, item):
        self.items.setdefault(nid, {})[key] = item
        self.count += 1
        if self.interrupt_after is not None and self.count >= self.interrupt_after:
            raise KeyboardInterrupt("simulated interruption")

    def on_node_done(self, nid, path, fp):
        self.done[nid] = (path, fp)

    def executor(self):
        return ProtocolExecutor(on_node_start=self.on_node_start,
                                on_spool_item=self.on_spool_item,
                                on_node_done=self.on_node_done)

    def resume_map(self, nid):
        return {nid: {"fingerprint": self.fingerprints[nid],
                      "items": dict(self.items.get(nid, {}))}}


def _digest(payload) -> str:
    return hashlib.sha256(dump_canonical(payload)).hexdigest()


class TestGenerateItemResume:
    def test_resume_after_an_interruption_is_byte_identical(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        full = _Spool()
        reference = full.executor().run(_gen_spec())
        assert calls.n == 6

        # Interrupt after 4 of 6 items; the spool holds those 4.
        calls.n = 0
        partial = _Spool(interrupt_after=4)
        with pytest.raises(KeyboardInterrupt):
            partial.executor().run(_gen_spec())
        assert calls.n == 4 and len(partial.items["gen"]) == 4

        calls.n = 0
        resumed = _Spool()
        out = resumed.executor().run(_gen_spec(), resume=partial.resume_map("gen"))
        assert calls.n == 2  # only the two missing items were computed
        assert _digest(out) == _digest(reference)
        # the node's content hash — what the bench would store — matches
        assert resumed.done["gen"][1] == full.done["gen"][1]

    def test_any_subset_resumes_in_canonical_order(self, monkeypatch):
        # Emission order must not depend on which items were spooled:
        # hand back only the LAST item and the result is still identical.
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        full = _Spool()
        reference = full.executor().run(_gen_spec())
        last_key = "neutral:2"
        assert last_key in full.items["gen"]
        calls.n = 0
        out = _Spool().executor().run(
            _gen_spec(),
            resume={"gen": {"fingerprint": full.fingerprints["gen"],
                            "items": {last_key: full.items["gen"][last_key]}}})
        assert calls.n == 5
        assert _digest(out) == _digest(reference)

    def test_a_fingerprint_mismatch_restarts_the_node(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        full = _Spool()
        full.executor().run(_gen_spec())
        calls.n = 0
        _Spool().executor().run(
            _gen_spec(),
            resume={"gen": {"fingerprint": "sha256:not-the-same",
                            "items": dict(full.items["gen"])}})
        assert calls.n == 6  # nothing reused

    def test_a_param_change_changes_the_fingerprint(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        a = _Spool(); a.executor().run(_gen_spec(n=3))
        b = _Spool(); b.executor().run(_gen_spec(n=4))
        assert a.fingerprints["gen"] != b.fingerprints["gen"]
        # and a downstream node's fingerprint follows its upstream's content
        assert a.fingerprints["stats"] != b.fingerprints["stats"]

    def test_a_consumer_requirement_above_the_offer_forces_restart(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        # Pretend generate were merely exchangeable and the consumer
        # requires reproducible: the partial must not be reused.
        monkeypatch.setitem(rm.BLOCK_RESUME, "~canonical/ops/generate/1",
                            {"level": "exchangeable", "items": True})
        spec = _gen_spec()
        spec.extra["graph"]["nodes"][1]["params"]["require_resume"] = {
            "documents": "reproducible"}
        full = _Spool()
        full.executor().run(spec)
        calls.n = 0
        _Spool().executor().run(spec, resume=full.resume_map("gen"))
        assert calls.n == 6

    def test_a_requirement_the_offer_meets_reuses_the_partial(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        spec = _gen_spec()
        spec.extra["graph"]["nodes"][1]["params"]["require_resume"] = {
            "documents": "reproducible"}
        full = _Spool()
        full.executor().run(spec)
        calls.n = 0
        _Spool().executor().run(spec, resume=full.resume_map("gen"))
        assert calls.n == 0

    def test_reused_items_count_as_progress_but_are_not_respooled(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        full = _Spool()
        full.executor().run(_gen_spec())
        resumed = _Spool()
        ticks = []
        resumed.executor().run(_gen_spec(), resume=full.resume_map("gen"),
                               on_progress=lambda d, t: ticks.append((d, t)))
        assert ticks[-1][0] == ticks[-1][1]  # progress reached the end
        assert resumed.items == {}  # nothing was spooled again


class TestNodeSkip:
    def test_a_done_node_is_fetched_not_recomputed(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        from mechbench_compute import bench

        store: dict[str, object] = {}

        def emit(path, payload, **kw):
            store[path] = payload
            return {"path": path}

        monkeypatch.setattr(bench, "emit", emit)
        monkeypatch.setattr(bench, "fetch", lambda path, **kw: {"payload": store[path]})
        spec = _gen_spec()
        spec.extra["resultPath"] = "u/p/results/j_1"
        full = _Spool()
        reference = full.executor().run(spec)
        gen_path, gen_fp = full.done["gen"]
        assert gen_path == "u/p/results/j_1/gen"
        calls.n = 0
        out = _Spool().executor().run(
            spec, resume={"gen": {"fingerprint": gen_fp, "done": gen_path}})
        assert calls.n == 0
        assert _digest(out) == _digest(reference)

    def test_a_done_node_under_another_fingerprint_is_recomputed(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        from mechbench_compute import bench

        monkeypatch.setattr(bench, "emit", lambda path, payload, **kw: {"path": path})
        monkeypatch.setattr(bench, "fetch",
                            lambda path, **kw: pytest.fail("must not fetch"))
        spec = _gen_spec()
        spec.extra["resultPath"] = "u/p/results/j_1"
        _Spool().executor().run(
            spec, resume={"gen": {"fingerprint": "sha256:stale",
                                  "done": "u/p/results/j_1/gen"}})
        assert calls.n == 6


class TestNonResumableBlocksIgnoreTheMap:
    def test_a_restart_level_block_recomputes(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        monkeypatch.setitem(rm.BLOCK_RESUME, "~canonical/ops/generate/1",
                            {"level": "restart", "items": False})
        full = _Spool()
        reference = full.executor().run(_gen_spec())
        calls.n = 0
        out = _Spool().executor().run(_gen_spec(), resume=full.resume_map("gen"))
        assert calls.n == 6
        assert _digest(out) == _digest(reference)


# --- training: state-restorable -------------------------------------------------


class _TinyLM(nn.Module):
    """A language model small enough to train in a test and shaped so
    `apply_lora` finds `model.layers[i].self_attn.q_proj`."""

    def __init__(self, vocab=16, dim=8, layers=2):
        super().__init__()

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.layers = [Layer() for _ in range(layers)]

        self.model = Inner()
        self.head = nn.Linear(dim, vocab, bias=False)

    def __call__(self, ids):
        h = self.model.embed(ids)
        for layer in self.model.layers:
            h = h + mx.tanh(layer.self_attn.q_proj(h))
        return self.head(h)


def _examples():
    return {
        "target": [Example([1, 2, 3], [], {4: 0.5, 5: 0.5}),
                   Example([2, 3], [], {6: 1.0})],
        "anchor": [Example([7, 8], [9, 10], None)],
    }


def _factory(rng):
    k = int(rng.integers(1, 14))
    return [Example([k], [k + 1, k + 2], None)]


def _weights_bytes(lm) -> bytes:
    from mlx.utils import tree_flatten

    return b"".join(np.array(v).tobytes() for _, v in tree_flatten(lm.trainable_parameters()))


def _fresh_lm(seed=11):
    mx.random.seed(seed)
    lm = _TinyLM()
    apply_lora(lm, rank=2, alpha=4, targets=("q_proj",))
    return lm


class TestTrainingCheckpointResume:
    def test_resume_from_a_checkpoint_is_byte_identical(self):
        lm_a = _fresh_lm()
        loss_a = train_soft_ce(lm_a, _examples(), {"target": 1, "anchor": 1, "seq": 1},
                               steps=6, lr=1e-2, seed=3,
                               factories={"seq": _factory})
        ref = _weights_bytes(lm_a)

        checkpoints = []
        lm_b = _fresh_lm()
        train_soft_ce(lm_b, _examples(), {"target": 1, "anchor": 1, "seq": 1},
                      steps=3, lr=1e-2, seed=3, factories={"seq": _factory},
                      checkpoint_every=3, on_checkpoint=checkpoints.append)
        # `steps=3` finishes AT the would-be checkpoint; a final step
        # never checkpoints, so capture one explicitly the way an
        # interrupted 6-step run at step 3 would have.
        assert checkpoints == []
        lm_c = _fresh_lm()
        got = []
        train_soft_ce(lm_c, _examples(), {"target": 1, "anchor": 1, "seq": 1},
                      steps=6, lr=1e-2, seed=3, factories={"seq": _factory},
                      checkpoint_every=3, on_checkpoint=got.append)
        assert [c["step"] for c in got] == [3]
        state = got[0]
        assert state["mx_key"] is None or hasattr(state["mx_key"], "shape")

        # Same BASE weights (the frozen model comes from the same files
        # in reality; the checkpoint carries only the trainable state),
        # with the LoRA state scrambled first: the checkpoint must
        # override whatever the fresh init produced.
        lm_d = _fresh_lm()
        from mlx.utils import tree_flatten, tree_unflatten

        scrambled = [(k, mx.random.normal(v.shape) * 5.0)
                     for k, v in tree_flatten(lm_d.trainable_parameters())]
        lm_d.update(tree_unflatten(scrambled))
        assert _weights_bytes(lm_d) != ref
        loss_d = train_soft_ce(lm_d, _examples(), {"target": 1, "anchor": 1, "seq": 1},
                               steps=6, lr=1e-2, seed=3, factories={"seq": _factory},
                               resume_state=state)
        assert _weights_bytes(lm_d) == ref
        assert loss_d == loss_a

    def test_checkpoints_fire_on_schedule_and_never_on_the_last_step(self):
        lm = _fresh_lm()
        got = []
        train_soft_ce(lm, _examples(), {"target": 1, "anchor": 1},
                      steps=10, lr=1e-2, seed=3,
                      checkpoint_every=4, on_checkpoint=got.append)
        assert [c["step"] for c in got] == [4, 8]
        assert set(got[0]) == {"step", "weights", "opt_state", "np_rng", "mx_key"}


class TestLevels:
    def test_declared_levels(self):
        assert rm.resume_level("~canonical/ops/generate/1") == "reproducible"
        assert rm.resume_level("~canonical/ops/finetune/lora/1") == "state-restorable"
        assert rm.resume_level("~canonical/ops/nothing/1") == "restart"
        assert rm.item_resumable("~canonical/ops/generate/1")
        assert not rm.item_resumable("~canonical/ops/text/stats/1")

    def test_satisfies(self):
        assert rm.satisfies("reproducible", "reproducible")
        assert rm.satisfies("state-restorable", "reproducible")
        assert not rm.satisfies("exchangeable", "reproducible")
        assert rm.satisfies("exchangeable", "exchangeable")
        assert rm.satisfies("restart", "reproducible")  # recomputes: nothing to mistrust
        with pytest.raises(ValueError):
            rm.satisfies("reproducible", "bit-perfect")

    def test_fingerprint_is_stable_and_sensitive(self):
        a = rm.node_fingerprint(block="b", params={"x": 1}, input_hashes=["h"],
                                core_version="1", model="m")
        b = rm.node_fingerprint(block="b", params={"x": 1}, input_hashes=["h"],
                                core_version="1", model="m")
        c = rm.node_fingerprint(block="b", params={"x": 1}, input_hashes=["h2"],
                                core_version="1", model="m")
        d = rm.node_fingerprint(block="b", params={"x": 1}, input_hashes=["h"],
                                core_version="2", model="m")
        assert a == b and a != c and a != d and a.startswith("sha256:")
