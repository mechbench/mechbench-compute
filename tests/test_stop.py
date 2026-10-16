from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from mechbench_compute import ops
from mechbench_compute import resume as rm
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec, Stop
from mechbench_compute.protocol import pipeline as pipeline_mod
from mechbench_compute.protocol.progress import Progress
from mechbench_compute.protocol.stop import StopCheck
from tests.test_dictionary_train import SMALL, make_records
from tests.test_keep_checkpoints import KEPT, RESULTS, stored, train_node
from tests.test_resume import (
    _Calls,
    _digest,
    _examples,
    _fake_generate_substrate,
    _fresh_lm,
    _gen_spec,
    _Spool,
)
from tests.tiny_models import build_tiny_model


def serve_tiny(monkeypatch, tmp_path, model_type: str) -> None:
    from mechbench_compute import architectures
    from mechbench_compute import model as model_mod

    def load(model_id, **_):
        tiny = build_tiny_model(model_type)
        return tiny._model, tiny._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE[model_type], load=load)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": model_type})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)


@pytest.fixture
def kept_store(monkeypatch):
    from mechbench_compute import bench

    kept: dict[str, object] = {}

    def emit(path, payload, **kw):
        kept[path] = payload
        return {"path": path}

    monkeypatch.setattr(bench, "emit", emit)
    monkeypatch.setattr(bench, "fetch", lambda path, **kw: {"payload": kept[path]})
    return kept


class After:
    def __init__(self, calls: int, *, once: bool = False) -> None:
        self.calls = calls
        self.once = once
        self.seen = 0

    def __call__(self) -> bool:
        self.seen += 1
        return self.seen == self.calls if self.once else self.seen >= self.calls


class CheckingTooEarly(Progress):
    def open_items(self, nid, stops=False):
        def on_item(key=None, item=None, reused=False):
            self.node_view["done"] += 1
            self.bump(1)
            if stops and key is not None and not reused:
                self.stops.at_item(nid, key)
            if self.on_spool_item is not None and key is not None and not reused:
                self.on_spool_item(nid, key, item)

        return on_item


def stop_after_item(monkeypatch, k: int) -> None:
    calls = _Calls()
    _fake_generate_substrate(monkeypatch, calls)
    full = _Spool()
    reference = full.executor().run(_gen_spec())

    calls.n = 0
    stopped = _Spool()
    with pytest.raises(Stop) as caught:
        stopped.executor().run(_gen_spec(), stop=After(k))
    assert caught.value.node == "gen"
    assert calls.n == k
    assert len(stopped.items.get("gen", {})) == k, "the item produced before the stop was not spooled"
    assert "gen" not in stopped.done

    calls.n = 0
    resumed = _Spool()
    out = resumed.executor().run(_gen_spec(), resume=stopped.resume_map("gen"))
    assert calls.n == 6 - k
    assert _digest(out) == _digest(reference)
    assert resumed.done["gen"][1] == full.done["gen"][1]


class TestAStopAtAnItem:
    @pytest.mark.parametrize("k", [1, 4])
    def test_a_resumed_run_is_identical_to_an_uninterrupted_one(self, monkeypatch, k):
        stop_after_item(monkeypatch, k)

    def test_a_check_before_the_spool_write_loses_the_item(self, monkeypatch):
        monkeypatch.setattr(pipeline_mod, "Progress", CheckingTooEarly)
        with pytest.raises(AssertionError, match="was not spooled"):
            stop_after_item(monkeypatch, 4)

    def test_a_replayed_item_does_not_ask(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        full = _Spool()
        full.executor().run(_gen_spec())
        asked = After(10**9)
        _Spool().executor().run(_gen_spec(), resume=full.resume_map("gen"), stop=asked)
        assert asked.seen == 0

    def test_no_stop_runs_as_before(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        a, b = _Spool(), _Spool()
        out_a = a.executor().run(_gen_spec())
        out_b = b.executor().run(_gen_spec(), stop=lambda: False)
        assert _digest(out_a) == _digest(out_b) and a.items == b.items

    def test_a_stop_never_yields_a_missing_node(self, monkeypatch):
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        failed: list[str] = []
        spool = _Spool()
        ex = spool.executor()
        ex._on_node_done = lambda nid, path, fp: failed.append(nid) if path is None else None
        with pytest.raises(Stop):
            ex.run(_gen_spec(), stop=After(2))
        assert failed == []
        assert len(spool.items["gen"]) == 2


class _Checkpoints:
    def __init__(self) -> None:
        self.fingerprints: dict[str, str] = {}
        self.items: dict[str, dict] = {}
        self.states: list[dict] = []

    def executor(self) -> ProtocolExecutor:
        return ProtocolExecutor(
            on_node_start=lambda nid, fp: self.fingerprints.setdefault(nid, fp),
            on_spool_item=lambda nid, key, item: self.items.__setitem__(key, item),
            on_checkpoint=lambda nid, state: self.states.append(state))

    def resume_map(self, nid: str) -> dict:
        return {nid: {"fingerprint": self.fingerprints[nid], "items": dict(self.items),
                      "checkpoint": self.states[-1]}}


def run(executor, nodes, *, outputs=None, result_path=RESULTS, **kw):
    extra = {"graph": {"dataflow": 2, "nodes": nodes, "edges": []}}
    if result_path is not None:
        extra["resultPath"] = result_path
    if outputs is not None:
        extra["outputs"] = outputs
    return executor.run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra=extra), **kw)


class TestAStopInAdapterTraining:
    def test_the_checkpoint_lands_at_the_step_and_the_resume_is_identical(
            self, monkeypatch, tmp_path, kept_store):
        serve_tiny(monkeypatch, tmp_path, "llama")
        node = train_node(keep_checkpoints=True)
        whole = _Checkpoints()
        run(whole.executor(), [node], outputs=KEPT)
        reference = {name: stored(kept_store, name) for name in ("adapter", "kept")}
        kept_store.clear()
        assert [s["step"] for s in whole.states] == [2, 4]

        stopped = _Checkpoints()
        with pytest.raises(Stop) as caught:
            run(stopped.executor(), [node], outputs=KEPT, stop=After(3))
        assert caught.value.node == "train"
        assert [s["step"] for s in stopped.states] == [2, 3]
        assert sorted(stopped.items) == ["step-2"]

        resumed = _Checkpoints()
        resumed.fingerprints = dict(stopped.fingerprints)
        ticks: list[tuple[int, int]] = []
        run(resumed.executor(), [node], outputs=KEPT, resume=stopped.resume_map("train"),
            on_progress=lambda d, t: ticks.append((d, t)))
        assert [s["step"] for s in resumed.states] == [4]
        assert ticks[-1][0] == ticks[-1][1]
        for name in ("adapter", "kept"):
            assert rm.content_hash(stored(kept_store, name)) == rm.content_hash(reference[name])

    def test_an_unpaused_run_spools_the_checkpoints_it_spooled_before(
            self, monkeypatch, tmp_path, kept_store):
        serve_tiny(monkeypatch, tmp_path, "llama")
        plain, asked = _Checkpoints(), _Checkpoints()
        run(plain.executor(), [train_node(steps=7, checkpoint_every=3)])
        run(asked.executor(), [train_node(steps=7, checkpoint_every=3)], stop=lambda: False)
        assert [s["step"] for s in plain.states] == [s["step"] for s in asked.states] == [3, 6]


class TestAStopInDictionaryTraining:
    def node(self):
        return {"id": "train", "block": "dictionary/train",
                "params": {**SMALL, "model": "tiny/gemma3"}, "inputs": {"records": make_records()}}

    def test_the_checkpoint_lands_at_the_next_chunk_boundary(self, monkeypatch, tmp_path, kept_store):
        serve_tiny(monkeypatch, tmp_path, "gemma3")
        whole = _Checkpoints()
        reference = run(whole.executor(), [self.node()], result_path=None).payload["outputs"]["train"]
        assert [s["step"] for s in whole.states] == [8, 16, 24]

        stopped = _Checkpoints()
        with pytest.raises(Stop):
            run(stopped.executor(), [self.node()], result_path=None, stop=After(3))
        assert [s["step"] for s in stopped.states] == [8, 12]

        resumed = _Checkpoints()
        resumed.fingerprints = dict(stopped.fingerprints)
        out = run(resumed.executor(), [self.node()], result_path=None,
                  resume=stopped.resume_map("train")).payload["outputs"]["train"]
        assert [s["step"] for s in resumed.states] == [16, 24]
        assert out["trained"] == reference["trained"]
        assert out["fidelity"] == reference["fidelity"]
        assert rm.content_hash(out) == rm.content_hash(reference)


ENDPOINT = {"provider": "mock", "model": "mock-large"}


def chat(nid):
    return {"id": nid, "block": "text/chat",
            "params": {"model": ENDPOINT, "budget_usd": 5.0, "concurrency": 1, "n": 4, "seed": 1},
            "inputs": {"records": [{"id": f"{nid}-r", "user": f"hello from {nid}"}]}}


class TestAStopInARemoteWave:
    def test_the_stop_reaches_the_caller_and_the_workers_go_quiet(self, monkeypatch):
        from mechbench_compute.protocol import dispatch
        from mechbench_compute.providers import mock as mock_mod

        real = mock_mod.MockTransport._chat
        in_flight, peak, in_flight_calls = [], [0], []
        lock = threading.Lock()

        def slow(self, req, **kw):
            with lock:
                in_flight_calls.append(1)
                in_flight.append(1)
                peak[0] = max(peak[0], len(in_flight))
            time.sleep(0.05)
            with lock:
                in_flight.pop()
            return real(self, req, **kw)

        monkeypatch.setattr(mock_mod.MockTransport, "_chat", slow)
        reads: list[object] = []
        real_read = dispatch.read_ahead

        def read_ahead(state, nid):
            try:
                return real_read(state, nid)
            except BaseException as exc:
                reads.append(exc)
                raise

        monkeypatch.setattr(dispatch, "read_ahead", read_ahead)
        nodes = [chat("left"), chat("right")]

        whole = _Spool()
        reference = whole.executor().run(_wave(nodes))

        stopped = _Spool()
        with pytest.raises(Stop):
            stopped.executor().run(_wave(nodes), stop=After(1, once=True))
        assert peak[0] == 2, "the two branches did not overlap"
        assert len(reads) == 1 and isinstance(reads[0], Stop)
        assert {nid: len(items) for nid, items in stopped.items.items()} == {"left": 1, "right": 1}
        assert stopped.fingerprints == whole.fingerprints

        calls_before = len(in_flight_calls)
        resumed = _Spool()
        out = resumed.executor().run(
            _wave(nodes), resume={nid: stopped.resume_map(nid)[nid] for nid in stopped.items})
        assert len(in_flight_calls) - calls_before == 6, "the peers' spooled items were not reused"
        assert _digest(read_items(out)) == _digest(read_items(reference))
        assert resumed.done["right"][1] == whole.done["right"][1]


def read_items(out):
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "latency_ms"}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return {nid: strip(out.payload["outputs"][nid]["items"]) for nid in ("left", "right")}


def _wave(nodes):
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"dataflow": 2, "nodes": nodes, "edges": []}})


CONTEXT_MEMBERS = ("loaded", "executor", "on_item", "on_start", "on_checkpoint", "resume_items",
                   "resume_state", "secrets", "input_paths", "run_params", "result_base", "stored",
                   "on_token", "refused", "host", "run_secrets", "scope", "node", "name", "fused",
                   "models")

STANDALONE = frozenset({
    "adapter/measure", "direction/add", "direction/average", "direction/classify",
    "direction/decompose", "direction/fit", "direction/normalize", "direction/orthogonalize",
    "direction/project", "direction/regress", "eval/expect", "eval/score", "geometry/align",
    "geometry/compare", "geometry/span", "records/cross", "records/diff", "records/jq",
    "records/measure-noise", "records/python", "records/tabulate", "records/union",
    "records/unnest", "records/zip", "text/extend", "text/measure", "text/render", "tools/calc",
    "tools/lookup", "trajectory/aggregate", "trajectory/compare", "trajectory/project"})


class TestOpsDoNotLearnAboutStops:
    def test_the_context_has_no_new_member(self):
        assert tuple(f.name for f in dataclasses.fields(ops.Context)) == CONTEXT_MEMBERS

    def test_the_standalone_set_and_the_levels_are_unchanged(self):
        from mechbench_compute.registry import REGISTRY

        standalone = frozenset(r.name for r in REGISTRY.resolved()
                               if not r.op.needs and not ops.read_context_uses(r.module))
        assert STANDALONE <= standalone
        assert not {"adapter/train", "dictionary/train", "text/generate", "text/chat"} & standalone
        assert rm.resume_level("adapter/train") == "state-restorable"
        assert rm.resume_level("dictionary/train") == "state-restorable"
        assert rm.resume_level("text/generate") == "reproducible"
        assert not any("stop" in use for r in REGISTRY.resolved()
                       for use in ops.read_context_uses(r.module))


class TestTheHookContract:
    def test_a_trainer_offers_every_step_but_the_last_and_marks_the_cadence(self):
        from mechbench_compute.finetune import train_soft_ce

        offers: list[tuple[int, bool]] = []

        def on_checkpoint(capture, *, due):
            offers.append((capture()["step"], due))

        train_soft_ce(_fresh_lm(), _examples(), {"target": 1, "anchor": 1}, steps=5, lr=1e-2,
                      seed=3, checkpoint_every=2, on_checkpoint=on_checkpoint)
        assert offers == [(1, False), (2, True), (3, False), (4, True)]

    def test_a_capture_is_taken_only_when_due_or_stopping(self):
        spooled: list[dict] = []
        taken = []

        def capture():
            taken.append(1)
            return {"step": len(taken)}

        offer = StopCheck(After(3)).offer("n", lambda nid, st: spooled.append(st))
        offer(capture, due=False)
        offer(capture, due=True)
        with pytest.raises(Stop, match="its checkpoint spooled"):
            offer(capture, due=False)
        assert len(taken) == 2 and [s["step"] for s in spooled] == [1, 2]
