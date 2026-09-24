from __future__ import annotations

import pytest

from mechbench_compute import model_ref as model_ref_mod
from mechbench_compute import resume as resume_mod
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

from tests.test_resume import _Calls, _fake_generate_substrate

BASE = "fake/base@rev"
LABEL = "someone/proj/adapters/big"
ADAPTER_BYTES = b"\x7f" * 200_000


def _resolved() -> model_ref_mod.ModelRef:
    return model_ref_mod.ModelRef(
        base_kind="hf", base=BASE, adapter_labels=(LABEL,),
        adapter_payloads=({"data": ADAPTER_BYTES, "lora": {"rank": 8}},))


class TestTheFingerprintIsOverWireForms:
    def test_a_resolved_ref_fingerprints_as_its_wire_form(self):
        ref = _resolved()
        live = resume_mod.node_fingerprint(
            block="text/generate", params={"model": ref, "n": 3},
            input_hashes=[], core_version="0.70.0")
        wire = resume_mod.node_fingerprint(
            block="text/generate",
            params={"model": ref.to_wire(), "n": 3},
            input_hashes=[], core_version="0.70.0")
        assert live == wire

    def test_the_adapter_bytes_do_not_influence_the_fingerprint(self):
        a = _resolved()
        b = model_ref_mod.ModelRef(
            base_kind="hf", base=BASE, adapter_labels=(LABEL,),
            adapter_payloads=({"data": b"\x00" * 10, "lora": {"rank": 8}},))
        fp = lambda r: resume_mod.node_fingerprint(  # noqa: E731
            block="b", params={"model": r}, input_hashes=[], core_version="v")
        assert fp(a) == fp(b)

    def test_an_unserializable_param_raises_and_names_itself(self):
        class Live:
            pass

        with pytest.raises(TypeError) as caught:
            resume_mod.node_fingerprint(
                block="~canonical/ops/x/1", params={"ok": 1, "handle": Live()},
                input_hashes=[], core_version="v")
        assert "handle" in str(caught.value)
        assert "repr" not in str(caught.value).lower() or "never" in str(caught.value)


class TestTheExecutorHashesBeforeItEmits:
    def test_a_live_object_in_a_result_never_reaches_emit(self, monkeypatch):
        from mechbench_compute import bench, protocol

        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)

        class Live:
            pass

        from mechbench_compute.ops.text import generate

        real = generate.run

        def poisoned(ctx, inputs, params):
            out = real(ctx, inputs, params)
            out["items"][0]["handle"] = Live()
            return out

        monkeypatch.setattr(generate, "run", poisoned)
        emitted: list[str] = []
        monkeypatch.setattr(bench, "emit",
                            lambda target, *a, **k: emitted.append(target) or {"path": target})
        monkeypatch.setattr(protocol, "bench", bench, raising=False)

        graph = {"dataflow": 2, "nodes": [{
            "id": "gen", "block": "text/generate",
            "params": {"model": BASE, "n": 1, "seed": 7},
                       "inputs": {"records": [{"id": "r", "user": "Write."}]}}],
            "edges": []}
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": graph, "resultPath": "owner/proj/results/j_test"})
        with pytest.raises(Exception) as caught:
            ProtocolExecutor().run(spec)
        assert "Live" in str(caught.value) or "encode" in str(caught.value).lower()
        assert emitted == [], "the un-encodable result was handed to emit"

    def test_a_clean_result_is_hashed_then_emitted(self, monkeypatch):
        from mechbench_compute import bench

        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        emitted: list[str] = []
        monkeypatch.setattr(bench, "emit",
                            lambda target, *a, **k: emitted.append(target) or {"path": target})
        graph = {"dataflow": 2, "nodes": [{
            "id": "gen", "block": "text/generate",
            "params": {"model": BASE, "n": 1, "seed": 7},
                       "inputs": {"records": [{"id": "r", "user": "Write."}]}}],
            "edges": []}
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": graph, "resultPath": "owner/proj/results/j_test"})
        ProtocolExecutor().run(spec)
        assert emitted == ["owner/proj/results/j_test/gen"]
