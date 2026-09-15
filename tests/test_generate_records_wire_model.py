"""A generated item records its model as the WIRE FORM, never the
resolved object (task 000488).

The resolved ModelRef carries `adapter_payloads` — the fetched adapter's
safetensors bytes. Before this test existed the generate block wrote the
object itself into `metadata.model` and `generation_spans[0].model`, and
`str()` of it into `trace.tokenizer`. For an adapted model that is the
whole adapter, three times, in every item: ~32 MB per story against the
4.5 KB a base-model story weighs. Experiment 014's adapted arm produced a
result the API could not receive without being OOM-killed, and three
runs and a 90-minute outage were spent finding out why.

Two things are pinned here, and the second is the one that will catch a
regression in a different block: the recorded model is the wire form,
and the item is canonical-CBOR-encodable at a size that has nothing to
do with the adapter.
"""

from __future__ import annotations

import mechbench_schema as ms
import pytest

from mechbench_compute import model_ref as model_ref_mod
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

from tests.test_resume import _Calls, _fake_generate_substrate

BASE = "fake/base@rev"
LABEL = "someone/proj/adapters/big"
#: Comfortably larger than any sane item, so a leak is unmistakable and
#: the size bound below cannot pass by accident.
ADAPTER_BYTES = b"\x00" * 300_000


def _resolved_ref() -> model_ref_mod.ModelRef:
    return model_ref_mod.ModelRef(
        base_kind="hf", base=BASE, adapter_labels=(LABEL,),
        adapter_payloads=({"data": ADAPTER_BYTES,
                           "lora": {"rank": 8, "alpha": 16, "scale": 2.0,
                                    "target_modules": ["q_proj", "v_proj"]}},),
    )


def _spec(fidelity: str):
    graph = {"nodes": [{
        "id": "gen", "block": "text/generate",
        "params": {"model": "$model", "n": 2, "seed": 7, "fidelity": fidelity},
        "inputs": {"records": [{"id": "flash", "user": "Write a story."}]},
    }], "edges": []}
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": graph,
        "bindings": {"model": {"base": {"hf": BASE},
                               "adapters": [{"bench": LABEL}]}},
    })


@pytest.fixture
def adapted_run(monkeypatch):
    calls = _Calls()
    _fake_generate_substrate(monkeypatch, calls)
    ref = _resolved_ref()
    # The executor resolves the binding through model_ref.resolve, which
    # fetches the adapter from the bench; stand that in with the resolved
    # object so the test sees exactly what a real run's params hold.
    monkeypatch.setattr(model_ref_mod, "resolve", lambda mval, **kw: ref)
    monkeypatch.setattr(ProtocolExecutor, "_record_model",
                        lambda self, *a, **k: None, raising=False)
    return ref


def _collection(out):
    """`run()` returns an Emitted envelope whose payload keys each
    TERMINAL node's result under `outputs`; `gen` is the only node."""
    payload = out.payload if hasattr(out, "payload") else out
    return payload["outputs"]["gen"]


def _items(fidelity, ref):
    items = _collection(ProtocolExecutor().run(_spec(fidelity)))["items"]
    assert len(items) == 2
    return items


class TestTheRecordedModel:
    def test_metadata_model_is_the_wire_form(self, adapted_run):
        ref = adapted_run
        for it in _items("trace", ref):
            assert it["metadata"]["model"] == ref.to_wire()
            assert it["metadata"]["model"] == {
                "base": {"hf": BASE}, "adapters": [{"bench": LABEL}]}

    def test_generation_span_model_is_the_wire_form(self, adapted_run):
        ref = adapted_run
        for it in _items("trace", ref):
            assert it["trace"]["generation_spans"][0]["model"] == ref.to_wire()

    def test_tokenizer_is_the_base_id_not_a_repr(self, adapted_run):
        for it in _items("trace", adapted_run):
            assert it["trace"]["tokenizer"] == BASE

    def test_a_base_model_string_still_passes_through(self, monkeypatch):
        """The base arm never had the bug; make sure the fix does not
        change what it records."""
        calls = _Calls()
        _fake_generate_substrate(monkeypatch, calls)
        graph = {"nodes": [{
            "id": "gen", "block": "text/generate",
            "params": {"model": BASE, "n": 1, "seed": 7, "fidelity": "trace"},
            "inputs": {"records": [{"id": "flash", "user": "Write."}]}}],
            "edges": []}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None, extra={"graph": graph}))
        it = _collection(out)["items"][0]
        assert it["metadata"]["model"] == BASE
        assert it["trace"]["tokenizer"] == BASE
        assert it["trace"]["generation_spans"][0]["model"] == BASE


class TestTheItemIsSmallAndEncodable:
    """The invariant that outlives this particular field: a result item
    must canonical-encode, and its size must not depend on the adapter."""

    @pytest.mark.parametrize("fidelity", ["text", "trace"])
    def test_canonical_cbor_encodes_and_is_tiny(self, adapted_run, fidelity):
        for it in _items(fidelity, adapted_run):
            encoded = ms.dump_canonical(it)  # raised on the ModelRef before
            assert len(encoded) < 4_096, (
                f"{len(encoded):,} bytes for one item — something is "
                f"embedding the adapter ({len(ADAPTER_BYTES):,} bytes)")

    def test_the_adapter_bytes_appear_nowhere_in_the_result(self, adapted_run):
        for it in _items("trace", adapted_run):
            assert ADAPTER_BYTES[:64] not in ms.dump_canonical(it)
            assert "adapter_payloads" not in repr(it)
