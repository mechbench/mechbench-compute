"""A bench-based ModelRef loads from its materialized directory —
from EVERY call site (2026-08-25).

The wrapper translated bench label -> local dir, but the block's own
no-op `_model_loaded` re-call did not, so the label went to the
HuggingFace hub as if it were a repo id — after the 10 GB
materialization had already succeeded. The translation now lives in
`_model_loaded` itself; these tests drive the exact production shape.
"""

from __future__ import annotations

import pytest

from mechbench_compute import model_ref
from mechbench_compute import protocol as protocol_mod
from mechbench_compute.protocol import ProtocolExecutor

LABEL = "benjismith/training/checkpoints/spinner-fair-v1"


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    snap = tmp_path / "materialized"
    snap.mkdir()
    loads: list[str] = []
    materialized: list[str] = []

    class FakeModel:
        pass

    def fake_load(model_id, **_kw):
        loads.append(str(model_id))
        return FakeModel()

    monkeypatch.setattr(protocol_mod.Model, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        ProtocolExecutor, "_materialize_checkpoint",
        lambda self, label: materialized.append(label) or snap,
    )
    return ProtocolExecutor(), snap, loads, materialized


def _bench_ref():
    return model_ref.parse({"base": {"bench": LABEL}})


class TestBenchBase:
    def test_the_wrapper_and_the_blocks_recall_load_the_same_weights(self, harness):
        executor, snap, loads, materialized = harness
        ref = _bench_ref()

        def block(inputs, params, on_item=None, on_start=None):
            # Real blocks re-ask for their model by the ref in params —
            # this is the call that used to reach the hub with a label.
            return executor._model_loaded(params.get("model"))

        result = executor._run_model_block(block, {}, {"model": ref})
        assert loads == [str(snap)], "one load, from the materialized dir"
        assert materialized == [LABEL, LABEL]  # both calls translated
        assert result is executor._model

    def test_a_bare_label_never_reaches_the_hub_loader(self, harness):
        executor, snap, loads, _ = harness
        executor._model_loaded(_bench_ref())
        assert loads == [str(snap)]
        assert LABEL not in loads

    def test_an_hf_base_still_loads_by_repo_id(self, harness):
        executor, _snap, loads, materialized = harness
        executor._model_loaded(model_ref.parse("mlx-community/gemma@abc"))
        assert loads == ["mlx-community/gemma@abc"]
        assert materialized == []


class TestMaterializeMemo:
    def test_one_manifest_fetch_per_label_per_executor(self, tmp_path, monkeypatch):
        from mechbench_compute import bench, checkpoint

        fetches: list[str] = []
        monkeypatch.setattr(
            bench, "fetch",
            lambda label, with_meta=False: (
                fetches.append(label),
                ({"kind": "checkpoint_manifest", "files": []}, {}),
            )[1],
        )
        monkeypatch.setattr(
            checkpoint, "materialize",
            lambda payload, fetch_file, root, **kw: tmp_path,
        )
        executor = ProtocolExecutor()
        d1 = executor._materialize_checkpoint(LABEL)
        d2 = executor._materialize_checkpoint(LABEL)
        assert d1 == d2 == tmp_path
        assert len(fetches) == 1
