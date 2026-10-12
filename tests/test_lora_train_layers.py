from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mechbench_compute import checkpoint, lora, resume
from mechbench_compute.adapter_keys import ADAPTER_KEYS
from mechbench_compute.adapters.read_lora_layers import read_lora_layers
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.ops.adapter.measure import measure_adapter
from mechbench_compute.peft import peft_export, peft_import
from tests.test_adapter_layers import SHARD, adapted, hold, read, write_snapshot
from tests.tiny_models import MODEL_TYPES, build_tiny_model

RECORDS = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]

PARAMS = {"model": "tiny", "target": {"uniform": ["cat", "dog"]}, "closer": " mat", "steps": 3,
          "lr": 0.05, "seed": 3}


def train(model, layers=None, **ctx):
    shape = {"rank": 2, "alpha": 4, **({} if layers is None else {"layers": layers})}
    out = train_op.run(Context(loaded=model, **ctx), {"records": RECORDS}, {**PARAMS, "lora": shape})
    return out["out"]


def read_weights(payload, tmp_path):
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(payload["data"])
    return dict(mx.load(str(path)))


def read_key_layers(payload, tmp_path):
    return sorted({int(ADAPTER_KEYS.key_re.match(k).group(1)) for k in read_weights(payload, tmp_path)})


def count_wrapped(model):
    return sum(isinstance(m, lora.LoRALinear) for _, m in model.lm.named_modules())


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param)


class TestReadingTheSelection:
    @pytest.mark.parametrize(("spec", "want"), [
        ("all", [0, 1, 2, 3]), (None, [0, 1, 2, 3]), ([-2, -1], [2, 3]), ([3, 0], [0, 3]),
        ([1, -1], [1, 3]), ({"from": 0, "to": 2}, [0, 1]), ({"from": -2}, [2, 3]),
        ({"to": -1}, [0, 1, 2]), ({}, [0, 1, 2, 3]), ({"from": 1, "to": 4}, [1, 2, 3]),
        ({"from": -4, "to": -3}, [0]),
    ])
    def test_a_selection_resolves_like_python(self, spec, want):
        assert read_lora_layers(spec, 4) == want

    @pytest.mark.parametrize(("spec", "said"), [
        ([4], "names 4, and this model has 4 layers, 0 through 3 (-4 through -1 from the end)"),
        ([0, -5], "names -5, and this model has 4 layers"),
        ([], "is empty; this model has 4 layers"),
        ([1, -3], "names layer 1 more than once"),
        ({"from": 2, "to": 2}, "selects no layer; this model has 4 layers"),
        ({"from": 3, "to": 1}, "selects no layer"),
        ({"from": 4}, "starts at 4, and this model has 4 layers"),
        ({"to": 5}, "ends at 5, and this model has 4 layers"),
        ({"from": -5}, "starts at -5"),
    ])
    def test_a_selection_the_model_cannot_meet_refuses_naming_its_layer_count(self, spec, said):
        with pytest.raises(ValueError, match=r"^adapter/train: `lora.layers` ") as caught:
            read_lora_layers(spec, 4)
        assert said in str(caught.value)

    @pytest.mark.parametrize("spec", ["3", 3, [True], [1.0], [[1]], {"start": 1}, {"from": "1"}, "last"])
    def test_a_selection_out_of_form_refuses_with_the_form(self, spec):
        with pytest.raises(ValueError, match=r'`lora.layers` is "all", a list of layer indices'):
            read_lora_layers(spec, 4)


class TestEveryLayerIsTodaysAdapter:
    def test_all_and_every_index_train_the_adapter_byte_for_byte(self, tiny):
        n = len(tiny.lm.model.layers)
        unnamed = train(tiny)
        for layers in ("all", list(range(n)), {"from": 0}, list(range(-n, 0))):
            named = train(tiny, layers)
            assert named["data"] == unnamed["data"]
            assert named["lora"] == unnamed["lora"]
            assert named["train"] == unnamed["train"]
        assert "layers" not in unnamed["lora"]


class TestTrainingOnChosenLayers:
    @pytest.mark.parametrize(("layers", "want"), [([-2, -1], "last two"), ({"from": 0, "to": 2}, [0, 1])])
    def test_the_adapter_holds_weights_for_those_layers_only(self, tiny, tmp_path, layers, want):
        n = len(tiny.lm.model.layers)
        chosen = [n - 2, n - 1] if want == "last two" else want
        payload = train(tiny, layers)
        assert read_key_layers(payload, tmp_path) == chosen
        assert payload["lora"]["layers"] == chosen
        everywhere = read_weights(train(tiny), tmp_path)
        mine = read_weights(payload, tmp_path)
        assert set(mine) == {k for k in everywhere if int(k.split(".")[2]) in chosen}
        assert payload["lora"]["params"] == sum(w.size for w in mine.values())
        assert count_wrapped(tiny) == 0

    def test_a_chosen_layer_starts_from_the_weights_the_full_run_draws_there(self, tiny):
        lora.apply_lora(tiny.lm, 2, 4, seed=3)
        try:
            full = {k: np.array(v) for k, v in tree_flatten(tiny.lm.trainable_parameters())}
        finally:
            lora.remove_lora(tiny.lm)
        lora.apply_lora(tiny.lm, 2, 4, seed=3, layers=[1])
        try:
            one = {k: np.array(v) for k, v in tree_flatten(tiny.lm.trainable_parameters())}
        finally:
            lora.remove_lora(tiny.lm)
        assert one and all(k.startswith("model.layers.1.") for k in one)
        assert all(np.array_equal(one[k], full[k]) for k in one)

    def test_a_layer_the_model_lacks_refuses_and_leaves_the_model_bare(self, tiny):
        n = len(tiny.lm.model.layers)
        with pytest.raises(ValueError, match=rf"names {n}, and this model has {n} layers"):
            train(tiny, [0, n])
        assert count_wrapped(tiny) == 0

    def test_a_target_on_no_chosen_layer_refuses(self):
        from tests.test_apply_lora import FakeLM

        lm = FakeLM([True, False, True])
        with pytest.raises(ValueError, match="exist on no chosen layer"):
            lora.apply_lora(lm, rank=2, alpha=4, layers=[1])
        assert lora.remove_lora(lm) == 0


@pytest.fixture(scope="module")
def last_two(tiny):
    return train(tiny, {"from": -2})


class TestTheAdapterActsWhereItWasTrained:
    def test_a_fused_reference_leaves_the_frozen_layers_exactly_as_the_base(self, tiny, last_two):
        n = len(tiny.lm.model.layers)
        executor = hold(tiny)
        bare = read(executor, "tiny")
        fused = read(executor, adapted(last_two))
        assert [fused["residuals"][i] == bare["residuals"][i] for i in range(n)] == [
            i < n - 2 for i in range(n)]
        assert fused["logits"] != bare["logits"]
        assert read(executor, "tiny") == bare

    def test_fusing_by_hand_moves_only_the_chosen_layers(self, tiny, last_two, tmp_path):
        n = len(tiny.lm.model.layers)
        handle = lora.fuse(tiny.lm, read_weights(last_two, tmp_path), scale=2.0)
        try:
            assert sorted({i for i, _c, _p in handle}) == [n - 2, n - 1]
        finally:
            lora.restore(tiny.lm, handle)

    def test_merging_changes_only_the_chosen_layers_weights(self, tiny, last_two, tmp_path):
        n = len(tiny.lm.model.layers)
        base = write_snapshot(tiny, tmp_path / "snap")
        checkpoint.export_merged(tmp_path / "snap", [last_two], tmp_path / "out")
        merged = dict(mx.load(str(tmp_path / "out" / SHARD)))
        moved = {name for name in base if not np.array_equal(np.array(merged[name]), np.array(base[name]))}
        assert moved
        assert {int(name.split(".")[2]) for name in moved} == {n - 2, n - 1}

    def test_measuring_reads_the_chosen_layers_and_carries_them(self, tiny, last_two):
        n = len(tiny.lm.model.layers)
        out = measure_adapter(last_two)
        assert sorted({item["coords"]["layer"] for item in out["items"]}) == [n - 2, n - 1]
        assert out["measured"]["layers"] == [n - 2, n - 1]
        assert out["lora"]["layers"] == [n - 2, n - 1]
        assert "layers" not in measure_adapter(train(tiny))["lora"]

    def test_a_peft_export_names_the_layers_and_imports_them_back(self, tiny, last_two, tmp_path):
        n = len(tiny.lm.model.layers)
        peft_export(last_two, str(tmp_path / "peft"))
        config = json.loads((tmp_path / "peft" / "adapter_config.json").read_text())
        assert config["layers_to_transform"] == [n - 2, n - 1]
        back = peft_import(str(tmp_path / "peft"))
        assert back["lora"]["layers"] == [n - 2, n - 1]
        assert read_weights(back, tmp_path).keys() == read_weights(last_two, tmp_path).keys()


class TestRunsOnOtherLayersAreOtherRuns:
    def test_the_node_fingerprint_sees_the_layers(self):
        def fingerprint(layers):
            shape = {"rank": 2, "alpha": 4, **({} if layers is None else {"layers": layers})}
            return resume.node_fingerprint(block="adapter/train", params={**PARAMS, "lora": shape},
                                           input_hashes=[], core_version="0.0.0", model="tiny")

        prints = {fingerprint(x) for x in (None, [-1], [-2, -1], {"from": 0, "to": 2})}
        assert len(prints) == 4

    def test_a_resume_from_a_run_on_other_layers_refuses(self):
        tiny = build_tiny_model("llama")
        states: list[dict] = []
        params = {**PARAMS, "steps": 4, "checkpoint_every": 2, "lora": {"rank": 2, "alpha": 4, "layers": [0]}}
        train_op.run(Context(loaded=tiny, on_checkpoint=states.append), {"records": RECORDS}, params)
        with pytest.raises(ValueError, match=r"saved training state holds 4 weights on layers 0 and "
                                             r"this node's adapter 8 weights on layers 0, 1: its `lora` differs"):
            train_op.run(Context(loaded=tiny, resume_state=states[0]), {"records": RECORDS},
                         {**params, "lora": {"rank": 2, "alpha": 4, "layers": [0, 1]}})
        assert count_wrapped(tiny) == 0

    def test_a_resume_on_the_same_layers_finishes_the_same_adapter(self):
        tiny = build_tiny_model("llama")
        states: list[dict] = []
        params = {**PARAMS, "steps": 4, "checkpoint_every": 2, "lora": {"rank": 2, "alpha": 4, "layers": [-1]}}
        whole = train_op.run(Context(loaded=tiny, on_checkpoint=states.append), {"records": RECORDS}, params)
        resumed = train_op.run(Context(loaded=tiny, resume_state=states[0]), {"records": RECORDS}, params)
        assert resumed["out"]["data"] == whole["out"]["data"]
