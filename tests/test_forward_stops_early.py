from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mechbench_compute import _arch
from mechbench_compute.model import Model
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from tests.tiny_models import KIT_MODELS, build_gemma4, build_tiny_model
from tests.tiny_tokenizer import build_tokenizer

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

SHARED = 2


class Refuses(nn.Module):
    def __call__(self, *args, **kwargs):
        raise AssertionError("a layer past the last one read ran")


def build_shared() -> Model:
    from mlx.utils import tree_map

    wrapped, lm = build_gemma4(num_kv_shared_layers=SHARED)
    keys = iter(mx.random.split(mx.random.key(11), 1000))
    lm.update(tree_map(lambda p: mx.random.normal(p.shape, key=next(keys)) * 0.5,
                       lm.parameters()))
    mx.eval(lm.parameters())
    return Model(wrapped, build_tokenizer())


def build(name: str) -> Model:
    return build_shared() if name == "gemma4-shared" else build_tiny_model(name)


NAMES = [name for name, _ in KIT_MODELS] + ["gemma4-shared"]


def read_points(model: Model, layers) -> list[str]:
    points = model.architecture.layer_points_of(model.arch)
    return [f"blocks.{i}.{p}" for i in layers for p in points
            if not (p in _arch.SHARED_LAYER_ABSENT_POINTS
                    and i >= model.arch.first_kv_shared_layer)]


def refuse_after(model: Model, layer: int) -> None:
    model.lm.model.layers[layer + 1].mlp = Refuses()


@pytest.mark.parametrize("name", NAMES)
def test_a_read_that_stops_early_is_bit_identical_to_the_whole_forward(name):
    model = build(name)
    ids = model.make_ids(IDS)
    for last in range(model.arch.n_layers):
        names = read_points(model, range(last + 1))
        whole = model.run(ids, capture=names)
        early = model.run(ids, capture=names, logits=False)
        assert early.logits is None and whole.logits is not None
        for n in names:
            a = np.array(whole.cache[n].astype(mx.float32))
            b = np.array(early.cache[n].astype(mx.float32))
            assert a.dtype == b.dtype and np.array_equal(a, b), f"{name} {n} (stopping at {last})"


def test_gemma4_shared_reads_either_side_of_the_first_shared_layer():
    for last in (SHARED - 1, SHARED):
        model = build_shared()
        assert model.arch.first_kv_shared_layer == SHARED
        assert model.arch.hidden_size_per_layer_input > 0
        ids = model.make_ids(IDS)
        names = read_points(model, [last])
        assert f"blocks.{last}.gate_out" in names and f"blocks.{last}.attn.k" in names
        whole = model.run(ids, capture=names)
        if last + 1 < model.arch.n_layers:
            refuse_after(model, last)
        early = model.run(ids, capture=names, logits=False)
        for n in names:
            assert np.array_equal(np.array(whole.cache[n]), np.array(early.cache[n])), n


@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
def test_the_forward_stops_after_the_last_layer_read(name):
    model = build(name)
    ids = model.make_ids(IDS)
    last = model.arch.n_layers // 2 - 1
    refuse_after(model, last)
    early = model.run(ids, capture=[f"blocks.{last}.resid_post"], logits=False)
    assert early.logits is None and f"blocks.{last}.resid_post" in early.cache
    hooked = model.run(ids, hooks={f"blocks.{last}.mlp_out": lambda a, info: a * 2},
                       capture=["embed"], logits=False)
    assert "embed" in hooked.cache


@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
@pytest.mark.parametrize("request_", [
    {"capture": ["blocks.0.resid_post"]},
    {"capture": ["blocks.0.resid_post", "blocks.2.resid_post"], "logits": False},
    {"capture": ["blocks.0.resid_post"], "hooks": {"blocks.2.attn_out": lambda a, info: a},
     "logits": False},
    {"capture": ["blocks.0.resid_post", "final_norm"], "logits": False},
    {"capture": ["logits"], "logits": False},
    {"capture": ["blocks.0.resid_post"], "logits": False, "kv_cache": True},
], ids=["logits", "a-later-capture", "a-later-hook", "final-norm", "logits-captured",
        "a-kv-cache"])
def test_a_request_that_reads_past_the_layer_runs_the_whole_model(name, request_):
    model = build(name)
    kwargs = dict(request_)
    if kwargs.get("kv_cache"):
        kwargs["kv_cache"] = model.prompt_cache()
    refuse_after(model, 0)
    with pytest.raises(AssertionError, match="past the last one read"):
        model.run(model.make_ids(IDS), **kwargs)


@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
def test_capture_reads_only_the_layers_it_names(name):
    model = build(name)
    record = {"id": "r", "user": "the cat sat on a mat"}
    want = capture_residual_vectors(model, [record], {"layers": [1], "storage": "json"})
    refuse_after(model, 1)
    got = capture_residual_vectors(model, [record], {"layers": [1], "storage": "json"})
    assert [it["vector"] for it in got["items"]] == [it["vector"] for it in want["items"]]
