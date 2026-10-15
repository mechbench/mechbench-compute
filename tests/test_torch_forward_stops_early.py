from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("the torch backend needs torch with nnsight", allow_module_level=True)

import torch

from mechbench_compute.arrays import read_f64
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model

IDS = [1, 5, 9, 2, 7, 3, 11, 4]


class Refuses(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("a layer past the last one read ran")


def read_points(model, layers) -> list[str]:
    from mechbench_compute import _arch

    points = [p for p in model.architecture.layer_points_of(model.arch)
              if model.architecture.supports(p, layer_scoped=True)]
    return [f"blocks.{i}.{p}" for i in layers for p in points
            if not (p in _arch.SHARED_LAYER_ABSENT_POINTS
                    and i >= model.arch.first_kv_shared_layer)]


def refuse_after(model, layer: int) -> None:
    from mechbench_compute.torch_backend.architectures._decoder import read_text

    read_text(model._model).layers[layer + 1].mlp = Refuses()


def read_last_layers(model) -> list[int]:
    n, shared = model.arch.n_layers, model.arch.first_kv_shared_layer
    return sorted({n // 2 - 1, n - 1} | ({shared - 1, shared} if shared < n else set()))


@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
def test_a_torch_read_that_stops_early_is_bit_identical_to_the_whole_forward(name):
    model = build_tiny_model(name)
    ids = model.make_ids(IDS)
    for last in read_last_layers(model):
        names = read_points(model, [last])
        whole = model.run(ids, capture=names)
        early = model.run(ids, capture=names, logits=False)
        assert early.logits is None and whole.logits is not None
        for n in names:
            assert torch.equal(whole.cache[n], early.cache[n]), f"{name} {n} (stopping at {last})"


@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
def test_the_torch_forward_stops_after_the_last_layer_read(name):
    model = build_tiny_model(name)
    last = model.arch.n_layers // 2 - 1
    refuse_after(model, last)
    ids = model.make_ids(IDS)
    early = model.run(ids, capture=[f"blocks.{last}.resid_post"], logits=False)
    assert early.logits is None and f"blocks.{last}.resid_post" in early.cache
    for kwargs in ({"capture": [f"blocks.{last}.resid_post"]},
                   {"capture": [f"blocks.{last + 1}.resid_post"], "logits": False},
                   {"capture": [f"blocks.{last}.resid_post", "final_norm"], "logits": False}):
        with pytest.raises(Exception, match="past the last one read"):
            model.run(ids, **kwargs)


needs_mlx = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                               reason="the cross-backend check needs MLX")


def list_pairs() -> list[str]:
    if importlib.util.find_spec("mlx") is None:
        return []
    from tests.mlx_twins import PAIRS

    return [n for n, _ in PAIRS]


@needs_mlx
@pytest.mark.parametrize("name", list_pairs())
def test_mlx_and_torch_agree_on_a_read_that_stops_early(name):
    from tests.mlx_twins import build_on_mlx
    from tests.test_torch_matches_mlx import TOLERANCE, TOLERANCE_OF

    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    tolerance = TOLERANCE_OF.get(on_torch.architecture.model_type, TOLERANCE)
    mlx_points = set(on_mlx.architecture.layer_points_of(on_mlx.arch))
    for last in read_last_layers(on_torch):
        names = [n for n in read_points(on_torch, [last]) if n.split(".", 2)[-1] in mlx_points]
        torch_run = on_torch.run(on_torch.make_ids(IDS), capture=names, logits=False)
        mlx_run = on_mlx.run(on_mlx.make_ids(IDS), capture=names, logits=False)
        assert torch_run.logits is None and mlx_run.logits is None
        for n in names:
            a, b = read_f64(mlx_run.cache[n]), read_f64(torch_run.cache[n])
            assert a.shape == b.shape, (n, a.shape, b.shape)
            err = float(np.abs(a - b).max())
            assert err <= tolerance * max(1.0, float(np.abs(a).max())), f"{name} {n}: {err:.3g}"
