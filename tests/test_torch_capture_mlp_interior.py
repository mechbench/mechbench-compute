from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("the torch backend needs torch with nnsight", allow_module_level=True)

import torch

from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.torch_backend.architectures._decoder import read_text
from tests.test_capture_mlp_interior import IDS, INTERIOR, RECORD
from tests.test_torch_forward_stops_early import list_pairs, needs_mlx, refuse_after
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model


def read_plainly(model, ids) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    seen: dict[str, torch.Tensor] = {}
    handles = []

    def keep(name, *, inputs=False):
        def hook(module, args, out=None):
            seen[name] = (args[0] if inputs else out).detach().clone()
        return hook

    for i, layer in enumerate(read_text(model._model).layers):
        mlp = layer.mlp
        handles += [
            mlp.register_forward_pre_hook(keep(f"blocks.{i}.mlp.in_norm", inputs=True)),
            mlp.gate_proj.register_forward_hook(keep(f"blocks.{i}.mlp.gate")),
            mlp.up_proj.register_forward_hook(keep(f"blocks.{i}.mlp.up")),
            mlp.act_fn.register_forward_hook(keep(f"blocks.{i}.mlp.act")),
            mlp.down_proj.register_forward_pre_hook(keep(f"blocks.{i}.mlp.down_in", inputs=True)),
        ]
    try:
        with torch.no_grad():
            logits = model._model(input_ids=ids).logits
    finally:
        for h in handles:
            h.remove()
    return seen, logits


@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
def test_a_torch_interior_read_is_the_plain_forward_s_bit_for_bit(name):
    model = build_tiny_model(name)
    ids = model.make_ids(IDS)
    names = [f"blocks.{i}.{p}" for i in range(model.arch.n_layers) for p in INTERIOR]
    plain, logits = read_plainly(model, ids)
    traced = model.run(ids, capture=names)
    assert torch.equal(traced.logits, logits), name
    for n in names:
        assert torch.equal(traced.cache[n], plain[n]), f"{name} {n}"


@pytest.mark.parametrize("point", INTERIOR)
@pytest.mark.parametrize("name", [n for n, _ in KIT_MODELS])
def test_a_torch_interior_read_stops_after_its_layer(name, point):
    model = build_tiny_model(name)
    layer = model.arch.n_layers // 2 - 1
    params = {"layers": [layer], "point": point, "storage": "json"}
    want = capture_residual_vectors(model, [RECORD], params)
    refuse_after(model, layer)
    got = capture_residual_vectors(model, [RECORD], params)
    assert [it["vector"] for it in got["items"]] == [it["vector"] for it in want["items"]]


@needs_mlx
@pytest.mark.parametrize("point", ["mlp.act", "mlp.down_in"])
@pytest.mark.parametrize("name", list_pairs())
def test_mlx_and_torch_capture_the_same_neurons(name, point):
    from tests.mlx_twins import build_on_mlx
    from tests.test_torch_matches_mlx import TOLERANCE, TOLERANCE_OF

    on_torch = build_tiny_model(name)
    on_mlx = build_on_mlx(on_torch)
    tolerance = TOLERANCE_OF.get(on_torch.architecture.model_type, TOLERANCE)
    params = {"point": point, "storage": "json"}
    a = capture_residual_vectors(on_mlx, [RECORD], params)
    b = capture_residual_vectors(on_torch, [RECORD], params)
    assert a["widths"] == b["widths"]
    for x, y in zip(a["items"], b["items"]):
        u, v = np.array(x["vector"]), np.array(y["vector"])
        err = float(np.abs(u - v).max())
        assert err <= tolerance * max(1.0, float(np.abs(u).max())) + 1e-5, (
            f"{name} {point} at {x['space']['layer']}: {err:.3g}")
