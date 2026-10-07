from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None or importlib.util.find_spec("mlx") is None:
    pytest.skip("the cross-backend check needs both torch (with nnsight) and MLX",
                allow_module_level=True)

from mechbench_compute import directions as dirs
from mechbench_compute import shapes as S
from mechbench_compute.ops import Context
from tests.mlx_twins import PAIRS, build_on_mlx
from tests.tiny_torch_models import build_tiny_model

TOLERANCE = 1e-3

# external: mlx-vlm models/gemma3/language.py — rounds Gemma 3's embedding scale to bf16 where transformers keeps the weights' dtype, so float32 weights part by about 1e-4 from the embedding on
TOLERANCE_OF = {"gemma3": 5e-3}

RECORDS = [{"id": "a", "user": "the cat sat on a", "tracked": {"answer": "mat"}},
           {"id": "b", "prompt": "the dog ran on the cat and the", "template": "raw",
            "tracked": {"answer": "cat"}}]

PAIRED = [{"id": "p", "a": "the cat sat on the", "b": "the dog sat on the", "template": "raw",
           "tracked": {"answer": "mat"}}]

LABELED = [{"id": "x1", "prompt": "the cat sat", "template": "raw", "coords": {"label": "cat"}},
           {"id": "x2", "prompt": "a cat sat on the mat", "template": "raw", "coords": {"label": "cat"}},
           {"id": "y1", "prompt": "the dog ran", "template": "raw", "coords": {"label": "dog"}},
           {"id": "y2", "prompt": "a dog ran on the mat", "template": "raw", "coords": {"label": "dog"}}]

UNREAD = ("backend", "accelerator", "attention")


def run(model, module, inputs, params):
    op = importlib.import_module(f"mechbench_compute.ops.{module}")
    return op.run(Context(loaded=model), inputs, {"model": "tiny", **params})


def capture_vectors(model, layer=1):
    return run(model, "activations.capture", {"records": LABELED}, {"layers": [layer]})


def make_direction(model, layer, seed):
    v = np.random.default_rng(seed).normal(size=model.arch.d_model).astype(np.float32)
    return dirs.make(v, S.space(model=S.model_id_of(model), layer=layer, point="resid_post", d=len(v)),
                     method="t")


def applying(spec, **params):
    return lambda m: ("intervene.apply", {"records": RECORDS}, {"spec": spec, **params})


CASES = {
    "operator": applying([{"point": "resid_post", "layers": [1], "f": "x * 2 if x > 0 else exp(x) - 1",
                           "positions": "all"}]),
    "operator-masked-with-constants": applying([
        {"point": "resid_post", "layers": [1, 2], "f": "min(max(x, -k), k) + c",
         "constants": {"k": 0.3, "c": [0.1, -0.2, 0.05]}, "mask": [0, 3, 5]}]),
    "operator-except": applying([{"point": "mlp_out", "layers": [2], "f": "x / 2 + x ** 3", "mask": [1, 2],
                                  "except": True, "positions": "all"}]),
    "operator-in-a-frame": lambda m: ("intervene.apply", {"records": RECORDS}, {"spec": [
        {"point": "resid_post", "layers": [1], "f": "x * 0", "positions": "all",
         "mask": [make_direction(m, 1, 4), make_direction(m, 1, 5)]}]}),
    "operator-bound-from-source": lambda m: ("intervene.apply", {"records": RECORDS, "source": capture_vectors(m)},
                                             {"spec": [{"point": "resid_post", "layers": [1], "f": "mu",
                                                        "constants": {"mu": {"source": "mean"}},
                                                        "mask": [0, 1, 2, 3]}]}),
    "head-and-edge-ablation": applying([
        {"point": "attn.per_head_out", "layers": [2], "op": "zero", "heads": [1]},
        {"point": "resid_post", "layers": [0], "op": "scale", "strength": 0.5, "positions": "all"},
        {"point": "attn.weights", "layers": [3], "op": "zero", "pattern": {"from": [0], "to": "last"}}]),
    "directions": lambda m: ("intervene.apply", {"records": RECORDS}, {"spec": [
        {"point": "resid_post", "layers": [1], "op": "rotate", "direction": make_direction(m, 1, 1),
         "direction2": make_direction(m, 1, 2), "strength": 0.7, "positions": "all"},
        {"point": "resid_post", "layers": [2], "op": "clamp", "direction": make_direction(m, 2, 6),
         "strength": 0.2}]}),
    "sweep": applying([{"point": "mlp_out", "layers": [1], "op": "scale", "strength": 0.0}],
                      sweep={"strength": [0.0, 0.5, 2.0]}),
    "capture-readout": applying([{"point": "mlp_out", "layers": [1], "op": "scale", "strength": 0.5}],
                                readout={"type": "capture", "points": ["blocks.2.resid_post"]}),
    "ablate-heads": lambda m: ("intervene.ablate_heads", {"records": RECORDS}, {"layers": [0, 2]}),
    "ablate-heads-logit": lambda m: ("intervene.ablate_heads", {"records": RECORDS}, {"metric": "logit"}),
    "ablate-layers": lambda m: ("intervene.ablate_layers", {"records": RECORDS}, {}),
    "steer": lambda m: ("intervene.steer", {"records": RECORDS, "vectors": capture_vectors(m)},
                        {"layer": 1, "direction": {"positive": "cat", "negative": "dog"},
                         "alphas": [-2.0, 0.0, 3.0]}),
    "patch": lambda m: ("intervene.patch", {"records": PAIRED}, {"layers": [0, 2]}),
    "patch-logit": lambda m: ("intervene.patch", {"records": PAIRED}, {"metric": "logit", "point": "resid_pre"}),
    "patch-attribution": lambda m: ("intervene.patch", {"records": PAIRED},
                                    {"method": "attribution", "point": "attn_out"}),
}


def read_leaves(x, path=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from read_leaves(x[k], f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from read_leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def check_agree(want, got, tolerance, what):
    a = {k: v for k, v in read_leaves(want) if not any(u in k for u in UNREAD)}
    b = {k: v for k, v in read_leaves(got) if not any(u in k for u in UNREAD)}
    assert set(a) == set(b), f"{what}: {sorted(set(a) ^ set(b))[:5]}"
    for k, x in a.items():
        y = b[k]
        if isinstance(x, float) or (isinstance(x, int) and not isinstance(x, bool) and isinstance(y, float)):
            assert abs(x - y) <= tolerance * max(1.0, abs(x)), f"{what} {k}: {x} vs {y}"
        else:
            assert x == y, f"{what} {k}: {x!r} vs {y!r}"


@pytest.fixture(scope="module", params=[n for n, _ in PAIRS])
def twins(request):
    on_torch = build_tiny_model(request.param)
    return request.param, on_torch, build_on_mlx(on_torch)


@pytest.mark.parametrize("case", list(CASES))
def test_an_intervention_reads_the_same_on_mlx_and_torch(twins, case):
    name, on_torch, on_mlx = twins
    want = run(on_mlx, *CASES[case](on_mlx))
    got = run(on_torch, *CASES[case](on_torch))
    assert got.get("items") or got.get("measures")
    check_agree(want, got, TOLERANCE_OF.get(on_torch.architecture.model_type, TOLERANCE), f"{name} {case}")
