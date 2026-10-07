from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from mechbench_compute import backends
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

FIXTURES = Path(__file__).parent / "fixtures" / "calibration"

VERSIONS = {2: "calibration.protocol.json", 3: "calibration-v3.protocol.json"}

HAS_TORCH = importlib.util.find_spec("nnsight") is not None and importlib.util.find_spec("torch") is not None
HAS_MLX = backends.is_importable("mlx.core")

TINY_LAYERS = 4

MODEL = "tiny/gemma3@rev"

TOKENS_ON_TORCH_NOW = {"activations/capture-tokens"}

WORDS = ("the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran")


def tiny_text(n: int, row: int) -> str:
    return " ".join(WORDS[(i * 7 + row * 3) % len(WORDS)] for i in range(max(2, min(n, 24))))


def tiny_records(ref: str) -> list[dict]:
    family, _, shape = ref.partition("/")
    if family == "forward":
        n, b = (int(part[1:]) for part in shape.split("-"))
        return [{"id": f"n{n}-r{r}", "prompt": tiny_text(n, r), "coords": {"n": n, "row": r}}
                for r in range(b)]
    if family == "decode":
        return [{"id": "decode-story", "user": "the cat sat on a"}]
    if family == "numeric":
        return [{"id": f"{topic}-{k:02d}", "coords": {"topic": topic},
                 "prompt": f"the {topic} sat on a mat and the {topic} ran"}
                for topic in ("cat", "dog") for k in range(8)]
    if family == "lora":
        return [{"id": f"color-{i}", "user": "the cat sat on a"} for i in range(4)]
    raise AssertionError(f"no tiny stand-in for {ref}")


def shrink_layers(value):
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return sorted({min(x, TINY_LAYERS - 1) for x in value})
    return value


def shrink(protocol: dict) -> dict:
    graph = copy.deepcopy(protocol["graph"])
    for node in graph["nodes"]:
        for port, value in list(node["inputs"].items()):
            if isinstance(value, dict) and "$ref" in value:
                ref = value["$ref"]["bench"].removeprefix("benjismith/calibration/")
                node["inputs"][port] = tiny_records(ref)
        params = node["params"]
        params["model"] = MODEL
        if "layers" in params:
            params["layers"] = shrink_layers(params["layers"])
        for item in params.get("spec") or []:
            item["layers"] = shrink_layers(item["layers"])
        if node["block"] == "text/generate":
            params["max_tokens"] = min(params["max_tokens"], 4)
        if node["block"] == "adapter/train":
            params.update(steps=2, lr=0.05, closer=" ran",
                          target={"uniform": ["mat", "dog", "cat"]},
                          lora={**params["lora"], "rank": 2, "alpha": 4})
    return graph


def run(protocol: dict, backend: str):
    out = ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
        "graph": shrink(protocol), "requirements": {"class": "local", "backend": backend}}))
    return out.payload if hasattr(out, "payload") else out


def read_protocol(version: int) -> dict:
    return json.loads((FIXTURES / VERSIONS[version]).read_text())


def test_v3_keeps_every_v2_node_under_its_id_and_shape():
    v2, v3 = read_protocol(2), read_protocol(3)
    by_id = {n["id"]: n for n in v3["graph"]["nodes"]}
    for node in v2["graph"]["nodes"]:
        kept = by_id[node["id"]]
        assert kept["block"] == node["block"] and kept["inputs"] == node["inputs"], node["id"]
        extra = {k: v for k, v in kept["params"].items() if node["params"].get(k) != v}
        assert set(extra) <= {"batch", "lora"}, (node["id"], extra)
    assert v3["graph"]["edges"] == v2["graph"]["edges"]
    assert v3["outputs"] == v2["outputs"]
    assert all(len(p[field]) <= 32 for p in (v2, v3) for field in ("params", "inputs", "outputs"))


def test_every_layer_the_protocols_read_is_below_22():
    for version in VERSIONS:
        for node in read_protocol(version)["graph"]["nodes"]:
            layers = [*(node["params"].get("layers") if isinstance(node["params"].get("layers"), list) else []),
                      *(x for item in node["params"].get("spec") or [] for x in item["layers"])]
            assert all(x <= 21 for x in layers), node["id"]


@pytest.fixture
def mlx_job(monkeypatch):
    from mechbench_compute.model import Model
    from tests.tiny_models import build_tiny_model

    tiny = build_tiny_model("gemma3")
    monkeypatch.setattr(Model, "load", classmethod(lambda cls, model_id, **_: tiny))
    return tiny


@pytest.fixture
def torch_job(monkeypatch):
    from mechbench_compute.torch_backend.model import TorchModel
    from tests.tiny_torch_models import build_tiny_model

    tiny = build_tiny_model("gemma3")
    monkeypatch.setattr(TorchModel, "load", classmethod(lambda cls, model_id, **_: tiny))
    return tiny


@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed here")
@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_the_calibration_protocol_runs_whole_on_mlx(mlx_job, version):
    protocol = read_protocol(version)
    payload = run(protocol, "mlx")
    ran = set(payload["node_summaries"])
    assert ran == {n["id"] for n in protocol["graph"]["nodes"]}
    if version == 3:
        assert_one_intervened_token(payload["outputs"]["intervene-n8192-b1"])


def assert_one_intervened_token(out: dict) -> None:
    assert [item["coords"]["factor"] for item in out["items"]] == [1.0]
    assert out["spec"] == [{"point": "mlp_out", "layers": [TINY_LAYERS - 1], "positions": "all",
                            "op": "zero"}]


def torch_refuses(protocol: dict) -> set[str]:
    torch_ops = set(next(b for b in backends.BACKENDS if b.name == "torch").ops)
    model_free = {"direction/classify", "geometry/compare"}
    return {n["id"] for n in protocol["graph"]["nodes"]
            if n["block"] not in torch_ops | model_free}


@pytest.mark.skipif(not HAS_TORCH, reason="nnsight is not installed: pip install 'mechbench-compute[torch]'")
@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_the_calibration_protocol_runs_on_torch_but_the_operations_torch_refuses(torch_job, version):
    protocol = read_protocol(version)
    refused = torch_refuses(protocol)
    assert {n["block"] for n in protocol["graph"]["nodes"] if n["id"] in refused} <= TOKENS_ON_TORCH_NOW
    if refused:
        with pytest.raises(backends.BackendRefused,
                           match=r"activations/capture-tokens does not run on the torch backend yet"):
            run(protocol, "torch")
    kept = copy.deepcopy(protocol)
    kept["graph"]["nodes"] = [n for n in kept["graph"]["nodes"] if n["id"] not in refused]
    payload = run(kept, "torch")
    assert set(payload["node_summaries"]) == {n["id"] for n in kept["graph"]["nodes"]}
    for nid, out in payload["outputs"].items():
        if nid.startswith("numeric-") and nid != "numeric-capture-k8-b128":
            continue
        assert out.get("backend") == "torch", nid
    if version == 3:
        assert payload["outputs"]["prefill-n2048-b16"]["batch"] == 16
        assert payload["outputs"]["decode-t128-n32"]["batch"] == 32
        assert_one_intervened_token(payload["outputs"]["intervene-n8192-b1"])
