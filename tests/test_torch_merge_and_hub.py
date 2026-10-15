from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

if importlib.util.find_spec("nnsight") is None:
    pytest.skip("nnsight is not installed: pip install 'mechbench-compute[torch]'",
                allow_module_level=True)
torch = pytest.importorskip("torch")

from safetensors.torch import load_file
from safetensors.torch import save as save_tensors
from transformers import AutoModelForCausalLM

from mechbench_compute import backends, checkpoint
from mechbench_compute.arrays import read_f64
from mechbench_compute.torch_backend.lora import (
    fuse_adapter_stack,
    restore_adapter_stack,
)
from mechbench_compute.torch_backend.model import TorchModel
from tests.tiny_tokenizer import build_tokenizer
from tests.tiny_torch_models import build_tiny_model, make_adapter

ROOT = pathlib.Path(__file__).resolve().parent.parent

HAS_MLX = importlib.util.find_spec("mlx") is not None

IDS = [1, 5, 9, 2, 7, 3, 11, 4]

TOLERANCE = 1e-5

CHOSEN = (1, 3)

NAMES = ("llama", "qwen2", "qwen3")


def write_snapshot(model, root: pathlib.Path) -> pathlib.Path:
    snap = root / "snap"
    model._model.save_pretrained(snap, max_shard_size="64KB")
    assert (snap / "model.safetensors.index.json").exists()
    return snap


def make_payload(model) -> dict:
    weights = make_adapter(model, model.architecture.adapter_keys)
    return {"kind": "adapter/lora", "data": save_tensors(weights),
            "lora": {"rank": 2, "alpha": 4.0, "layers": list(CHOSEN)}}


def read_logits(model) -> np.ndarray:
    return read_f64(model.run(model.make_ids(IDS)).logits)


def load_on_torch(folder: pathlib.Path) -> TorchModel:
    return TorchModel(AutoModelForCausalLM.from_pretrained(folder).eval(), build_tokenizer())


def check_close(got: np.ndarray, want: np.ndarray, what: str) -> None:
    assert got.shape == want.shape and np.abs(want).max() > 0.25
    err = float(np.abs(got - want).max())
    assert err <= TOLERANCE * max(1.0, float(np.abs(want).max())), f"{what}: logits differ by {err:.3g}"


@pytest.fixture(scope="module", params=NAMES)
def merged(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    base = build_tiny_model(request.param)
    snap = write_snapshot(base, root)
    payload = make_payload(base)
    files = checkpoint.export_merged(snap, [payload], root / "torch", layers=[CHOSEN], arrays="torch")
    handles = fuse_adapter_stack(base.lm, [payload], layers=[CHOSEN])
    try:
        fused = read_logits(base)
    finally:
        restore_adapter_stack(base.lm, handles)
    return {"root": root, "base": base, "snap": snap, "payload": payload, "files": files, "fused": fused}


def test_the_torch_merge_runs_as_the_base_with_its_adapter_fused(merged):
    on_torch = load_on_torch(merged["root"] / "torch")
    check_close(read_logits(on_torch), merged["fused"], "merged against fused")
    assert not np.allclose(read_logits(merged["base"]), merged["fused"])


def test_the_torch_merge_moves_the_chosen_layers_only(merged):
    keys = merged["base"].architecture.adapter_keys
    moved = set()
    for shard in {f for f in merged["files"] if f.endswith(".safetensors")}:
        before, after = load_file(str(merged["snap"] / shard)), load_file(str(merged["root"] / "torch" / shard))
        assert before.keys() == after.keys()
        for name in before:
            if not torch.equal(before[name], after[name]):
                moved.add(name)
    layers = {int(name.split(".")[2]) for name in moved}
    projections = {name.split(".")[4] for name in moved}
    assert layers == set(CHOSEN) and projections <= set(keys.containers)


@pytest.mark.skipif(not HAS_MLX, reason="the MLX half needs MLX")
def test_the_torch_merge_loads_on_mlx_and_matches_the_mlx_merge(merged):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from tests.mlx_twins import PAIRS, build_on_mlx

    if merged["base"].architecture.model_type not in {t for _n, t in PAIRS}:
        pytest.skip(f"MLX does not load {merged['base'].architecture.model_type}")
    root = merged["root"]
    checkpoint.export_merged(merged["snap"], [merged["payload"]], root / "mlx", layers=[CHOSEN], arrays="mlx")
    for shard in (f for f in merged["files"] if f.endswith(".safetensors")):
        ours, theirs = load_file(str(root / "torch" / shard)), load_file(str(root / "mlx" / shard))
        for name in ours:
            assert np.allclose(ours[name].numpy(), theirs[name].numpy(), atol=1e-6), name

    on_mlx = build_on_mlx(merged["base"])
    wanted = dict(tree_flatten(on_mlx.lm.parameters()))
    loaded: dict = {}
    for shard in (f for f in merged["files"] if f.endswith(".safetensors")):
        loaded.update(mx.load(str(root / "torch" / shard)))
    assert set(wanted) <= set(loaded)
    on_mlx.lm.update(tree_unflatten([(k, loaded[k]) for k in wanted]))
    mx.eval(on_mlx.lm.parameters())
    check_close(read_logits(on_mlx), merged["fused"], "torch merge on mlx")
    check_close(read_logits(load_on_torch(root / "mlx")), merged["fused"], "mlx merge on torch")


def test_the_three_run_on_torch():
    torch_ops = next(b for b in backends.BACKENDS if b.name == "torch").ops
    assert {"adapter/merge", "adapter/measure", "adapter/publish", "dictionary/load"} <= set(torch_ops)


WITHOUT_MLX = """
import json, pathlib, sys, tempfile
from types import SimpleNamespace
for name in ("mlx", "mlx.core", "mlx.utils", "mlx.nn", "mlx_lm", "mlx_vlm"):
    sys.modules[name] = None
import huggingface_hub
import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import load as load_tensors, load_file, save as save_tensors
from mechbench_compute import hub, model_ref
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import measure, merge, publish
from mechbench_compute.ops.dictionary import encode, load
from tests.tiny_torch_models import build_tiny_model, make_adapter

root = pathlib.Path(tempfile.mkdtemp())
uploads = []

class FakeHub:
    def __init__(self, token=None):
        assert token == "hf_test"
    def create_repo(self, repo, **_):
        uploads.append(("create", repo))
    def upload_folder(self, repo_id, folder_path, **_):
        folder = pathlib.Path(folder_path)
        uploads.append(("upload", repo_id, {p.name: p.read_bytes() for p in folder.iterdir()}))
        return SimpleNamespace(oid="f" * 40)

huggingface_hub.HfApi = FakeHub
secrets = {"hf": {"token": "hf_test"}}

tiny = build_tiny_model("llama")
snap = root / "snap"
tiny._model.save_pretrained(snap, max_shard_size="64KB")
weights = make_adapter(tiny, tiny.architecture.adapter_keys)
payload = {"kind": "adapter/lora", "data": save_tensors(weights), "base_model": "org/base",
           "lora": {"rank": 2, "alpha": 4.0, "target_modules": ["q_proj"]}}
hub.ensure_model = lambda ref: ("org/base", "a" * 40, str(snap))
mref = model_ref.resolve({"base": "org/base", "adapters": [{"bench": "you/lab/die", "layers": [1, 3]}]},
                         fetch=lambda _label: payload)
pointer = merge.run(SimpleNamespace(secrets=secrets, on_start=None, on_item=None), {},
                    {"model": mref, "to": {"hf": {"repo": "you/merged"}}})
merged_files = uploads[-1][2]
assert pointer["base"] == {"hf": "you/merged@" + "f" * 40}, pointer
assert "model.safetensors.index.json" in merged_files and "config.json" in merged_files

delta = measure.measure_adapter(payload)
assert len(delta["items"]) == len(weights) // 2

pushed = publish.run(SimpleNamespace(secrets=secrets), {"adapter": payload}, {"repo": "you/die"})
files = uploads[-1][2]
peft = load_tensors(files["adapter_model.safetensors"])
stem = "base_model.model.model.layers.1.self_attn.q_proj"
assert torch.equal(peft[stem + ".lora_A.weight"], weights["model.layers.1.self_attn.q_proj.lora_a"])
assert pushed["hf_adapter_ref"] == {"repo": "you/die", "revision": "f" * 40}
assert json.loads(files["adapter_config.json"])["r"] == 2

from mechbench_compute.peft import peft_import
folder = root / "peft"
folder.mkdir()
for name, data in files.items():
    (folder / name).write_bytes(data)
back = load_tensors(peft_import(str(folder))["data"])
assert back.keys() == weights.keys() and all(torch.equal(back[k], weights[k]) for k in weights)

gemma = build_tiny_model("gemma3")
d = gemma.arch.d_model
sae = root / "sae" / "resid_post" / "layer_2"
sae.mkdir(parents=True)
(sae / "config.json").write_text(json.dumps({
    "hf_hook_point_in": "model.layers.2.output", "width": d, "model_name": "tiny/gemma3",
    "architecture": "jump_relu", "type": "sae"}))
save_file({"w_enc": np.eye(d, dtype=np.float32), "b_enc": np.zeros(d, dtype=np.float32),
           "threshold": np.full(d, -1e30, dtype=np.float32), "w_dec": np.eye(d, dtype=np.float32),
           "b_dec": np.zeros(d, dtype=np.float32)}, str(sae / "params.safetensors"))
load.fetch_folder = lambda repo, path, rev: (root / "sae" / path, "c" * 40)
load.read_architecture = lambda model_id: "gemma3"
dictionary = load.run(Context(), {}, {"repo": "google/gemma-scope-tiny", "path": "resid_post/layer_2"})
out = encode.run(Context(loaded=gemma), {"records": [{"id": "a", "user": "the cat sat"}],
                                         "dictionary": dictionary}, {})
fidelity = out["fidelity"]
assert fidelity["variance_explained"] >= 0.999999 and fidelity["l0"] == d, fidelity
print(len(uploads), sys.modules["mlx"] is not None)
"""


def test_merge_publish_measure_and_a_loaded_dictionary_run_where_mlx_cannot_be_imported():
    out = subprocess.run([sys.executable, "-c", WITHOUT_MLX], cwd=ROOT, capture_output=True,
                         text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["4", "False"]
