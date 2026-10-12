from __future__ import annotations

import json
import mmap
import types

import pytest

torch = pytest.importorskip("torch")

import safetensors
import transformers.modeling_utils
from safetensors.torch import load_file, save_file

from mechbench_compute.torch_backend import streaming
from mechbench_compute.torch_backend.loading import load_transformers
from mechbench_compute.torch_backend.streaming import (
    StreamRefused,
    decide_streaming,
    read_checkpoint,
)
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model

CLASSES = {"gemma3": "Gemma3ForConditionalGeneration", "gemma4": "Gemma4ForConditionalGeneration"}

LEGACY_GEMMA3 = (("model.language_model.", "language_model."), ("model.vision_tower.", "vision_tower."),
                 ("model.multi_modal_projector.", "multi_modal_projector."),
                 ("lm_head.", "language_model.lm_head."))


def refuse_mapping(*args, **kwargs):
    raise AssertionError("the streaming reader memory-mapped a checkpoint")


def forbid_mmap(monkeypatch):
    monkeypatch.setattr(mmap, "mmap", refuse_mapping)
    monkeypatch.setattr(safetensors, "safe_open", refuse_mapping)
    monkeypatch.setattr(transformers.modeling_utils, "safe_open", refuse_mapping)


def save_checkpoint(tmp_path, name, **kwargs):
    tiny = build_tiny_model(name)
    tiny._model.save_pretrained(tmp_path, **kwargs)
    tiny.tokenizer.save_pretrained(tmp_path)
    return tiny


def load_reference(path, dtype):
    import transformers

    config = transformers.AutoConfig.from_pretrained(path)
    name = CLASSES.get(config.model_type, "AutoModelForCausalLM")
    return getattr(transformers, name).from_pretrained(path, dtype=dtype, device_map=torch.device("cpu"))


def load_streamed(path, dtype, monkeypatch):
    monkeypatch.setenv(streaming.SWITCH, "on")
    with monkeypatch.context() as m:
        forbid_mmap(m)
        model, _ = load_transformers(str(path), classes=CLASSES, device="cpu", dtype=dtype)
    return model


def read_ties(model):
    groups: dict[int, set[str]] = {}
    for key, p in model.named_parameters(remove_duplicate=False):
        groups.setdefault(p.data_ptr(), set()).add(key)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def assert_identical(got, want):
    for read in ("named_parameters", "named_buffers"):
        a = dict(getattr(want, read)(remove_duplicate=False))
        b = dict(getattr(got, read)(remove_duplicate=False))
        assert a.keys() == b.keys()
        for key, value in a.items():
            assert b[key].dtype == value.dtype, key
            assert b[key].device == value.device, key
            assert torch.equal(b[key], value), key
    assert read_ties(got) == read_ties(want)
    assert got.generation_config.to_dict() == want.generation_config.to_dict()
    assert got.config.to_dict() == want.config.to_dict()


def read_logits(model):
    ids = torch.tensor([[1, 5, 9, 2, 7, 3, 11, 4]])
    with torch.no_grad():
        return model(input_ids=ids).logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", [name for name, _ in KIT_MODELS])
def test_a_streamed_load_is_bit_identical_to_from_pretrained(tmp_path, monkeypatch, name, dtype):
    save_checkpoint(tmp_path, name)
    want = load_reference(tmp_path, dtype)
    got = load_streamed(tmp_path, dtype, monkeypatch)
    assert type(got) is type(want)
    assert not got.training
    assert_identical(got, want)
    assert torch.equal(read_logits(got), read_logits(want))


@pytest.mark.parametrize("name", ["gemma3-vlm", "gemma4-vlm", "llama"])
def test_a_sharded_checkpoint_streams_from_every_shard(tmp_path, monkeypatch, name):
    save_checkpoint(tmp_path, name, max_shard_size="20KB")
    shards = set(json.loads((tmp_path / streaming.INDEX).read_text())["weight_map"].values())
    assert len(shards) > 1
    assert {e.path.name for e in read_checkpoint(tmp_path).values()} == shards
    assert_identical(load_streamed(tmp_path, torch.float32, monkeypatch),
                     load_reference(tmp_path, torch.float32))


def rewrite_checkpoint(path, rename):
    single = path / streaming.SINGLE
    tensors = load_file(single)
    (path / streaming.SINGLE).unlink()
    save_file({rename(k): v for k, v in tensors.items()}, single, metadata={"format": "pt"})


def to_legacy(key):
    for new, old in LEGACY_GEMMA3:
        if key.startswith(new):
            return old + key[len(new):]
    return key


def test_a_checkpoint_in_the_older_layout_is_renamed_as_from_pretrained_renames_it(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, "gemma3-vlm")
    rewrite_checkpoint(tmp_path, to_legacy)
    assert all(not k.startswith("model.") for k in read_checkpoint(tmp_path))
    got = load_streamed(tmp_path, torch.bfloat16, monkeypatch)
    want = load_reference(tmp_path, torch.bfloat16)
    assert_identical(got, want)
    assert torch.equal(read_logits(got), read_logits(want))


def add_stray_key(path):
    tensors = load_file(path / streaming.SINGLE)
    tensors["model.stray.weight"] = torch.ones(3)
    save_file(tensors, path / streaming.SINGLE, metadata={"format": "pt"})


def drop_a_layer_norm(path):
    tensors = load_file(path / streaming.SINGLE)
    dropped = next(k for k in tensors if k.endswith("input_layernorm.weight"))
    del tensors[dropped]
    save_file(tensors, path / streaming.SINGLE, metadata={"format": "pt"})


@pytest.mark.parametrize("spoil", [add_stray_key, drop_a_layer_norm])
def test_a_checkpoint_the_streamer_cannot_map_loads_through_from_pretrained(tmp_path, monkeypatch, caplog, spoil):
    save_checkpoint(tmp_path, "llama")
    spoil(tmp_path)
    tensors = read_checkpoint(tmp_path)
    import transformers

    config = transformers.AutoConfig.from_pretrained(tmp_path)
    with pytest.raises(StreamRefused):
        streaming.load_streaming(transformers.LlamaForCausalLM, str(tmp_path), config=config,
                                 dtype=torch.float32, device=torch.device("cpu"))
    monkeypatch.setenv(streaming.SWITCH, "on")
    model, _ = load_transformers(str(tmp_path), classes={}, device="cpu", dtype=torch.float32)
    assert "streaming" in caplog.text and "refused" in caplog.text
    loaded = model.state_dict()
    assert all(torch.equal(loaded[k], tensors[k][...]) for k in tensors if k in loaded)


def test_the_reader_reads_with_plain_reads_not_a_memory_map(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, "qwen3")
    want = load_file(tmp_path / streaming.SINGLE)
    forbid_mmap(monkeypatch)
    got = read_checkpoint(tmp_path)
    assert got.keys() == want.keys()
    assert all(torch.equal(got[k][...], v) and got[k].get_shape() == list(v.shape) for k, v in want.items())


def test_a_header_whose_sizes_disagree_is_refused(tmp_path):
    path = tmp_path / streaming.SINGLE
    save_file({"w": torch.zeros(2, 3)}, path)
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])
    header["w"]["shape"] = [2, 4]
    text = json.dumps(header, separators=(",", ":")).encode().ljust(size)
    path.write_bytes(raw[:8] + text + raw[8 + size:])
    with pytest.raises(StreamRefused):
        read_checkpoint(tmp_path)


def test_a_folder_without_safetensors_is_refused(tmp_path):
    with pytest.raises(StreamRefused):
        read_checkpoint(tmp_path)


CUDA = types.SimpleNamespace(type="cuda", index=0)


def fake_gpu(monkeypatch, integrated):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: types.SimpleNamespace(is_integrated=integrated))


@pytest.mark.parametrize(("switch", "integrated", "device", "streams"), [
    (None, True, CUDA, True),
    ("auto", True, CUDA, True),
    ("auto", False, CUDA, False),
    ("auto", True, torch.device("cpu"), False),
    ("on", False, torch.device("cpu"), True),
    ("off", True, CUDA, False),
])
def test_streaming_is_on_for_an_integrated_gpu_and_by_the_switch(monkeypatch, switch, integrated, device, streams):
    fake_gpu(monkeypatch, integrated)
    if switch is None:
        monkeypatch.delenv(streaming.SWITCH, raising=False)
    else:
        monkeypatch.setenv(streaming.SWITCH, switch)
    assert decide_streaming(device) is streams


def test_an_unknown_switch_value_is_an_error(monkeypatch):
    monkeypatch.setenv(streaming.SWITCH, "sometimes")
    with pytest.raises(ValueError):
        decide_streaming(torch.device("cpu"))
