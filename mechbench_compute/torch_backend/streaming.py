from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SWITCH = "MECHBENCH_STREAM_LOAD"

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"

CHUNK = 64 * 2**20

DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}


class StreamRefused(Exception):
    pass


def decide_streaming(device: Any) -> bool:
    chosen = os.environ.get(SWITCH, "auto").strip().lower()
    if chosen in ("on", "1", "true"):
        return True
    if chosen in ("off", "0", "false"):
        return False
    if chosen != "auto":
        raise ValueError(f"{SWITCH} is on, off or auto, not {chosen!r}")
    if getattr(device, "type", None) != "cuda":
        return False
    import torch

    from mechbench_compute.torch_backend.memory import read_index

    return bool(getattr(torch.cuda.get_device_properties(read_index(device)), "is_integrated", False))


@dataclass(frozen=True)
class ShardTensor:
    path: Path
    start: int
    length: int
    dtype: str
    shape: tuple[int, ...]

    def get_shape(self) -> list[int]:
        return list(self.shape)

    def get_dtype(self) -> str:
        return self.dtype

    def __getitem__(self, index: Any) -> Any:
        tensor = read_tensor(self)
        return tensor if index is Ellipsis else tensor[index]


def read_tensor(entry: ShardTensor) -> Any:
    import torch

    dtype = getattr(torch, DTYPES[entry.dtype])
    if entry.length == 0:
        return torch.empty(entry.shape, dtype=dtype)
    buffer = bytearray(entry.length)
    view = memoryview(buffer)
    with open(entry.path, "rb", buffering=0) as f:
        f.seek(entry.start)
        done = 0
        while done < entry.length:
            n = f.readinto(view[done:done + CHUNK])
            if not n:
                raise StreamRefused(f"{entry.path.name} ends before its header says it does")
            done += n
    return torch.frombuffer(buffer, dtype=dtype).reshape(entry.shape)


def read_header(path: Path) -> dict[str, ShardTensor]:
    import torch

    with open(path, "rb", buffering=0) as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    base = 8 + size
    out: dict[str, ShardTensor] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        kind = meta["dtype"]
        if kind not in DTYPES or not hasattr(torch, DTYPES[kind]):
            raise StreamRefused(f"{path.name} stores {key} as {kind}, which the reader does not decode")
        begin, end = meta["data_offsets"]
        shape = tuple(int(n) for n in meta["shape"])
        numel = 1
        for n in shape:
            numel *= n
        if end - begin != numel * getattr(torch, DTYPES[kind]).itemsize:
            raise StreamRefused(f"{path.name} gives {key} {end - begin} bytes for shape {shape}")
        out[key] = ShardTensor(path, base + begin, end - begin, kind, shape)
    return out


def read_shards(root: Path) -> list[Path]:
    index = root / INDEX
    if index.is_file():
        files = json.loads(index.read_text())["weight_map"].values()
        return [root / name for name in sorted(set(files))]
    if (root / SINGLE).is_file():
        return [root / SINGLE]
    raise StreamRefused(f"{root} has neither {INDEX} nor {SINGLE}")


def read_checkpoint(root: Path) -> dict[str, ShardTensor]:
    out: dict[str, ShardTensor] = {}
    for shard in read_shards(root):
        for key, entry in read_header(shard).items():
            if key in out:
                raise StreamRefused(f"{key} is stored in both {out[key].path.name} and {shard.name}")
            out[key] = entry
    return out


def check_loaded(info: dict[str, Any]) -> None:
    problems = {name: sorted(str(k) for k in info.get(name) or ())
                for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    problems["conversion_errors"] = sorted(str(k) for k in (info.get("conversion_errors") or {}))
    found = {name: keys for name, keys in problems.items() if keys}
    if found:
        raise StreamRefused("; ".join(f"{name}: {', '.join(keys[:5])}" for name, keys in found.items()))


def load_generation_config(model: Any, root: Path) -> None:
    from transformers import GenerationConfig

    if not model.can_generate():
        return
    try:
        model.generation_config = GenerationConfig.from_pretrained(root)
    except OSError:
        model.generation_config = GenerationConfig.from_pretrained(
            root, config_file_name="config.json", _from_model_config=True)


def resolve_class(cls: Any, config: Any) -> Any:
    mapping = getattr(cls, "_model_mapping", None)
    if mapping is None:
        return cls
    from transformers.models.auto.auto_factory import _get_model_class

    if type(config) not in mapping:
        raise StreamRefused(f"{cls.__name__} has no class for {type(config).__name__}")
    model_class = _get_model_class(config, mapping)
    if getattr(model_class, "config_class", None) is not type(config):
        raise StreamRefused(f"{cls.__name__} loads {type(config).__name__} through another config")
    return model_class


# external: transformers PreTrainedModel.from_pretrained — given state_dict= and no path it builds the model on the meta device and runs its own key renaming, dtype plan, buffer initialisation and weight tying over the dict; its loader takes each value with value[...] and moves it to the device, as it does a safetensors slice
def load_streaming(cls: Any, model_id: str, *, config: Any, dtype: Any, device: Any) -> Any:
    import gc

    import torch

    root = Path(model_id)
    state = read_checkpoint(root)
    model, info = resolve_class(cls, config).from_pretrained(
        None, config=config, state_dict=state, dtype=dtype, device_map=device, output_loading_info=True)
    try:
        check_loaded(info)
    except StreamRefused:
        del model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
        raise
    model.config.name_or_path = model_id
    load_generation_config(model, root)
    return model
