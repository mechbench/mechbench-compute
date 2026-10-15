from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

NUMPY_DTYPES: dict[str, str] = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2",
    "I8": "i1", "U8": "u1", "BOOL": "?",
}


@dataclass(frozen=True)
class RawTensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes


def read_raw(blob: bytes) -> dict[str, RawTensor]:
    view = memoryview(blob)
    if len(view) < 8:
        raise ValueError("not a safetensors file: shorter than its header length")
    (n,) = struct.unpack("<Q", view[:8])
    if 8 + n > len(view):
        raise ValueError("not a safetensors file: its header runs past the end")
    header = json.loads(bytes(view[8:8 + n]))
    body = view[8 + n:]
    out: dict[str, RawTensor] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if not 0 <= start <= end <= len(body):
            raise ValueError(f"safetensors tensor {name!r} lies outside the file")
        out[name] = RawTensor(str(entry["dtype"]), tuple(int(d) for d in entry["shape"]),
                              bytes(body[start:end]))
    return out


def write_raw(tensors: Mapping[str, RawTensor]) -> bytes:
    header: dict[str, object] = {"__metadata__": None}
    offset = 0
    for name in sorted(tensors):
        t = tensors[name]
        header[name] = {"data_offsets": [offset, offset + len(t.data)], "dtype": t.dtype,
                        "shape": list(t.shape)}
        offset += len(t.data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return b"".join([struct.pack("<Q", len(encoded)), encoded,
                     *(tensors[name].data for name in sorted(tensors))])


def read_f64(t: RawTensor) -> np.ndarray:
    if t.dtype == "BF16":
        wide = np.frombuffer(t.data, dtype="<u2").astype(np.uint32) << 16
        return wide.view(np.float32).astype(np.float64).reshape(t.shape)
    dtype = NUMPY_DTYPES.get(t.dtype)
    if dtype is None:
        raise ValueError(f"safetensors dtype {t.dtype} is not one compute reads as numbers")
    return np.frombuffer(t.data, dtype=dtype).astype(np.float64).reshape(t.shape)
