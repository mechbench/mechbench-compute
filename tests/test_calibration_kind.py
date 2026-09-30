from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mechbench_compute import platform_kinds
from mechbench_compute.calibration import write_shape_key
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon.kinds.platform.calibration import KIND, PRIMITIVES
from mechbench_compute.registry import SEALED

jsonschema = pytest.importorskip("jsonschema")

MARKS = Path(__file__).parent.parent / "mechbench_compute" / "ops" / "records" / "marks.generated.json"


def write_record(primitive, shape, seconds, **extra):
    key = write_shape_key(shape)
    return {"id": f"{primitive}:{key}", "chip": "Apple M4 Max", "stack": "sha256:ab12",
            "model": "mlx-community/gemma-4-e2b-it-bf16@0f3c", "dtype": "bfloat16",
            "primitive": primitive, "shape": shape, "shape_key": key, "seconds": seconds,
            "peak_memory_bytes": 10_400_000_000, "repeats": 5, "spread": 0.002,
            "warmup_seconds": seconds * 3, **extra}


HAND_MADE = K.collection("platform/calibration", [
    write_record("forward", {"n": 128, "b": 1}, 0.041),
    write_record("forward", {"n": 512, "b": 4}, 0.52),
    write_record("decode", {"t": 256}, 0.012),
    write_record("load", {"variant": "cold"}, 9.8, bytes_per_second=1.1e9),
], machine={"chip": "Apple M4 Max", "memory_gb": 128, "gpu_cores": 40, "os": "26.1",
            "python": "3.11.9"},
   stack_components={"compute": "0.171.0", "mlx": "0.32.1", "mlx_lm": "0.28.0",
                     "mlx_vlm": "0.6.15", "torch": None, "os": "26.1", "python": "3.11.9",
                     "checkpoint": "0f3c", "dtype": "bfloat16"},
   taken_at="2026-09-30T12:00:00Z", quiet=True)


def fill_speak(template, item, header):
    def one(m):
        path = m.group(1)
        return str(header[path[len("header."):]] if path.startswith("header.") else item[path])
    return re.sub(r"\{([^}]+)\}", one, template)


def test_the_kind_is_a_sealed_platform_record_keyed_by_its_measurement():
    assert KIND.platform and KIND.family == "platform" and "platform" in SEALED
    assert KIND.key == ("chip", "stack", "model", "dtype", "primitive", "shape_key")
    assert set(KIND.key) <= set(KIND.fields)
    assert K.satisfies("platform/calibration", "records/record")
    assert set(KIND.fields["primitive"]["enum"]) == set(PRIMITIVES) >= {
        "load", "forward", "decode", "backward", "capture", "intervene", "lora_step", "numeric", "io"}


def test_a_hand_made_collection_validates_and_keeps_its_header():
    assert HAND_MADE["key"] == list(KIND.key)
    schema = platform_kinds.item_schema(KIND)
    for item in HAND_MADE["items"]:
        jsonschema.validate(item, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**HAND_MADE["items"][0], "primitive": "sleep"}, schema)
    assert K.canonical_collection(HAND_MADE)["machine"]["chip"] == "Apple M4 Max"
    json.dumps(HAND_MADE)


def test_a_hand_made_collection_speaks():
    said = [fill_speak(KIND.speak, item, HAND_MADE) for item in HAND_MADE["items"]]
    assert said[0] == "forward n=128,b=1 on Apple M4 Max: 0.041 s (×5)"
    assert said[3] == "load variant=cold on Apple M4 Max: 9.8 s (×5)"


def test_the_draw_is_a_bar_over_declared_fields():
    bar = next(m for m in json.loads(MARKS.read_text())["marks"] if m["name"] == "bar")
    assert KIND.draw.mark == "bar"
    required = {c for c, need in bar["channels"].items() if need == "required"}
    assert required <= set(KIND.draw.encoding) <= set(bar["channels"])
    assert set(KIND.draw.encoding.values()) <= set(KIND.fields)


def test_a_shape_key_is_canonical_and_refuses_an_unknown_axis():
    assert write_shape_key({"b": 4, "n": 128}) == write_shape_key({"n": 128, "b": 4}) == "n=128,b=4"
    assert write_shape_key({}) == ""
    assert write_shape_key({"k": 4, "n": 512, "t": None}) == "n=512,k=4"
    with pytest.raises(ValueError, match="batch"):
        write_shape_key({"batch": 4})
