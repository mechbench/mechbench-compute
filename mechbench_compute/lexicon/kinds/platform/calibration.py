from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import F

PRIMITIVES = ("load", "forward", "decode", "backward", "capture", "intervene", "lora_step",
              "numeric", "io", "memcopy", "matmul")

KIND = Kind(
    "platform/calibration",
    "How long one primitive takes on one chip under one software stack, at one shape, with its peak memory.",
    doc="Written by the runner's calibration: one record per (chip, stack, model, dtype, primitive, shape). "
        "`seconds` is the median of `repeats` timed runs after a warm-up, which is reported apart as "
        "`warmup_seconds`; `spread` is the interquartile range of the timed runs, in seconds. `load` is "
        "the weights read from disk (`shape.variant` `cold` or `warm`, by the page cache); `forward` and "
        "`backward` one pass at `n` tokens and batch `b`; `decode` one step at a cache of `t` tokens; "
        "`capture` the added cost of hooking `k` layers at `n` positions; `intervene` the added cost of "
        "one spec item on a forward; `lora_step` one training step with the optimizer; `numeric` a "
        "model-free op's cost per item (`shape.variant` names the op); `io` serializing and storing "
        "bytes; `memcopy` and `matmul` the two micro-measurements that read no model, the chip's memory "
        "bandwidth and bf16 throughput. A primitive that reads no model has an empty `model`.",
    extends="records/record",
    fields={
        "chip": F("string", "The chip, as the machine names it (`Apple M4 Max`)."),
        "stack": F("string", "The stack fingerprint: a hash over the header's `stack_components`."),
        "model": F("string", "The checkpoint as `repo@revision`; empty for a primitive that reads no model."),
        "dtype": F("string", "The weights' dtype or quantization (`bfloat16`, `4bit`)."),
        "primitive": F("string", "What was timed.", enum=list(PRIMITIVES)),
        "shape": F("object", "The shape the primitive ran at: `n` tokens, `b` batch, `k` hooked layers, "
                   "`t` cached tokens, each only where it applies, and `variant` (`cold` or `warm` for "
                   "`load`, the op for `numeric`).",
                   properties={"n": {"type": "integer"}, "b": {"type": "integer"},
                               "k": {"type": "integer"}, "t": {"type": "integer"},
                               "variant": {"type": "string"}},
                   additionalProperties=False),
        "shape_key": F("string", "The shape as one canonical string, its keys in the order n, b, k, t, "
                       "variant: `n=128,b=4`; empty for an empty shape."),
        "seconds": F("number", "The median seconds of one run over the timed repeats."),
        "bytes_per_second": F("number", "Bytes moved per second, where the primitive moves bytes "
                              "(`load`, `io`, `memcopy`); absent elsewhere."),
        "peak_memory_bytes": F("integer", "The most memory MLX held during a timed run, weights included."),
        "repeats": F("integer", "How many timed runs the median is over, the warm-up excluded."),
        "spread": F("number", "The interquartile range of the timed runs, in seconds."),
        "warmup_seconds": F("number", "The seconds of the untimed first run."),
    },
    required=("id", "chip", "stack", "model", "dtype", "primitive", "shape", "shape_key", "seconds",
              "peak_memory_bytes", "repeats", "spread", "warmup_seconds"),
    key=("chip", "stack", "model", "dtype", "primitive", "shape_key"),
    platform=True,
    header={
        "machine": "The machine the timings were taken on: `{chip, memory_gb, gpu_cores, os, python}`.",
        "stack_components": "Every version the timings depend on: `{compute, mlx, mlx_lm, mlx_vlm, torch, "
                            "os, python, checkpoint, dtype}`, a component the machine lacks null.",
        "taken_at": "When the calibration ran, as an ISO 8601 UTC timestamp.",
        "quiet": "Whether the machine was quiet throughout: no thermal throttling, no busy process on the "
                 "deny-list, the canary near its idle baseline.",
    },
    speak="{primitive} {shape_key} on {chip}: {seconds} s (×{repeats})",
    draw=Draw(mark="bar", encoding={"x": "shape_key", "y": "seconds", "series": "primitive"}),
)
