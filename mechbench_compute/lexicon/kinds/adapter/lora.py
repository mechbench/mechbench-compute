from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "adapter/lora",
    "A trained LoRA adapter: its weights, the base and stack it was trained on, its shape, and the training's methods section.",
    fields={"format": F("string", "`safetensors`."), "base_model": F("string", "The base it was trained on."),
            "trained_on": F("object", "`{base, adapters}` — the full stack."),
            "lora": F("object", "`{rank, alpha, scale, target_modules, params}`."),
            "train": F("object", "Steps, lr, seed, batch, final loss, the target spec, and the counts."),
            "data": F("string", "The safetensors bytes.", contentEncoding="binary")},
    required=("format", "lora", "data"),
    doc="What `adapter/train` produces, and what a model-running node takes on its `adapter` port. `train` records "
        "the whole training — steps, learning rate, seed, batch, final loss, the target it was trained toward "
        "and the counts — so the adapter's own object is the methods section of the experiment that made it, "
        "and `trained_on` names the base and the adapters it was stacked on, which is what a later fusion must "
        "match.",
)
