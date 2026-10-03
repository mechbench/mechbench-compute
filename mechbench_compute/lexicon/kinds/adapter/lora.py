from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import COORDS, F

KIND = Kind(
    "adapter/lora",
    "A trained LoRA adapter: its weights, the base and stack it was trained on, its shape, and the training's methods section.",
    fields={"format": F("string", "`safetensors`."), "base_model": F("string", "The base it was trained on."),
            "trained_on": F("object", "`{base, adapters}` — the full stack."),
            "lora": F("object", "`{rank, alpha, scale, target_modules, params}`."),
            "train": F("object", "Steps, lr, seed, batch, final loss, the target spec (under `objective: \"sft\"`, "
                                 "the objective and its settings), and the counts."),
            "data": F("string", "The safetensors bytes.", contentEncoding="binary"),
            "id": F("string", "In a collection of checkpoints: `step-<n>`."),
            "coords": COORDS,
            "loss": F("number", "In a collection of checkpoints: the training loss at this checkpoint's step.")},
    required=("format", "lora", "data"),
    key=("id", "coords"),
    header={"base_model": "The base the adapters were trained on.",
            "trained_on": "`{base, adapters}` — the full stack.",
            "lora": "`{rank, alpha, scale, target_modules, params}` of every adapter in it.",
            "train": "The training's methods section, as the final adapter carries it.",
            "checkpoint_every": "The cadence the checkpoints were kept at, in steps."},
    doc="What `adapter/train` produces, and what a model-running node takes on its `adapter` port. `train` records "
        "the whole training — steps, learning rate, seed, batch, final loss, the target it was trained toward "
        "(or the `sft` objective and its settings) and the counts — so the adapter's own object is the methods "
        "section of the experiment that made it, and `trained_on` names the base and the adapters it was "
        "stacked on, which is what a later fusion must match. With `keep_checkpoints` the training also emits a collection of these, one per kept step: each "
        "item a whole adapter, `coords.step` its step and `loss` the training loss there, so any operation "
        "mapped over the collection is a sweep over training time.",
)
