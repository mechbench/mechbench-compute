from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "records/condition",
    "A prompt for a model, chat-shaped: a user turn, an optional system prompt, and an optional prefill that begins the assistant's turn.",
    extends="records/record",
    fields={
        "user": F("string", "The user turn.", **{"x-mechbench-text": True}),
        "system": F("string", "The system prompt, when there is one."),
        "prefill": F("string", "Text the assistant's turn begins with, so a read happens at the first token after it."),
        "tracked": F("object", "Named tokens a read reports on, by name; the first is the target.", additionalProperties={"type": "string"}),
        "template": F("boolean", "`false` to tokenize the record raw rather than through the chat template."),
    },
    required=("id", "user"),
    key=("id",),
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="Every model-running op renders a condition the same way: `system` and `user` through the model's chat "
        "template as one user turn, the assistant's turn begun with `prefill`, so the decision point — where "
        "`logits/read` reads — is the first token after the prefill, and every capture, sweep and lens can "
        "read there too. `records/derive` writes conditions from a design with templates; a record carrying only `text` or "
        "`prompt` is tokenized raw instead, as is one that says `template: false`.",
)
