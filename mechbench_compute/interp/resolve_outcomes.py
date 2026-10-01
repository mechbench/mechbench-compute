from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.interp.answer import Answer, encode_answer, make_answer


def resolve_outcomes(model, record: Mapping[str, Any]) -> list[Answer]:
    rid = record.get("id")
    texts = list(dict.fromkeys(str(o) for o in record.get("outcomes") or []))
    if not texts:
        raise ValueError(f"record {rid!r} has no `outcomes`; the outcome metrics read the distribution over them")
    answers = []
    for text in texts:
        try:
            matched = encode_answer(model.tokenizer, text)
        except ValueError as err:
            raise ValueError(f"record {rid!r}: outcome {text!r}: {err}") from err
        whole = [i for i in matched.ids if model.tokenizer.decode([i]).strip() == text.strip()]
        if not whole:
            raise ValueError(
                f"record {rid!r}: outcome {text!r} is not one token under this tokenizer; it begins with "
                f"{model.tokenizer.decode([matched.preferred])!r}")
        answers.append(make_answer(whole))
    return answers
