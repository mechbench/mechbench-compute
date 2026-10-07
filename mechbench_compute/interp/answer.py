from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mechbench_compute import shapes as S
from mechbench_compute.distill import encode, suffix_tokens


@dataclass(frozen=True)
class Answer:
    ids: tuple[int, ...]
    preferred: int

    def logp(self, lp: np.ndarray) -> float:
        picked = np.asarray(lp).reshape(-1)[list(self.ids)]
        return float(np.logaddexp.reduce(picked.astype(np.float64)) if picked.size > 1
                     else picked[0])

    def p(self, lp: np.ndarray) -> float:
        picked = np.asarray(lp).reshape(-1)[list(self.ids)]
        return float(np.exp(picked.astype(np.float64)).sum() if picked.size > 1
                     else np.exp(picked[0]))

    def rank(self, lp: np.ndarray) -> int:
        row = np.asarray(lp).reshape(-1)
        return min(int(np.sum(row > row[i])) for i in self.ids)

    def anchored(self, lp: np.ndarray | None) -> Answer:
        if lp is None:
            return self
        row = np.asarray(lp).reshape(-1)
        outside = [i for i in self.ids if not 0 <= i < row.size]
        if outside:
            raise ValueError(f"token {outside[0]} is outside this model's vocabulary of {row.size}")
        best = max(self.ids, key=lambda i: (float(row[i]), -self.ids.index(i)))
        return replace(self, preferred=int(best))

    def read(self, metric: str, lp: np.ndarray, logits: np.ndarray | None = None) -> float:
        if metric == "logit":
            return float(np.asarray(logits).reshape(-1)[self.preferred])
        return self.p(lp) if metric == "prob" else self.logp(lp)

    def read_differentiable(self, metric: str, row: Any) -> Any:
        if metric == "logit":
            return row[self.preferred]
        from mechbench_compute.intervene.array_ops import read_array_ops

        xp = read_array_ops(row)
        set_lp = xp.logsumexp((row - xp.logsumexp(row))[xp.array(list(self.ids))])
        return xp.exp(set_lp) if metric == "prob" else set_lp

    def variants(self, tokenizer, lp: np.ndarray) -> list[dict[str, Any]]:
        row = np.asarray(lp, dtype=np.float64).reshape(-1)
        return [S.read_token(tokenizer, i, float(row[i])) for i in self.ids]

    def entry(self, tokenizer, lp: np.ndarray) -> dict[str, Any]:
        here = self.anchored(lp)
        return {**S.read_token(tokenizer, here.preferred, self.logp(lp)),
                "rank": self.rank(lp),
                "variants": self.variants(tokenizer, lp)}


def spell_answer_variants(text: str) -> tuple[str, str]:
    bare = str(text).lstrip()
    if not bare:
        raise ValueError(
            f"a tracked answer is text to look for; {text!r} is empty or only "
            "whitespace")
    return " " + bare, bare


def encode_answer(tokenizer, text: str) -> Answer:
    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids: list[int] = []
    for spelling in spell_answer_variants(text):
        first = next((int(t) for t in encode(tokenizer, spelling) if t not in specials), None)
        if first is None:
            raise ValueError(f"tracked answer {spelling!r} tokenized to specials only")
        if _carries_text(tokenizer, first):
            ids.append(first)
    if not ids:
        raise ValueError(f"tracked answer {text!r} begins with no token that carries it")
    return make_answer(ids)


def _carries_text(tokenizer, token_id: int) -> bool:
    return bool(tokenizer.decode([token_id]).strip())


def encode_answer_in_context(tokenizer, text: str, prefix: str,
                             prefix_ids: list[int]) -> Answer:
    ids: list[int] = []
    refused: ValueError | None = None
    for spelling in spell_answer_variants(text):
        try:
            first = int(suffix_tokens(tokenizer, prefix, prefix_ids, spelling)[0])
            if _carries_text(tokenizer, first):
                ids.append(first)
        except ValueError as err:
            refused = refused or err
    if not ids:
        raise refused or ValueError(f"tracked answer {text!r} tokenized to nothing")
    return make_answer(ids)


def make_answer(ids: Sequence[int]) -> Answer:
    unique = tuple(dict.fromkeys(int(i) for i in ids))
    return Answer(unique, unique[0])


def read_answer(tokenizer, value: Any, *, by_id: bool = False) -> Answer:
    if not (by_id and isinstance(value, Mapping)):
        return encode_answer(tokenizer, str(value))
    tid = value.get("id")
    if isinstance(tid, bool) or not isinstance(tid, int) or tid < 0 or set(value) - {"id", "text"}:
        raise ValueError(f"a token by id is {{\"id\": <int>}}, with its \"text\" beside it if "
                         f"you like; not {dict(value)!r}")
    if "text" in value and tokenizer.decode([tid]) != value["text"]:
        raise ValueError(f"token {tid} is {tokenizer.decode([tid])!r} in this model's "
                         f"vocabulary, not {value['text']!r}")
    return make_answer([tid])
