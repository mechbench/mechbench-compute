from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class Prompt:
    text: str
    target: Optional[str] = None
    subject: Optional[str] = None
    category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSet:
    prompts: tuple[Prompt, ...]
    name: Optional[str] = None

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self.prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx) -> Prompt:
        return self.prompts[idx]

    def by_category(self, category: str) -> "PromptSet":
        return PromptSet(
            prompts=tuple(p for p in self.prompts if p.category == category),
            name=f"{self.name}[{category}]" if self.name else category,
        )

    def categories(self) -> list[str]:
        return list(
            dict.fromkeys(
                p.category for p in self.prompts if p.category is not None
            )
        )

    def validate(
        self,
        model,
        *,
        min_confidence: float = 0.5,
        require_target_match: bool = True,
        verbose: bool = True,
    ) -> "ValidatedPromptSet":
        items = []
        skipped = []
        for prompt in self.prompts:
            input_ids = model.tokenize(prompt.text)
            result = model.run(input_ids)
            last = result.last_logits.astype(mx.float32)
            lp = last - mx.logsumexp(last)
            probs = mx.softmax(last)
            mx.eval(lp, probs)
            lp_np = np.array(lp)
            probs_np = np.array(probs)
            top1_id = int(np.argmax(probs_np))
            top1_prob = float(probs_np[top1_id])
            top1_tok = model.tokenizer.decode([top1_id])

            passes_conf = top1_prob >= min_confidence
            passes_target = True
            if require_target_match and prompt.target is not None:
                t = prompt.target.lower()
                tok = top1_tok.lower()
                passes_target = t in tok or tok.strip() in t
            passes = passes_conf and passes_target

            if verbose:
                status = "OK" if passes else "SKIP"
                short = prompt.text[:55]
                print(f"  [{status}] {short:55s}  top1={top1_tok!r:15s} p={top1_prob:.3f}")

            if passes:
                items.append(ValidatedPrompt(
                    prompt=prompt,
                    input_ids=input_ids,
                    target_id=top1_id,
                    target_token=top1_tok,
                    baseline_lp=float(lp_np[top1_id]),
                    confidence=top1_prob,
                ))
            else:
                skipped.append(prompt)

        if verbose:
            print(f"\n{len(items)} / {len(self.prompts)} prompts validated.")

        return ValidatedPromptSet(
            items=tuple(items), skipped=tuple(skipped), source_name=self.name,
        )


@dataclass(frozen=True)
class ValidatedPrompt:
    prompt: Prompt
    input_ids: mx.array
    target_id: int
    target_token: str
    baseline_lp: float
    confidence: float


@dataclass(frozen=True)
class ValidatedPromptSet:
    items: tuple[ValidatedPrompt, ...]
    skipped: tuple[Prompt, ...] = ()
    source_name: Optional[str] = None

    def __iter__(self) -> Iterator[ValidatedPrompt]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx) -> ValidatedPrompt:
        return self.items[idx]

    @property
    def categories(self) -> list[str]:
        return list(
            dict.fromkeys(
                vp.prompt.category for vp in self.items if vp.prompt.category is not None
            )
        )

    @property
    def labels(self) -> np.ndarray:
        return np.array([vp.prompt.category for vp in self.items])

    def by_category(self, category: str) -> "ValidatedPromptSet":
        kept = tuple(vp for vp in self.items if vp.prompt.category == category)
        return ValidatedPromptSet(
            items=kept, skipped=(),
            source_name=f"{self.source_name}[{category}]" if self.source_name else category,
        )
