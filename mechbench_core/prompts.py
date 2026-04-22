"""Prompt and PromptSet dataclasses + the validate() machinery.

A Prompt is an immutable description of one prompt to send through the model
plus optional metadata used by analyses (target answer, subject-entity
substring, category label). A PromptSet is an ordered collection of Prompts.

PromptSet.validate(model) runs each prompt through the model, captures the
top-1 prediction + confidence, and returns a ValidatedPromptSet that has
input_ids and baseline log-probability attached to each item. Most
experiments start with this validate() call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class Prompt:
    """A single prompt.

    Attributes:
        text: The prompt string. Will be passed through Model.tokenize, which
            applies the chat template.
        target: Expected answer token (substring match against the model's
            top-1 decoded token). If None, validation only checks confidence.
            Case-insensitive substring match in either direction.
        subject: Substring identifying the 'subject entity' position in the
            tokenized prompt. Used by fact_vectors() to pick which token's
            residual to extract. Optional.
        category: Categorical label for grouped analyses (used heavily in
            step_10/11/12/13). Optional.
        metadata: Free-form dict for any extra context. Optional.
    """

    text: str
    target: Optional[str] = None
    subject: Optional[str] = None
    category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSet:
    """An ordered collection of Prompts.

    Pass to PromptSet.validate(model) to run them through the model and get
    a ValidatedPromptSet. Most analysis helpers (fact_vectors, etc.) accept
    a ValidatedPromptSet as input.
    """

    prompts: tuple[Prompt, ...]
    name: Optional[str] = None

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self.prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx) -> Prompt:
        return self.prompts[idx]

    def by_category(self, category: str) -> "PromptSet":
        """Return a new PromptSet containing only prompts in this category."""
        return PromptSet(
            prompts=tuple(p for p in self.prompts if p.category == category),
            name=f"{self.name}[{category}]" if self.name else category,
        )

    def categories(self) -> list[str]:
        """List of distinct categories present, in first-seen order."""
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
        """Run each prompt through the model; keep ones that pass validation.

        A prompt PASSES if:
          - top-1 probability at the final position is >= min_confidence, AND
          - if require_target_match is True AND prompt.target is set, the
            target is a case-insensitive substring of the top-1 decoded token
            (or vice versa, handling surface variants like 'Paris' vs ' Paris'
            vs 'paris').

        Pass require_target_match=False for geometric experiments where you
        want to keep every prompt regardless of whether the model answers
        correctly (e.g. step_13's stress tests, where the analysis is about
        WHERE the representation lives, not whether the model is right).

        verbose=True prints a one-line OK/SKIP summary per prompt as it goes.
        """
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
    """One prompt that passed validation, with model-side artifacts attached."""

    prompt: Prompt
    input_ids: mx.array
    target_id: int
    target_token: str
    baseline_lp: float
    confidence: float


@dataclass(frozen=True)
class ValidatedPromptSet:
    """A PromptSet that's been run through the model.

    Iterates over ValidatedPrompts (the kept items). The .skipped attribute
    holds Prompts that failed validation, for diagnostic purposes.
    """

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
        """Distinct category values present, in first-seen order. Excludes None.
        For a per-prompt array (potentially with Nones), use .labels instead."""
        return list(
            dict.fromkeys(
                vp.prompt.category for vp in self.items if vp.prompt.category is not None
            )
        )

    @property
    def labels(self) -> np.ndarray:
        """One-per-prompt array of each prompt's .category, in iteration order.

        May contain None entries if some prompts didn't specify a category.
        Composes directly with iterate_clusters and the geometry stats helpers.
        """
        return np.array([vp.prompt.category for vp in self.items])

    def by_category(self, category: str) -> "ValidatedPromptSet":
        kept = tuple(vp for vp in self.items if vp.prompt.category == category)
        return ValidatedPromptSet(
            items=kept, skipped=(),
            source_name=f"{self.source_name}[{category}]" if self.source_name else category,
        )
