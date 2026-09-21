from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mechbench_compute.intervene.spec import Spec


class SpecIntervention:
    """An `Intervention` (as_hooks / as_captures) over a whole spec list
    for one record's tokens.

    `tokens` is the sequence the positions resolve against. A decoder
    that runs the sequence in chunks — the prompt, then one token per
    step — calls `on_token` with each token it produces, so a selector
    like `{"tokens": ["lighthouse"]}` or `"generated"` sees the words
    as they arrive (000601)."""

    def __init__(self, specs: Sequence[Spec], tokens: Sequence[str],
                 record: Mapping[str, Any] | None = None,
                 prompt_len: int | None = None, growing: bool = False) -> None:
        self.tokens: list[str] = list(tokens)
        self.prompt_len = len(self.tokens) if prompt_len is None else int(prompt_len)
        self._hooks: dict[str, Callable] = {}
        for spec in specs:
            for layer, name in zip(spec.layers, spec.hook_names(), strict=True):
                fn = spec.build(layer, self.tokens, record, prompt_len=self.prompt_len,
                                growing=growing)
                prev = self._hooks.get(name)
                if prev is None:
                    self._hooks[name] = fn
                else:
                    def chained(act, info, _a=prev, _b=fn):
                        out = _a(act, info)
                        return _b(out if out is not None else act, info)
                    self._hooks[name] = chained

    def as_hooks(self) -> dict[str, Callable]:
        return dict(self._hooks)

    def as_captures(self) -> list[str]:
        return []

    def on_token(self, token: str) -> None:
        """The decoder produced one more token: the sequence grew."""
        self.tokens.append(token)
