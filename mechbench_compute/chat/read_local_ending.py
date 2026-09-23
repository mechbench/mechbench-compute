from __future__ import annotations

from collections.abc import Sequence


def read_local_ending(tok, out_ids: Sequence[int], *, stop_strings: Sequence[str],
                      max_tokens: int) -> str:
    """How a local sample ended, from the ids the sampler produced:
    at a `stop` string, at `max_tokens`, or at the model's own turn end."""
    if stop_strings and any(s in tok.decode(list(out_ids)) for s in stop_strings):
        return "stop"
    if len(out_ids) >= max_tokens:
        return "max_tokens"
    return "end"
