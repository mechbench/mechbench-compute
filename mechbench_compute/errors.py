"""Custom exceptions raised by mechbench_compute.

Each error message is designed to tell the user exactly how to fix the problem
— closest valid name suggestions for typos, valid layer ranges for out-of-bounds
indices, captured-keys hints when looking up missing cache entries.
"""

from __future__ import annotations

import difflib
from typing import Iterable


class InterpError(Exception):
    """Base class for every error this package raises.

    Catch this if you want a single except clause that handles all
    framework-originating errors.
    """


class InvalidHookName(InterpError):
    """The given name is not a recognized hook point on Gemma 4 E4B."""

    def __init__(self, name: str, valid_names: Iterable[str]):
        valid = list(valid_names)
        suggestions = difflib.get_close_matches(name, valid, n=3, cutoff=0.6)
        msg = f"Unknown hook point: {name!r}"
        if suggestions:
            sugg = ", ".join(repr(s) for s in suggestions)
            msg += f"\n  Did you mean: {sugg}?"
        msg += (
            "\n  Use mechbench_compute.all_hook_names() to list every valid"
            " hook point."
        )
        super().__init__(msg)
        self.name = name
        self.suggestions = suggestions


class LayerIndexOutOfRange(InterpError):
    """A hook references a layer index that doesn't exist."""

    def __init__(self, layer_idx: int, n_layers: int):
        msg = (
            f"Layer index {layer_idx} is out of range. "
            f"Gemma 4 E4B has {n_layers} layers; valid indices are "
            f"0 through {n_layers - 1}."
        )
        super().__init__(msg)
        self.layer_idx = layer_idx
        self.n_layers = n_layers


class CacheKeyError(InterpError):
    """A user looked up a cache key that wasn't captured during the forward."""

    def __init__(self, key: str, captured_keys: Iterable[str]):
        captured = list(captured_keys)
        suggestions = difflib.get_close_matches(key, captured, n=3, cutoff=0.6)

        msg = f"{key!r} is not in the cache."
        if not captured:
            msg += (
                "\n  No keys were captured. Pass capture=[...] to Model.run()"
                " naming the activations you want to keep."
            )
        elif suggestions:
            sugg = ", ".join(repr(s) for s in suggestions)
            msg += f"\n  Did you mean: {sugg}?"
            msg += f"\n  ({len(captured)} keys total in the cache.)"
        else:
            sample = ", ".join(repr(k) for k in captured[:5])
            msg += f"\n  Captured keys (first 5): {sample}"
            if len(captured) > 5:
                msg += f"; ... and {len(captured) - 5} more"
            msg += "\n  Pass this key in capture=[...] when calling Model.run()."

        super().__init__(msg)
        self.key = key
        self.captured_keys = captured
