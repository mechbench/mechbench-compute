"""`mlx.core`, for a module that has to import where mlx does not exist.

An operation's file holds its declaration as well as its mechanism, and
the declaration is read on machines that will never run the mechanism:
the documentation build, the API, a runner without Apple silicon. So the
import cannot fail there. What fails instead is the first use, and it
says what is missing.
"""
from __future__ import annotations

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised by tests/test_ops_layout.py

    class _Missing:
        def __getattr__(self, name: str):
            raise ImportError(
                f"mlx is not installed, so `mx.{name}` is unavailable: this "
                "operation needs local weights on Apple silicon "
                "(its declaration says `requires: mlx-local`).")

    mx = _Missing()  # type: ignore[assignment]

__all__ = ["mx"]
