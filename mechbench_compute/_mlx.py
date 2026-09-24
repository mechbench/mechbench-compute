from __future__ import annotations

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover

    class _Missing:
        def __getattr__(self, name: str):
            raise ImportError(
                f"mlx is not installed, so `mx.{name}` is unavailable: this "
                "operation needs local weights on Apple silicon "
                "(its declaration says `requires: mlx-local`).")

    mx = _Missing()  # type: ignore[assignment]

__all__ = ["mx"]
