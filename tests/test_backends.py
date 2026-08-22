"""Installing anywhere, computing where a backend exists (Benji, 2026-08-22).

`pip install mechbench-compute` resolves its own substrate — MLX on Apple
Silicon, nothing yet elsewhere — so a machine without one has to explain
itself rather than fail inside a dependency the user never named.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from mechbench_compute import backends


def test_this_machine_reports_its_backend():
    active = backends.active()
    assert active is not None, "the test machine should have MLX"
    assert active.name == "mlx"
    assert "darwin" in backends.describe_platform()


def test_a_machine_with_no_backend_gets_an_explanation(monkeypatch):
    # Simulate the platform we cannot run here: nothing importable.
    real_find_spec = importlib.util.find_spec

    def blind(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "torch"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(backends.importlib.util, "find_spec", blind)

    assert backends.available() == []
    assert backends.active() is None
    with pytest.raises(ImportError) as exc:
        backends.require()

    message = str(exc.value)
    # It has to say what is wrong, where, and what to do — a bare
    # "No module named 'mlx'" is what this exists to replace.
    assert "no compute backend" in message
    assert backends.describe_platform() in message
    assert "macOS on Apple Silicon" in message
    assert "doctor" in message


def test_backend_detection_does_not_import_the_substrate(monkeypatch):
    # available() runs at package import; loading MLX there would cost
    # seconds on every `mechbench-runner status`.
    loaded = []
    real_import = __import__

    def watched(name, *args, **kwargs):
        if name.split(".")[0] == "mlx":
            loaded.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", watched)
    for m in [k for k in sys.modules if k.startswith("mlx")]:
        pass  # already-imported modules are fine; we watch for NEW imports
    before = len(loaded)
    backends.available()
    assert len(loaded) == before
