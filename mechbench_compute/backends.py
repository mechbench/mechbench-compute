"""Which substrate this machine can actually compute on.

`mechbench-compute` installs on any platform and resolves its own substrate:
the dependency markers in pyproject bring MLX to Apple Silicon and nothing to
a machine that has no backend yet. This module is the other half of that —
it answers "can this machine run a model, and with what" before anything
tries, so an unsupported platform gets a sentence rather than a traceback out
of a package it never asked for.

Only the MLX backend exists today. A CUDA one means writing the hook-aware
forward passes again against Torch — the interventions reach *inside* the
forward, so there is no thin shim that would make it free — and this is where
it would register when it does.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """A substrate that can hold a model and run instrumented forwards."""

    name: str
    module: str
    label: str
    #: What platform this backend can serve, for reporting on machines where
    #: it is unavailable.
    platform_label: str


BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="mlx",
        module="mlx.core",
        label="MLX (Apple Silicon, unified memory)",
        platform_label="macOS on Apple Silicon",
    ),
    # Backend(name="torch", module="torch", label="PyTorch (CUDA)", ...)
    # arrives with the Torch forward passes, not before.
)


def describe_platform() -> str:
    """`darwin/arm64 python 3.11.1` — what a report or a question should say."""
    v = sys.version_info
    return (
        f"{sys.platform}/{platform.machine()} "
        f"python {v.major}.{v.minor}.{v.micro}"
    )


def available() -> list[Backend]:
    """Backends whose substrate is importable on this machine.

    Checked by spec rather than by importing: this runs at package import
    time, and loading MLX is not free.
    """
    return [b for b in BACKENDS if importlib.util.find_spec(b.module) is not None]


def active() -> Backend | None:
    """The backend that will be used, or None if this machine has none."""
    found = available()
    return found[0] if found else None


def require() -> Backend:
    """The active backend, or an explanation of why there isn't one."""
    backend = active()
    if backend is not None:
        return backend

    supported = ", ".join(b.platform_label for b in BACKENDS)
    raise ImportError(
        f"mechbench-compute has no compute backend on this machine "
        f"({describe_platform()}).\n"
        f"\n"
        f"Supported today: {supported}. The package itself installs "
        f"anywhere — its platform-independent half is useful for reading "
        f"results — but running a model needs a backend, and the only one "
        f"implemented is MLX.\n"
        f"\n"
        f"On Apple Silicon this usually means the dependency did not "
        f"install: try `pip install --force-reinstall mechbench-compute`.\n"
        f"Run `mechbench-runner doctor` for a fuller check."
    )
