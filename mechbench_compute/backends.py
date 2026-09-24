from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    label: str
    platform_label: str


BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="mlx",
        module="mlx.core",
        label="MLX (Apple Silicon, unified memory)",
        platform_label="macOS on Apple Silicon",
    ),
)


def describe_platform() -> str:
    v = sys.version_info
    return (
        f"{sys.platform}/{platform.machine()} "
        f"python {v.major}.{v.minor}.{v.micro}"
    )


def available() -> list[Backend]:
    return [b for b in BACKENDS if importlib.util.find_spec(b.module) is not None]


def active() -> Backend | None:
    found = available()
    return found[0] if found else None


def require() -> Backend:
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
