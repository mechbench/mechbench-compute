from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ACCELERATORS: tuple[str, ...] = ("metal", "cuda", "rocm", "tpu", "cpu")


@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    label: str
    platform_label: str
    accelerators: tuple[str, ...]
    requires: tuple[str, ...] = ()
    extra: str | None = None

    @property
    def modules(self) -> tuple[str, ...]:
        return (self.module, *self.requires)


BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="mlx",
        module="mlx.core",
        label="MLX (Apple Silicon, unified memory)",
        platform_label="macOS on Apple Silicon",
        accelerators=("metal",),
    ),
)


class BackendRefused(ValueError):
    pass


def describe_platform() -> str:
    v = sys.version_info
    return (
        f"{sys.platform}/{platform.machine()} "
        f"python {v.major}.{v.minor}.{v.micro}"
    )


def detect_accelerator() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "metal"
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    if shutil.which("rocm-smi") is not None:
        return "rocm"
    return "cpu"


def is_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def is_installed(backend: Backend) -> bool:
    return all(is_importable(m) for m in backend.modules)


def read_absence(backend: Backend, accelerator: str | None = None) -> str | None:
    missing = [m for m in backend.modules if not is_importable(m)]
    if missing:
        how = (f": pip install 'mechbench-compute[{backend.extra}]'"
               if backend.extra else "")
        return f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not installed{how}"
    on = accelerator or detect_accelerator()
    if on not in backend.accelerators:
        return (f"it runs on {' or '.join(backend.accelerators)}, and this machine's "
                f"accelerator is {on}")
    return None


def available(accelerator: str | None = None,
              declared: Sequence[Backend] = BACKENDS) -> list[Backend]:
    on = accelerator or detect_accelerator()
    return [b for b in declared if read_absence(b, on) is None]


def is_available(name: str, accelerator: str | None = None,
                 declared: Sequence[Backend] = BACKENDS) -> bool:
    return any(b.name == name for b in available(accelerator, declared))


def describe(accelerator: str | None = None,
             declared: Sequence[Backend] = BACKENDS) -> list[dict[str, Any]]:
    on = accelerator or detect_accelerator()
    out: list[dict[str, Any]] = []
    for b in declared:
        absent = read_absence(b, on)
        out.append({"name": b.name, "label": b.label,
                    "accelerators": list(b.accelerators), "present": absent is None,
                    **({"absent": absent} if absent is not None else {})})
    return out


def advertise(accelerator: str | None = None,
              declared: Sequence[Backend] = BACKENDS) -> dict[str, Any]:
    on = accelerator or detect_accelerator()
    return {"accelerator": on, "backends": [b.name for b in available(on, declared)]}


def select(capabilities: Mapping[str, Any], *, backend: str | None = None,
           accelerator: str | None = None,
           declared: Sequence[Backend] = BACKENDS) -> Backend:
    has = [str(b) for b in capabilities.get("backends") or []]
    on = capabilities.get("accelerator")
    if accelerator is not None and accelerator != on:
        raise BackendRefused(
            f"it needs a {accelerator} accelerator, and this runner has "
            f"{on or 'none it names'}")
    by_name = {b.name: b for b in declared}
    if backend is not None:
        if backend not in by_name:
            raise BackendRefused(
                f"{backend!r} is not a backend compute declares; it declares "
                f"{', '.join(by_name)}")
        if backend not in has:
            raise BackendRefused(
                f"it needs the {backend} backend, and this runner has "
                f"{', '.join(has) or 'none'}")
        return by_name[backend]
    for b in declared:
        if b.name in has and (on is None or on in b.accelerators):
            return b
    raise BackendRefused(
        f"this runner has no backend compute declares: it advertises "
        f"{', '.join(has) or 'none'} on {on or 'an accelerator it does not name'}; "
        f"compute declares {', '.join(by_name)}")


def active() -> Backend | None:
    found = available()
    return found[0] if found else None


def require() -> Backend:
    backend = active()
    if backend is not None:
        return backend

    absences = "\n".join(f"  {d['name']}: {d['absent']}" for d in describe())
    supported = ", ".join(b.platform_label for b in BACKENDS)
    raise ImportError(
        f"mechbench-compute has no compute backend on this machine "
        f"({describe_platform()}, accelerator {detect_accelerator()}).\n"
        f"\n"
        f"{absences}\n"
        f"\n"
        f"Supported today: {supported}. The package itself installs "
        f"anywhere — its platform-independent half is useful for reading "
        f"results — but running a model needs a backend.\n"
        f"\n"
        f"On Apple Silicon this usually means the dependency did not "
        f"install: try `pip install --force-reinstall mechbench-compute`.\n"
        f"Run `mechbench-runner doctor` for a fuller check."
    )
