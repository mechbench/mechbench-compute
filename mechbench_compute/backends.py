from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ACCELERATORS: tuple[str, ...] = ("metal", "cuda", "rocm", "tpu", "cpu")

DEFAULT_BACKEND = "mlx"


@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    label: str
    platform_label: str
    accelerators: tuple[str, ...]
    requires: tuple[str, ...] = ()
    extra: str | None = None
    model: str | None = None
    architectures: str | None = None
    lora: str | None = None
    ops: tuple[str, ...] | None = None
    devices: tuple[tuple[str, str], ...] = ()

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
        model="mechbench_compute.model:Model",
        architectures="mechbench_compute.architectures",
        lora="mechbench_compute.lora",
    ),
    Backend(
        name="torch",
        module="torch",
        label="PyTorch with nnsight (NVIDIA GPU, or the CPU)",
        platform_label="Linux with an NVIDIA GPU (CUDA), or any machine's CPU",
        accelerators=("cuda", "rocm", "cpu"),
        requires=("nnsight", "transformers", "accelerate"),
        extra="torch",
        model="mechbench_compute.torch_backend.model:TorchModel",
        architectures="mechbench_compute.torch_backend.architectures",
        lora="mechbench_compute.torch_backend.lora",
        ops=("activations/capture", "activations/capture-attention", "activations/capture-tokens",
             "activations/contrast", "activations/differentiate", "activations/examples",
             "adapter/train", "dictionary/encode", "direction/unembed", "eval/benchmark",
             "eval/judge", "intervene/ablate-circuit", "intervene/ablate-heads",
             "intervene/ablate-layers", "intervene/apply", "intervene/patch", "intervene/path",
             "intervene/steer", "logits/attribute", "logits/read", "logits/read-layers",
             "logits/scan", "text/chat", "text/generate", "text/resample", "text/score",
             "text/tokenize", "trajectory/capture", "weights/capture", "weights/decompose"),
        devices=(("cuda", "cuda"), ("rocm", "cuda"), ("cpu", "cpu")),
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


def detect_accelerators() -> list[str]:
    found: list[str] = []
    if sys.platform == "darwin" and platform.machine() == "arm64":
        found.append("metal")
    if shutil.which("nvidia-smi") is not None:
        found.append("cuda")
    if shutil.which("rocm-smi") is not None:
        found.append("rocm")
    found.append("cpu")
    return found


def detect_accelerator() -> str:
    return detect_accelerators()[0]


def join_or(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


def join_and(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


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
        named = join_and(missing)
        return f"{named} {'is' if len(missing) == 1 else 'are'} not installed{how}"
    if accelerator is not None:
        if accelerator not in backend.accelerators:
            return (f"it runs on {join_or(backend.accelerators)}, and this machine's "
                    f"accelerator is {accelerator}")
        return None
    here = detect_accelerators()
    if not any(a in backend.accelerators for a in here):
        return (f"it runs on {join_or(backend.accelerators)}, and this machine's "
                f"{'accelerator is' if len(here) == 1 else 'accelerators are'} {join_and(here)}")
    return None


def available(accelerator: str | None = None,
              declared: Sequence[Backend] = BACKENDS) -> list[Backend]:
    return [b for b in declared if read_absence(b, accelerator) is None]


def is_available(name: str, accelerator: str | None = None,
                 declared: Sequence[Backend] = BACKENDS) -> bool:
    return any(b.name == name for b in available(accelerator, declared))


def describe(accelerator: str | None = None,
             declared: Sequence[Backend] = BACKENDS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in declared:
        absent = read_absence(b, accelerator)
        out.append({"name": b.name, "label": b.label,
                    "accelerators": list(b.accelerators), "present": absent is None,
                    **({"absent": absent} if absent is not None else {})})
    return out


def advertise(accelerator: str | None = None,
              declared: Sequence[Backend] = BACKENDS) -> dict[str, Any]:
    here = [accelerator] if accelerator else detect_accelerators()
    pairs = {a: [b.name for b in available(a, declared) if b.model is not None] for a in here}
    pairs = {a: names for a, names in pairs.items() if names}
    return {"accelerator": here[0],
            "backends": [b.name for b in declared if any(b.name in n for n in pairs.values())],
            "accelerators": pairs}


def read_pairs(capabilities: Mapping[str, Any]) -> dict[str | None, list[str]]:
    pairs = capabilities.get("accelerators")
    if isinstance(pairs, Mapping):
        return {str(a): [str(b) for b in names or []] for a, names in pairs.items()}
    on = capabilities.get("accelerator")
    return {str(on) if on else None: [str(b) for b in capabilities.get("backends") or []]}


def find(name: str, declared: Sequence[Backend] = BACKENDS) -> Backend:
    for b in declared:
        if b.name == name:
            return b
    raise BackendRefused(
        f"{name!r} is not a backend compute declares; it declares "
        f"{', '.join(b.name for b in declared)}")


def check_op(backend: Backend, name: str, needs: frozenset[str] | set[str], tier: str = "core") -> None:
    if backend.ops is None or tier != "core" or not any(n.startswith("model.") for n in needs):
        return
    if name not in backend.ops:
        raise BackendRefused(
            f"{name} does not run on the {backend.name} backend yet; there it runs "
            f"{', '.join(backend.ops)}")


def backend_of(model: Any) -> str:
    return str(getattr(getattr(model, "architecture", None), "backend", DEFAULT_BACKEND))


def read_required(requirements: Mapping[str, Any] | None,
                  declared: Sequence[Backend] = BACKENDS) -> Backend:
    name = (requirements or {}).get("backend") or DEFAULT_BACKEND
    return find(str(name), declared)


def device_for(backend: Backend, accelerator: str | None) -> str | None:
    if not backend.devices or accelerator is None:
        return None
    device = dict(backend.devices).get(accelerator)
    if device is None:
        raise BackendRefused(
            f"the {backend.name} backend does not run on {accelerator}; it runs on "
            f"{join_or(backend.accelerators)}")
    return device


def import_attribute(path: str, backend: Backend) -> Any:
    module, _, attribute = path.partition(":")
    try:
        found = importlib.import_module(module)
    except ModuleNotFoundError as e:
        absent = read_absence(backend, backend.accelerators[0]) or str(e)
        raise BackendRefused(f"this job runs on the {backend.name} backend, and {absent}") from e
    return getattr(found, attribute) if attribute else found


def load_model_class(backend: Backend) -> Any:
    if backend.model is None:
        raise BackendRefused(f"the executor does not run jobs on the {backend.name} backend yet")
    return import_attribute(backend.model, backend)


def load_lora(backend: Backend) -> Any:
    if backend.lora is None:
        raise BackendRefused(f"the {backend.name} backend fuses no stored adapter yet")
    return import_attribute(backend.lora, backend)


def load_architectures(backend: Backend) -> tuple[Any, ...]:
    if backend.architectures is None:
        return ()
    return tuple(import_attribute(backend.architectures, backend).ARCHITECTURES)


def select(capabilities: Mapping[str, Any], *, backend: str | None = None,
           accelerator: str | None = None,
           declared: Sequence[Backend] = BACKENDS) -> Backend:
    pairs = read_pairs(capabilities)
    named = [a for a in pairs if a is not None] or (
        [str(capabilities["accelerator"])] if capabilities.get("accelerator") else [])
    has = list(dict.fromkeys(n for names in pairs.values() for n in names))
    if accelerator is not None and accelerator not in pairs:
        raise BackendRefused(
            f"it needs a {accelerator} accelerator, and this runner has "
            f"{join_and(named) if named else 'none it names'}")
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
        if accelerator is not None and backend not in pairs[accelerator]:
            where = ", ".join(f"{', '.join(names)} on {a}" for a, names in pairs.items() if names)
            raise BackendRefused(
                f"it needs the {backend} backend on {accelerator}, and this runner has {where}")
        return by_name[backend]
    for b in declared:
        for a, names in pairs.items():
            if (b.name in names and (accelerator is None or a == accelerator)
                    and (a is None or a in b.accelerators)):
                return b
    raise BackendRefused(
        f"this runner has no backend compute declares: it advertises "
        f"{', '.join(has) or 'none'} on {join_and(named) if named else 'an accelerator it does not name'}; "
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
        f"({describe_platform()}, accelerators {join_and(detect_accelerators())}).\n"
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
