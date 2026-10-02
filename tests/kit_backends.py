from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from mechbench_compute import backends
from mechbench_compute.backends import Backend
from mechbench_compute.support import Architecture

WINDOW = 3

KIT_MODULES: dict[str, str] = {
    "mlx": "tests.tiny_models",
    "fake": "tests.fake_backend",
    "torch": "tests.tiny_torch_models",
}


@dataclass(frozen=True)
class KitBackend:
    backend: Backend
    capabilities: Mapping[str, Any]
    architectures: Mapping[str, Architecture]
    models: Sequence[tuple[str, str]]
    build: Callable[..., Any]
    read_parameter_names: Callable[[Any], set[str]]
    fit_adapter: Callable[[Any, Any], int]
    read_config: Callable[[Any], Mapping[str, Any]]
    declared: Sequence[Backend] = field(default=backends.BACKENDS)

    @property
    def name(self) -> str:
        return self.backend.name

    def select(self) -> Backend:
        return backends.select(self.capabilities, backend=self.name, declared=self.declared)


def load_kit(name: str) -> KitBackend:
    try:
        module = importlib.import_module(KIT_MODULES[name])
    except ImportError as e:
        pytest.skip(f"the {name} backend's tiny models need what this machine lacks: {e}")
    return module.KIT


def list_kit_params() -> list[Any]:
    out: list[Any] = []
    for name, module in KIT_MODULES.items():
        try:
            kit = importlib.import_module(module).KIT
        except ImportError as e:
            out.append(pytest.param((name, None, None), id=name,
                                    marks=pytest.mark.skip(reason=f"{name}: {e}")))
            continue
        out.extend(pytest.param((name, m, t), id=f"{name}-{m}") for m, t in kit.models)
    return out
