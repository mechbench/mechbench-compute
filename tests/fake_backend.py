from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np

from mechbench_compute import attribution, backends
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.arrays import read_f32
from mechbench_compute.backends import Backend
from mechbench_compute.cache import ActivationCache
from mechbench_compute.run_result import RunResult
from mechbench_compute.support import Unembed
from tests import tiny_models
from tests.kit_backends import KitBackend

FAKE = Backend(
    name="fake",
    module="numpy",
    label="numpy arrays over MLX's forward, for the test suite",
    platform_label="the test suite",
    accelerators=("cpu",),
)

DECLARED = (*backends.BACKENDS, FAKE)


class FakeNorm:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.weight = inner.weight
        self.eps = inner.eps
        self.gain_offset = attribution._read_gain_offset(inner)

    def __call__(self, x) -> np.ndarray:
        return read_f32(self.inner(mx.array(np.asarray(x, dtype=np.float32))))


def make_fake(architecture):
    def read_unembed(model) -> Unembed:
        u = architecture.attribution_unembed(model)
        return Unembed(norm=FakeNorm(u.norm), project=u.project, softcap=u.softcap)

    return dataclasses.replace(architecture, backend=FAKE.name, attribution_unembed=read_unembed)


ARCHITECTURES = {t: make_fake(a) for t, a in BY_MODEL_TYPE.items()}


class FakeModel:
    def __init__(self, inner) -> None:
        self.inner = inner
        self._model = inner._model
        self.arch = inner.arch
        self.architecture = ARCHITECTURES.get(inner.architecture.model_type, inner.architecture)
        self.tokenizer = inner.tokenizer

    @property
    def lm(self):
        return self.inner.lm

    def make_ids(self, ids) -> np.ndarray:
        return np.asarray([[int(t) for t in ids]], dtype=np.int64)

    def run(self, input_ids, *, hooks=None, capture=None, interventions=None, kv_cache=None):
        result = self.inner.run(mx.array(np.asarray(input_ids), dtype=mx.int32), hooks=hooks,
                                capture=capture, interventions=interventions, kv_cache=kv_cache)
        cache = ActivationCache({k: read_f32(v) for k, v in result.cache.items()},
                                offset=result.cache.offset)
        return RunResult(logits=read_f32(result.logits), cache=cache)

    def head_logits(self, hidden) -> np.ndarray:
        return read_f32(self.inner.head_logits(mx.array(np.asarray(hidden, dtype=np.float32))))

    def tokenize(self, prompt: str, *, chat_template: bool = True) -> np.ndarray:
        return np.asarray(self.inner.tokenize(prompt, chat_template=chat_template))


def build_fake_model(name: str, architecture=None) -> FakeModel:
    inner = (BY_MODEL_TYPE.get(architecture.model_type, architecture)
             if architecture is not None else None)
    return FakeModel(tiny_models.build_tiny_model(name, inner))


def read_parameter_names(model: FakeModel) -> set[str]:
    return tiny_models.read_parameter_names(model.inner)


def fit_adapter(model: FakeModel, keys) -> int:
    return tiny_models.fit_adapter(model.inner, keys)


def read_config(model: FakeModel) -> dict:
    return tiny_models.read_config(model.inner)


KIT = KitBackend(
    backend=FAKE,
    capabilities={"accelerator": "cpu", "backends": [FAKE.name]},
    architectures=ARCHITECTURES,
    models=tiny_models.KIT_MODELS,
    build=build_fake_model,
    read_parameter_names=read_parameter_names,
    fit_adapter=fit_adapter,
    read_config=read_config,
    declared=DECLARED,
)
