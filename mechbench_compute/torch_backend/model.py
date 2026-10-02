from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from mechbench_compute import support
from mechbench_compute.errors import InvalidHookName
from mechbench_compute.hooks import HookFn, parse_hook_name
from mechbench_compute.interventions import Intervention, compose
from mechbench_compute.run_result import RunResult
from mechbench_compute.spans import add_to_span
from mechbench_compute.torch_backend.architectures import (
    ARCHITECTURES,
    for_model,
    for_type,
)
from mechbench_compute.torch_backend.forward import read_attention
from mechbench_compute.torch_backend.loading import (
    read_accelerator,
    read_device,
    read_stack,
)


class TorchModel:
    def __init__(self, model: Any, processor: Any, arch: Any = None,
                 architecture: support.Architecture | None = None):
        self._model = model
        self._processor = processor
        self.architecture = architecture if architecture is not None else for_model(model)
        self.arch = arch if arch is not None else self.architecture.arch_of(model)
        self.repo_id: str | None = None
        self.revision: str | None = None
        self.requested_ref: str | None = None
        self.node_adapter: dict | None = None
        self.fused_reference: Any = None
        self.attention: set[str] = set()

    @classmethod
    def load(cls, model_id: str, *, device: str | None = None, dtype: Any = None,
             on_download: Callable[[str, str | None], None] | None = None,
             on_download_bytes: Callable[[int, int], None] | None = None) -> TorchModel:
        import json
        from pathlib import Path

        from mechbench_compute.hub import ensure_model

        requested = model_id
        started = time.perf_counter()
        if Path(model_id).is_dir():
            repo_id, revision, snapshot = "local-checkpoint", Path(model_id).name, Path(model_id)
        else:
            repo_id, revision, snapshot = ensure_model(
                model_id, on_download=on_download, on_bytes=on_download_bytes)
        loading = time.perf_counter()
        add_to_span(download_seconds=loading - started)
        config = json.loads((Path(snapshot) / "config.json").read_text())
        model_type = str(config.get("model_type") or "").lower().removesuffix("_text")
        refused = support.refusal({**config, "model_type": model_type}, ARCHITECTURES)
        if refused is not None:
            raise NotImplementedError(f"the torch backend cannot load {requested!r}: {refused}.")
        architecture = for_type(model_type)
        m, p = architecture.load(str(snapshot), device=device, dtype=dtype)
        model = cls(m, p, arch=architecture.arch_of(m, str(snapshot)), architecture=architecture)
        model.repo_id = repo_id
        model.revision = revision
        model.requested_ref = requested
        add_to_span(model_load_seconds=time.perf_counter() - loading)
        return model

    @property
    def tokenizer(self) -> Any:
        return getattr(self._processor, "tokenizer", self._processor)

    @property
    def lm(self) -> Any:
        return self.architecture.lm(self._model)

    @property
    def device(self) -> Any:
        return read_device(self._model)

    @property
    def accelerator(self) -> str:
        return read_accelerator(self.device)

    def describe_hardware(self) -> dict[str, Any]:
        return read_stack(self.device)

    def prompt_cache(self) -> Any:
        return self.architecture.prompt_cache(self._model)

    def head_logits(self, hidden: Any) -> Any:
        return self.architecture.head_logits(self._model, hidden)

    def project_to_logits(self, residual: Any) -> Any:
        return self.architecture.project_to_logits(self._model, residual)

    def tokenize(self, prompt: str, *, chat_template: bool = True) -> Any:
        return self.architecture.tokenize(self._model, self._processor, prompt,
                                          chat_template=chat_template)

    def make_ids(self, ids: Iterable[int]) -> Any:
        import torch

        return torch.tensor([[int(t) for t in ids]], dtype=torch.long, device=self.device)

    def run(self, input_ids: Any, *, hooks: dict[str, HookFn] | None = None,
            capture: list[str] | None = None, interventions: list[Intervention] | None = None,
            kv_cache: Any = None) -> RunResult:
        final_hooks, final_capture = compose(interventions, hooks=hooks, capture=capture)
        names = set(final_hooks) | set(final_capture)
        self.check_hook_names(names)
        logits, cache = self.architecture.forward(
            self._model, input_ids, hooks=final_hooks, capture=final_capture,
            arch=self.arch, kv_cache=kv_cache)
        self.attention.add(read_attention(self._model, names, self.arch))
        add_to_span(forwards=1, tokens_in=int(input_ids.numel()), bytes_captured=sum(
            int(cache[n].numel() * cache[n].element_size()) for n in set(final_capture) if n in cache))
        return RunResult(logits=logits, cache=cache)

    def check_hook_names(self, names: Iterable[str]) -> None:
        declared = self.architecture
        absent = declared.absent_points(self.arch)
        valid = [n for n in self.arch.all_hook_names()
                 if declared.supports(n.split(".", 2)[-1] if n.startswith("blocks.") else n,
                                      layer_scoped=n.startswith("blocks."))]
        for n in names:
            info = parse_hook_name(n, arch=self.arch)
            if not declared.supports(info.point, layer_scoped=info.layer is not None):
                raise InvalidHookName(
                    f"{n} (not implemented by the {self.arch.model_type!r} forward on the "
                    f"{declared.backend} backend)", valid)
            if info.point in absent:
                raise InvalidHookName(f"{n} (absent: {absent[info.point]})", valid)

    def prefill_decision(self, prompt_ids: list[int], *, interventions: Any = None) -> tuple[Any, Any]:
        ids = self.make_ids(prompt_ids)
        result = self.run(ids, interventions=list(interventions or []))
        return None, result.logits[0, -1, :].float()

    def decoded_distribution(self, vector: Any) -> np.ndarray:
        import torch

        from mechbench_compute.arrays import read_softmax

        weight = self.lm.lm_head.weight
        v = torch.as_tensor(np.asarray(vector) if isinstance(vector, np.ndarray) else vector,
                            device=weight.device).to(weight.dtype)
        while v.ndim < 3:
            v = v.unsqueeze(0)
        last = self.project_to_logits(v)[..., -1, :]
        while last.ndim > 1:
            last = last[0]
        return read_softmax(last)

    def unadapted(self) -> Any:
        from mechbench_compute.torch_backend.lora import unfused

        if self.node_adapter is None:
            raise ValueError("no adapter is fused on this model from a node's `adapter` port")
        return unfused(self.lm, self.node_adapter)
