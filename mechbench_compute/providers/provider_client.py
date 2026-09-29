from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProviderClient:
    def __init__(self, ref: Any, *, secrets: Mapping[str, Any] | None = None,
                 limiter: Any = None, job_budget: Any = None) -> None:
        from mechbench_compute import model_ref as model_ref_mod

        self.ref = model_ref_mod.parse(ref)
        if not self.ref.is_endpoint:
            raise ValueError(
                f"{self.ref.describe()} is local weights, not a provider's "
                "endpoint: ask for the model, not a provider")
        self.secrets = secrets
        self.limiter = limiter
        self.job_budget = job_budget

    def chat(self, records: Sequence[Any], params: Mapping[str, Any], *,
             cassette: Any = None, cassette_mode: str | None = None,
             on_item=None, on_start=None, resume_items=None) -> dict[str, Any]:
        from mechbench_compute import chat as chat_mod

        return chat_mod.run_remote(
            self.ref, records, params, secrets=self.secrets,
            cassette=cassette, cassette_mode=cassette_mode,
            limiter=self.limiter, job_budget=self.job_budget,
            on_item=on_item, on_start=on_start, resume_items=resume_items)

    def embed(self, texts: Sequence[str], params: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.ref.describe()}: no provider transport embeds text yet, "
            "so provider.embed has nothing to call")
