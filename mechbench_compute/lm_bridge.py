from __future__ import annotations

from typing import Any

import mlx.core as mx
from lm_eval.api.model import LM


# external: mlx-lm — mlx_lm.evaluate cannot load VLM-shaped checkpoints (Gemma 4 keeps its weights under language_model.*)
class MechbenchLM(LM):
    def __init__(self, model: Any, on_request: Any = None) -> None:
        super().__init__()
        self.model = model
        self.on_request = on_request

    def _encode(self, text: str) -> list[int]:
        tok = self.model.tokenizer
        ids = list(tok.encode(text, add_special_tokens=False))
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None:
            ids = [int(bos)] + ids
        return [int(t) for t in ids]

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        out = []
        for i, req in enumerate(requests):
            ctx, cont = req.args
            ctx_ids = self._encode(ctx)
            full_ids = self._encode(ctx + cont)
            cont_ids = full_ids[len(ctx_ids):]
            if not cont_ids or full_ids[: len(ctx_ids)] != ctx_ids:
                ctx_ids = full_ids[:1]
                cont_ids = full_ids[1:]
            result = self.model.run(mx.array([full_ids], dtype=mx.int32))
            logits = result.logits[0].astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            start = len(ctx_ids) - 1
            total = 0.0
            greedy = True
            for j, tok in enumerate(cont_ids):
                lp = logprobs[start + j]
                total += float(lp[int(tok)])
                if int(mx.argmax(lp)) != int(tok):
                    greedy = False
            out.append((total, greedy))
            if self.on_request:
                self.on_request(i, len(requests))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "rolling-perplexity tasks are not bridged yet; use "
            "loglikelihood tasks (arc_*, hellaswag, mmlu, winogrande)")

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "generation tasks (gsm8k, ...) are not bridged yet; use "
            "loglikelihood tasks (arc_*, hellaswag, mmlu, winogrande)")
