from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute.torch_backend.batching import bound_batch, chunk

SCORE_BATCH = 16


def read_target_logprobs(rows: Any, targets: Any) -> Any:
    import torch

    rows = rows.float()
    picked = torch.gather(rows, -1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(rows, dim=-1)


def score_items(model: Any, prompt_ids: list[int], sequences: Mapping[str, list[int]], *,
                batch_size: int | None = None) -> dict[str, float]:
    import torch

    n_prompt = len(prompt_ids)
    groups: dict[int, list[str]] = {}
    for item, seq in sequences.items():
        groups.setdefault(len(seq), []).append(item)
    out: dict[str, float] = {}
    for n, items in sorted(groups.items()):
        size = bound_batch(model, batch_size or SCORE_BATCH, n_prompt + n, logits_rows=n)
        for part in chunk(items, size):
            fed = torch.tensor([prompt_ids + list(sequences[it][:-1]) if n > 1 else list(prompt_ids)
                                for it in part], dtype=torch.long, device=model.device)
            targets = torch.tensor([list(sequences[it]) for it in part], dtype=torch.long,
                                   device=model.device)
            with torch.no_grad():
                hidden = model.trunk_hidden(fed)
                rows = model.head_logits(hidden[:, n_prompt - 1:n_prompt - 1 + n, :])
                lp = read_target_logprobs(rows, targets).double().sum(dim=1).cpu().numpy()
            for it, value in zip(part, lp, strict=True):
                out[it] = float(value)
    return out


def score_tokens(model: Any, sequences: Sequence[Sequence[int]], *,
                 batch_size: int | None = None) -> list[np.ndarray]:
    import torch

    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    out = [np.zeros(0, dtype=np.float32) for _ in sequences]
    longest = max((len(s) for s in sequences), default=1)
    size = bound_batch(model, batch_size or SCORE_BATCH, longest, logits_rows=longest)
    for part in chunk(order, size):
        width = max(len(sequences[i]) for i in part)
        ids = torch.zeros((len(part), width), dtype=torch.long, device=model.device)
        mask = torch.zeros((len(part), width), dtype=torch.long, device=model.device)
        for row, i in enumerate(part):
            seq = list(sequences[i])
            ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, :len(seq)] = 1
        with torch.no_grad():
            hidden = model.trunk_hidden(ids, attention_mask=mask)
            for row, i in enumerate(part):
                n = len(sequences[i])
                if n >= 2:
                    rows = model.head_logits(hidden[row:row + 1, :n - 1, :])[0]
                    out[i] = read_target_logprobs(rows, ids[row, 1:n]).cpu().numpy()
    return out
