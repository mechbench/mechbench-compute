from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from mechbench_compute.spans import add_to_span

ROWS = 256


@dataclass(frozen=True)
class TorchLoss:
    count: Callable[[Any], int]
    total: Callable[[Any, Any], torch.Tensor]
    tokens: Callable[[Any], int]


def forward_logits(model: Any, ids: list[int], keep: int = 0) -> torch.Tensor:
    fed = torch.tensor([ids], dtype=torch.long, device=model.device)
    out = model._model(input_ids=fed, use_cache=False, logits_to_keep=keep)
    add_to_span(forwards=1, tokens_in=len(ids))
    return out.logits[0]


def compute_soft_row(row: torch.Tensor, target: dict[int, float]) -> torch.Tensor:
    tid = torch.tensor(list(target.keys()), dtype=torch.long, device=row.device)
    tw = torch.tensor(list(target.values()), dtype=torch.float32, device=row.device)
    return torch.logsumexp(row, dim=-1) - torch.sum(tw * row[tid])


def count_soft_rows(ex: Any) -> int:
    return 1 if isinstance(ex.soft, dict) else len(ex.tokens)


def read_soft_fed(ex: Any) -> list[int]:
    if isinstance(ex.soft, dict):
        return list(ex.prompt_ids)
    return list(ex.prompt_ids) + list(ex.tokens[:-1]) if len(ex.tokens) > 1 else list(ex.prompt_ids)


def total_soft_ce(model: Any, ex: Any) -> torch.Tensor:
    ids, seq, soft = ex.prompt_ids, ex.tokens, ex.soft
    if isinstance(soft, dict):
        if seq:
            raise ValueError("single-soft Example must have empty tokens")
        return compute_soft_row(forward_logits(model, list(ids), 1)[-1].float(), soft)
    if not seq:
        raise ValueError("hard Example must have at least one token")
    if soft is not None and len(soft) != len(seq):
        raise ValueError("per-position soft list must match tokens")
    rows = forward_logits(model, read_soft_fed(ex), len(seq))
    total = torch.zeros((), dtype=torch.float32, device=rows.device)
    for j, t in enumerate(seq):
        row = rows[j].float()
        tgt = soft[j] if soft is not None else None
        total = total + (torch.logsumexp(row, dim=-1) - row[t] if tgt is None
                         else compute_soft_row(row, tgt))
    return total


def total_sft(model: Any, item: Any) -> torch.Tensor:
    logits = forward_logits(model, list(item.ids[:-1]))
    trained = item.trained[1:]
    targets = torch.tensor(item.ids[1:], dtype=torch.long, device=logits.device)
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for start in range(0, len(trained), ROWS):
        weights = trained[start:start + ROWS]
        if not any(weights):
            continue
        rows = logits[start:start + ROWS].float()
        nll = (torch.logsumexp(rows, dim=-1)
               - rows.gather(-1, targets[start:start + ROWS, None])[:, 0])
        total = total + torch.sum(nll * torch.tensor([float(w) for w in weights], device=rows.device))
    return total


LOSSES = {
    "soft_ce": TorchLoss(count=count_soft_rows, total=total_soft_ce,
                         tokens=lambda ex: len(read_soft_fed(ex))),
    "compute_sft_loss": TorchLoss(count=lambda item: sum(item.trained[1:]), total=total_sft,
                                  tokens=lambda item: len(item.ids) - 1),
}


def read_torch_loss(loss_fn: Callable) -> TorchLoss:
    found = LOSSES.get(getattr(loss_fn, "__name__", ""))
    if found is None:
        raise NotImplementedError(
            f"adapter/train: the loss {getattr(loss_fn, '__name__', loss_fn)!r} trains on the mlx "
            f"backend only; torch trains {', '.join(sorted(LOSSES))}")
    return found
