from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mechbench_compute.spans import add_to_span
from mechbench_compute.torch_backend.batching import bound_batch, chunk
from mechbench_compute.torch_backend.decoding import make_prompt_cache

GENERATE_BATCH = 8


@dataclass
class Draw:
    prompt_ids: list[int]
    rng: np.random.Generator
    on_token: Callable[[dict[str, Any]], None] | None = None
    out_ids: list[int] = field(default_factory=list)
    hit_stop: bool = False
    done: bool = False
    said: str = ""
    pieces: list[str] = field(default_factory=list)


def read_pad_id(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id", "unk_token_id"):
        found = getattr(tokenizer, name, None)
        if found is not None:
            return int(found)
    return 0


def left_pad(prompts: Sequence[Sequence[int]], pad_id: int, device: Any) -> tuple[Any, Any, Any]:
    import torch

    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, p in enumerate(prompts):
        if p:
            ids[row, width - len(p):] = torch.tensor(list(p), dtype=torch.long)
            mask[row, width - len(p):] = 1
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), positions.to(device)


def forward_rows(model: Any, ids: Any, mask: Any, positions: Any, cache: Any) -> Any:
    import torch

    add_to_span(forwards=1, tokens_in=int(mask[:, -ids.shape[1]:].sum()))
    with torch.no_grad():
        out = model._model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :].float()


def accept(draw: Draw, next_id: int, *, tokenizer: Any, stop: set[int], stops: tuple[str, ...],
           window: int, tracking: bool, index: int) -> None:
    from mechbench_compute.generate import read_stop_hit

    if next_id in stop:
        draw.done = True
        return
    draw.out_ids.append(int(next_id))
    if stops and read_stop_hit(tokenizer, draw.out_ids, stops, window):
        draw.hit_stop = draw.done = True
        return
    piece = None
    if tracking:
        now = tokenizer.decode(draw.out_ids)
        piece, draw.said = now[len(draw.said):], now
        draw.pieces.append(piece)
    if draw.on_token is not None:
        draw.on_token({"index": index, "id": int(next_id), "text": piece})


def write_batch(model: Any, draws: Sequence[Draw], *, max_tokens: int, temperature: float,
                top_p: float, stops: tuple[str, ...]) -> None:
    import torch

    from mechbench_compute.generate import read_stop_ids, sample_next

    tok = model.tokenizer
    stop = read_stop_ids(tok)
    window = (max(len(s) for s in stops) + 8) if stops else 0
    tracking = any(d.on_token is not None for d in draws)
    pad_id = read_pad_id(tok)
    ids, mask, positions = left_pad([d.prompt_ids for d in draws], pad_id, model.device)
    cache = make_prompt_cache(model._model)
    rows = forward_rows(model, ids, mask, positions, cache)
    last = positions[:, -1:]
    for index in range(int(max_tokens)):
        fed = []
        for row, draw in enumerate(draws):
            if not draw.done:
                accept(draw, sample_next(rows[row], temperature=temperature, top_p=top_p,
                                          rng=draw.rng),
                       tokenizer=tok, stop=stop, stops=stops, window=window,
                       tracking=tracking, index=index)
            fed.append(draw.out_ids[-1] if draw.out_ids and not draw.done else pad_id)
        if all(d.done for d in draws) or index == int(max_tokens) - 1:
            break
        last = last + 1
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        step = torch.tensor([[t] for t in fed], dtype=torch.long, device=model.device)
        rows = forward_rows(model, step, mask, last, cache)
    add_to_span(tokens_out=sum(len(d.out_ids) for d in draws))


def write_draws(model: Any, draws: Sequence[Draw], *, max_tokens: int, temperature: float,
                top_p: float, stop_strings: Sequence[str] = (),
                batch_size: int | None = None) -> int:
    stops = tuple(s for s in (stop_strings or ()) if s)
    longest = max((len(d.prompt_ids) for d in draws), default=0) + int(max_tokens)
    size = bound_batch(model, batch_size or GENERATE_BATCH, longest)
    order = sorted(range(len(draws)), key=lambda i: len(draws[i].prompt_ids))
    for part in chunk(order, size):
        write_batch(model, [draws[i] for i in part], max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, stops=stops)
    return size
