from __future__ import annotations

from typing import Iterable, Sequence

import mlx.core as mx
import numpy as np

from .model import Model
from .prompts import Prompt, PromptSet

_GEMMA4_END_OF_TURN_ID = 106


def _sample_next(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0:
        return int(np.argmax(np.array(logits.astype(mx.float32))))
    scaled = logits.astype(mx.float32) / float(temperature)
    probs = mx.softmax(scaled)
    mx.eval(probs)
    p = np.array(probs).astype(np.float64)
    order = np.argsort(-p)
    sorted_p = p[order]
    cum = np.cumsum(sorted_p)
    cutoff = int(np.searchsorted(cum, top_p) + 1)
    cutoff = max(1, min(cutoff, len(sorted_p)))
    kept = sorted_p[:cutoff]
    kept = kept / kept.sum()
    choice = rng.choice(cutoff, p=kept)
    return int(order[choice])


def generate_text(
    model: Model,
    prompt: str,
    *,
    max_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: int | None = None,
    stop_token_ids: Iterable[int] | None = None,
    verbose: bool = False,
) -> str:
    rng = np.random.default_rng(seed)
    stop = set(stop_token_ids or ())
    stop.add(_GEMMA4_END_OF_TURN_ID)
    eos = getattr(model.tokenizer, "eos_token_id", None)
    if eos is not None:
        stop.add(int(eos))

    ids = model.tokenize(prompt)
    generated: list[int] = []
    for _ in range(max_tokens):
        result = model.run(ids)
        next_id = _sample_next(
            result.last_logits, temperature=temperature, top_p=top_p, rng=rng,
        )
        if next_id in stop:
            break
        generated.append(next_id)
        ids = mx.concatenate(
            [ids, mx.array([[next_id]], dtype=ids.dtype)], axis=1,
        )
        if verbose:
            tok = model.tokenizer.decode([next_id])
            print(tok, end="", flush=True)
    if verbose:
        print()
    return model.tokenizer.decode(generated)


def generate_labeled_corpus(
    model: Model,
    concept: str,
    *,
    topics: list[str],
    stories_per_topic: int = 4,
    max_tokens: int = 180,
    temperature: float = 0.9,
    top_p: float = 0.95,
    category: str | None = None,
    name: str | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> PromptSet:
    cat = category or f"emotion_{concept}"
    meta_template = (
        "Write a one-paragraph story (3-5 sentences) where a character "
        "experiences {concept}, on the topic of: {topic}. Describe the "
        "situation and the character's response. Do not name the emotion "
        "directly. Only write the story itself, no introduction. Story:"
    )
    prompts = []
    idx = 0
    for t_i, topic in enumerate(topics):
        for k in range(stories_per_topic):
            meta_prompt = meta_template.format(concept=concept, topic=topic)
            story = generate_text(
                model, meta_prompt,
                max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                seed=seed + idx, verbose=False,
            ).strip()
            if verbose:
                first_line = story.split("\n", 1)[0][:100]
                print(f"  [{concept:>8s}] {topic[:20]:>20s} #{k}  {first_line}")
            prompts.append(Prompt(text=story, category=cat))
            idx += 1
    return PromptSet(
        name=name or f"GENERATED_{concept.upper()}",
        prompts=tuple(prompts),
    )


_TURN_MARKERS = {"<end_of_turn>", "<turn|>", "<|im_end|>", "<|eot_id|>",
                 "<|endoftext|>"}


def _stop_ids(tokenizer) -> set[int]:
    stop: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stop.add(int(eos))
    unk = getattr(tokenizer, "unk_token_id", None)
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid, tok_obj in added.items():
        content = getattr(tok_obj, "content", str(tok_obj))
        if content in _TURN_MARKERS:
            stop.add(int(tid))
    for marker in _TURN_MARKERS:
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:  # noqa: BLE001
            continue
        if tid is not None and tid >= 0 and tid != unk:
            stop.add(int(tid))
    return stop


def offsets_by_cumulative_decode(tokenizer, ids):
    offs = []
    prev = ""
    for i in range(len(ids)):
        cur = tokenizer.decode(ids[: i + 1])
        offs.append((len(prev), len(cur)))
        prev = cur
    return offs, prev


def cut_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    cut = len(text)
    for s in stop_strings:
        if not s:
            continue
        at = text.find(s)
        if at != -1:
            cut = min(cut, at)
    return text[:cut]


def sample_completion_cached(model, prompt_ids, *, max_tokens=256,
                             temperature=0.9, top_p=0.95, rng=None,
                             prefill=None, return_ids=False,
                             stop_strings: Sequence[str] = (),
                             interventions=None):
    import numpy as _np

    from .distill import _copy_prefix_cache
    from .distill import prefill_decision as _prefill

    rng = rng or _np.random.default_rng()
    stop = _stop_ids(model.tokenizer)
    stops = tuple(s for s in (stop_strings or ()) if s)
    window = (max(len(s) for s in stops) + 8) if stops else 0

    ivs = list(interventions or [])
    if prefill is None:
        prefill = _prefill(model, list(prompt_ids), interventions=ivs)
    cache = _copy_prefix_cache(prefill[0])
    row = prefill[1]

    lm = model.lm
    out_ids: list[int] = []
    hit_stop = False
    for _ in range(int(max_tokens)):
        next_id = _sample_next(row, temperature=temperature,
                               top_p=top_p, rng=rng)
        if next_id in stop:
            break
        out_ids.append(int(next_id))
        if stops:
            tail = model.tokenizer.decode(out_ids[-window:])
            if any(s in tail for s in stops):
                hit_stop = True
                break
        if ivs:
            grew = model.tokenizer.decode([int(next_id)])
            for iv in ivs:
                on_token = getattr(iv, "on_token", None)
                if on_token is not None:
                    on_token(grew)
            res = model.run(mx.array([[int(next_id)]]), interventions=ivs,
                            kv_cache=cache)
            row = res.logits[0, -1, :].astype(mx.float32)
        else:
            o = lm(mx.array([[int(next_id)]]), cache=cache)
            row = (o.logits if hasattr(o, "logits")
                   else o)[0, -1, :].astype(mx.float32)
    text = model.tokenizer.decode(out_ids)
    if hit_stop:
        text = cut_at_stop(text, stops)
    return (text, out_ids) if return_ids else text
