"""Text generation on top of Model.run.

mlx_vlm's bundled generate function is broken for Gemma 4 (produces
repeated-token output, see README of this repo). We do our own naive
autoregressive loop on top of Model.run. It is slow (no KV cache — each
step is a full forward pass) but correct, and it is the only generator
we need for the corpus-generation workflow. Opitmising for throughput
is future work if generation becomes a bottleneck.

Usage:
    from mechbench_core import Model, generate_text

    model = Model.load()
    story = generate_text(
        model,
        "Write a one-paragraph story where a character experiences calm. "
        "Only write the story itself, no introduction. Story:",
        max_tokens=180, temperature=0.9, top_p=0.95, seed=42,
    )
    print(story)

For building labeled corpora across many concepts/topics, see
generate_labeled_corpus.
"""

from __future__ import annotations

from typing import Iterable

import mlx.core as mx
import numpy as np

from .model import Model
from .prompts import Prompt, PromptSet


# Gemma 4 chat-template turn-end marker. Generation stops when the model
# emits this token (or the tokenizer's eos).
_GEMMA4_END_OF_TURN_ID = 106


def _sample_next(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Sample the next token id from logits using temperature + top-p filtering.

    temperature=0 selects the argmax (greedy). Otherwise we renormalize
    probabilities, sort descending, take the smallest prefix whose cumulative
    probability exceeds top_p, and sample from that prefix.
    """
    if temperature <= 0:
        return int(np.argmax(np.array(logits.astype(mx.float32))))
    scaled = logits.astype(mx.float32) / float(temperature)
    probs = mx.softmax(scaled)
    mx.eval(probs)
    p = np.array(probs).astype(np.float64)
    # Descending sort by probability
    order = np.argsort(-p)
    sorted_p = p[order]
    cum = np.cumsum(sorted_p)
    # Smallest prefix whose cumulative mass >= top_p; always keep at least 1
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
    """Generate a text completion for `prompt`, returning only the model's output.

    The prompt is run through Model.tokenize (which applies Gemma's chat
    template), then generate loops calling Model.run() one token at a time
    until max_tokens or a stop-token is emitted. The prompt itself is NOT
    included in the returned string.

    Args:
        model: Model instance.
        prompt: The user message (will be wrapped by the chat template).
        max_tokens: Hard cap on generated token count.
        temperature: 0 for greedy, higher = more diverse. Default 0.9.
        top_p: Nucleus sampling cutoff. Default 0.95.
        seed: Random seed for reproducibility. None = fresh each call.
        stop_token_ids: Additional token ids that end generation. Gemma's
            end-of-turn (106) and the tokenizer's eos are always included.
        verbose: Print a running-token preview.

    Returns:
        str: The decoded generated text.
    """
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
    """Ask the model to write N stories per topic illustrating a concept.

    Produces `len(topics) * stories_per_topic` passages, each tagged with
    `category`. This is the recipe from Anthropic's 'Emotion Concepts'
    paper, simplified: instead of 100 topics x 12 stories per concept, let
    the caller pick a tractable scale.

    Each story is elicited with a meta-prompt like:
        "Write a one-paragraph story where a character experiences
         {concept}, on the topic of: {topic}. Only write the story itself,
         no introduction. Story:"

    The model's output is taken verbatim (no post-processing beyond
    tokenizer decoding). Some outputs will contain preambles like "Here's
    a story:" — for tiny-scale experiments the caller can filter those;
    for larger scales, the noise averages out in the difference-of-means
    centroid anyway.

    Args:
        model: Model instance.
        concept: The concept name ('calm', 'happy', etc.) for the prompt.
        topics: List of topic strings to vary the meta-prompt over.
        stories_per_topic: Number of independently-sampled stories per topic.
        max_tokens: Hard cap per story.
        temperature: Sampling temperature.
        top_p: Nucleus cutoff.
        category: Category label attached to each resulting Prompt. Defaults
            to f'emotion_{concept}'.
        name: PromptSet name. Defaults to f'GENERATED_{concept.upper()}'.
        seed: Base seed; each (topic, k) pair uses seed + index for
            reproducibility across runs.
        verbose: Print each generated story's first line.

    Returns:
        PromptSet with one Prompt per generated story.
    """
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


def _stop_ids(tokenizer) -> set[int]:
    """Family-generic stop set: eos plus an end-of-turn token when the
    vocabulary has one (Gemma's <end_of_turn>, chat-template families)."""
    stop: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stop.add(int(eos))
    for marker in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:  # noqa: BLE001 — tokenizer-family differences
            continue
        if tid is not None and tid >= 0:
            stop.add(int(tid))
    return stop


def offsets_by_cumulative_decode(tokenizer, ids):
    """Per-token [start, end) char offsets into the decoded text, by
    cumulative decode — byte-exact against tokenizer join behavior
    (the 019 backfill's reconstruction approach). Returns (offsets,
    text)."""
    offs = []
    prev = ""
    for i in range(len(ids)):
        cur = tokenizer.decode(ids[: i + 1])
        offs.append((len(prev), len(cur)))
        prev = cur
    return offs, prev


def sample_completion_cached(model, prompt_ids, *, max_tokens=256,
                             temperature=0.9, top_p=0.95, rng=None,
                             prefill=None, return_ids=False):
    """Sample one completion with a KV cache: the prompt is encoded
    once (or reused via `prefill` — a (cache, last_row) pair from
    `distill.prefill_decision`, copied per call), then decoding feeds
    one token per forward. Replaces the O(n^2) full-re-encode loop of
    `generate_text` for block-scale generation (task 000258 arc D).

    Deterministic in `rng`: pass a seeded numpy Generator; the sampler
    draws only from it.
    """
    import numpy as _np

    from .distill import prefill_decision as _prefill
    from .distill import _copy_prefix_cache

    rng = rng or _np.random.default_rng()
    stop = _stop_ids(model.tokenizer)

    if prefill is None:
        prefill = _prefill(model, list(prompt_ids))
    cache = _copy_prefix_cache(prefill[0])
    row = prefill[1]

    lm = model.lm
    out_ids: list[int] = []
    for _ in range(int(max_tokens)):
        next_id = _sample_next(row, temperature=temperature,
                               top_p=top_p, rng=rng)
        if next_id in stop:
            break
        out_ids.append(int(next_id))
        o = lm(mx.array([[int(next_id)]]), cache=cache)
        row = (o.logits if hasattr(o, "logits")
               else o)[0, -1, :].astype(mx.float32)
    text = model.tokenizer.decode(out_ids)
    return (text, out_ids) if return_ids else text
