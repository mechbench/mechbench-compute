from __future__ import annotations

from typing import Any

from mechbench_compute import lexicon
from mechbench_compute._mlx import mx
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="logits/read",
    requires="mlx-local",
    summary=(
        "Read the model's exact next-token distribution at a decision point "
        "— entropy, the top tokens, and the probability mass on each named "
        "outcome — for every chat-shaped record."
    ),
    description="""\
Each record is rendered as a chat and, when it carries a `prefill`, the
assistant's turn is begun with it, so the read happens at the first token
*after* the prefill — `'{ "name": "'` reads the first token of a JSON value.
One prefill pass per record gives the full distribution; nothing is sampled.

`tracked` turns the read into a forced choice: each named token is
tokenized as a continuation of the rendered prompt and the probability of
its first token is recorded under `tracked` by its name — `{"1": "1", …,
"6": "6"}` for a die. A `rollout` goes further, expanding the most probable
*complete* outcomes token by token (best-first, reusing the prompt cache)
so that multi-token answers are compared as wholes.

`complete` reads a named set of multi-token outcomes **exactly**: each
outcome followed by its `closer` is scored by teacher forcing, and its
probability goes into `tracked` under its own name, with `text`, the
number of `tokens`, `p` (eight decimals) and `logp`. The closer is what
makes an outcome complete: "Mystery" is scored as `Mystery"`, so it does
not also count the mass of "Mystery Thriller". An `opener` is text before
each outcome that is not part of its name — after a list's comma the next
outcome is scored as " Humor" and recorded as "Humor". `complete_mass` is
the total over the set, which is how much of what the model says at all
falls in it, and `complete_entropy_bits` the entropy of that mass
renormalized. `eval/expect` then judges the set as it judges tokens, so a
791-outcome target is checked against the model outcome by outcome. A
record's own `complete` is laid over the block's field by field: a probe
midway through a list names its opener and closer and keeps the block's
outcomes.

Records keep their `coords`, so a grid of conditions comes out as a grid of
readings.
""",
    inputs=(
        In("conditions", "records/record",
           "Chat-shaped records: `user` (required), `system` and `prefill` "
           "(optional), an `id`, and optionally `coords`; a record may carry "
           "its own `tracked`.", many=True),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('logits/decision', collection=True, doc='One item per input record: `id`, `coords`, `entropy_bits`, `top` (the `top_k` most probable tokens, each `{token, p, logp}`), `tracked` (each tracked token by name, `{token, p, logp}`; each complete outcome by name, `{text, tokens, p, logp}`), `complete_mass` and `complete_entropy_bits` with `complete`, and `rollout` when one was requested. The header carries `top_k`.'),
    params=(
        P("tracked", "map[string, string]",
          "Tokens to follow by name, `{\"yes\": \" Yes\"}` — the candidate "
          "answers of a forced choice, each recorded under `tracked`. A "
          "record's own `tracked` takes precedence.",
          None),
        P("top_k", "int", "How many of the most likely tokens to record.", 10),
        P("rollout", "object",
          "Expand complete multi-token outcomes best-first from the "
          "decision point: `{\"top_k\": 10, \"max_tokens\": 8, "
          "\"max_forwards\": 128, \"floor\": 0.001, \"terminators\": "
          "[\"\\\"\"]}` — keep the `top_k` most probable completions, each "
          "at most `max_tokens` long, stopping at a terminator, pruning "
          "branches below `floor`, within a budget of `max_forwards` model "
          "calls.",
          None, fields=(
              P("top_k", "int", "How many complete outcomes to keep.", 10),
              P("max_tokens", "int", "The longest outcome, in tokens.", 8),
              P("max_forwards", "int", "The budget of model calls, the prefill included.", 128),
              P("floor", "float", "Prune a path whose whole probability falls below this.", 0.001),
              P("terminators", "list[string]",
                "An outcome is complete at the first token containing one of these.", ['"']),
          )),
        P("complete", "object",
          "Score a set of complete outcomes exactly: `{\"items\": [...], "
          "\"closer\": \"\\\"\"}`, recorded under `tracked`. A record's own "
          "`complete` is laid over this one field by field.",
          None, fields=(
              P("items", "list[string] | object",
                "The outcomes: a list, or a target spec (`weights` or `uniform`, with any "
                "`transform`) whose support is read, so a `top_k` reads a rung's own vocabulary. "
                "A stored word list may be given by reference.",
                None, fields=(
                    P("uniform", "list[string]",
                      "The outcomes, weighted equally. Wins over `weights` when "
                      "both are given.",
                      None),
                    P("weights", "map[string, float]",
                      "Outcome → weight, each finite and at least 0: raw corpus "
                      "frequencies, say. A stored word list may be given by "
                      "reference.",
                      None, stored="text/word-list"),
                    P("transform", "list[object]",
                      "Steps that reshape the distribution, applied in order; "
                      "the result is always normalised.",
                      [], fields=(
                          P("op", "string",
                            "The step.",
                            choices=("sqrt", "pow", "temper", "temper_to_entropy", "mix_uniform", "top_k", "normalize")),
                          P("exponent", "float",
                            "For `pow`: the power each weight is raised to.",
                            None),
                          P("temperature", "float",
                            "For `temper`: divides the log-weights; above 0.",
                            None),
                          P("bits", "float",
                            "For `temper_to_entropy`: the entropy to reach, in "
                            "bits.",
                            None),
                          P("tolerance", "float",
                            "For `temper_to_entropy`: how close is close enough, "
                            "in bits.",
                            0.0001),
                          P("epsilon", "float",
                            "For `mix_uniform`: the share of uniform mixed in, "
                            "from 0 to 1.",
                            None),
                          P("k", "int",
                            "For `top_k`: how many of the heaviest outcomes to "
                            "keep.",
                            None),
                      )),
                ),
                stored="text/word-list"),
              P("opener", "string",
                "Text before each outcome that is not part of its name, such as the space "
                "after a list's comma.", ""),
              P("closer", "string", "The text that ends an outcome.", '"'),
          )),
    ),
    example={
        "model": {"$param": "model"},
        "tracked": {"red": "red", "blue": "blue", "green": "green"},
    },
    example_inputs={"conditions": {"$ref": {"bench": "you/lab/conditions"}}},
)


def run(ctx, inputs, params):
    import numpy as np

    from mechbench_compute.distill import (
        expand_top_outcomes_cached,
        prefill_decision,
        render,
        score_complete,
        suffix_tokens,
    )

    model = ctx.model(params.get("model"))
    tok = model.tokenizer
    conditions = lexicon.items_of(inputs.get("conditions") or [])
    if not conditions:
        raise ValueError(
            "logits/read: no conditions to read — wire records to "
            "the `conditions` port")
    if ctx.on_start:
        ctx.on_start(len(conditions))
    rollout = params.get("rollout")
    complete = params.get("complete")
    top_k = int(params.get("top_k", 10))
    from mechbench_compute import shapes as S
    out = []
    for cond in conditions:
        key = str(cond["id"])
        if ctx.resume_items and key in ctx.resume_items:
            out.append(ctx.resume_items[key])
            if ctx.on_item:
                ctx.on_item(key, ctx.resume_items[key], True)
            continue
        r = render(model, cond)
        rendered, ids = r.text, r.ids
        prefill = prefill_decision(model, ids)
        lp = np.array(prefill[1] - mx.logsumexp(prefill[1])).astype(np.float64)
        tracked: dict[str, int] = {}
        for o in (cond.get("outcomes") or []):
            tracked[str(o)] = int(suffix_tokens(tok, rendered, ids, o)[0])
        for name, text in dict(cond.get("tracked") or params.get("tracked") or {}).items():
            tracked.setdefault(str(name), int(suffix_tokens(tok, rendered, ids, str(text))[0]))
        entry: dict[str, Any] = {
            "id": cond["id"],
            "coords": dict(cond.get("coords", {})),
            **S.distribution(lp, tok, top_k=top_k, tracked=tracked),
        }
        if rollout:
            entry["rollout"] = expand_top_outcomes_cached(
                model, tok, ids, rollout, prefill=prefill)
        spec = {**(complete or {}), **(cond.get("complete") or {})}
        if spec:
            scored, mass, entropy = score_complete(model, tok, rendered, ids, spec)
            entry["tracked"] = {**(entry.get("tracked") or {}), **scored}
            entry["complete_mass"] = round(mass, 6)
            entry["complete_entropy_bits"] = round(entropy, 4)
        out.append(entry)
        if ctx.on_item:
            ctx.on_item(key, entry)
    return lexicon.collection("logits/decision", out, top_k=top_k)
