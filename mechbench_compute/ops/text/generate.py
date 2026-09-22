from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute import thinking as THINK
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.lexicon.model import (
    _DIRECTION_PORT,
    _SOURCE_PORT,
    _SPEC_FIELDS,
    _SWEEP_PARAMS,
    ADAPTER,
    INTERVENTION,
)
from mechbench_compute.protocol.read_tokenizer_id import read_tokenizer_id
from mechbench_compute.protocol.serialize_model import serialize_model

OP = Op(
    name="text/generate",
    requires="mlx-local",
    summary=(
        "Sample completions from the model for each chat-shaped record — n "
        "per record, reproducibly seeded — into a document collection."
    ),
    description="""\
Each record is rendered as a chat (`system` and `user` turns) and prefilled
once; then `n` completions are sampled from that prefix with the given
temperature and nucleus settings. Every sample's random stream is derived
from (`seed`, record id, sample index), so a sample is a pure function of
its key: running the same node again reproduces the same texts, and growing
a corpus later is the same node over a later `start` range, unioned with
the first.

`continue_prefill` begins each sample's assistant turn with the record's
`prefill`, so the model continues exactly the envelope a `logits/read` of
the same record reads, and a training step conditions on: `'{ "genre": "'`
samples the genre itself, not the model's choice of JSON layout. The
item's `text` is the prefill followed by what the model wrote. `stop` ends
a sample at the first of its strings, which are not kept in the text:
with `stop: ["\""]` a sample is the answer and nothing after it. Each
item's `metadata.sampling.ended` says how it ended — `"stop"`, `"end"`
(the model ended its turn) or `"max_tokens"` — so an answer that never
closed is distinguishable from one that did.

With `fidelity: "trace"` each item also keeps its token ids, character
offsets and the prompt/body segmentation, which is what `score` needs to
annotate it token by token.

### Sampling under an intervention

An intervention — inline `spec` items, or an `intervene/spec` object on
the `intervention` port — is live at every forward pass the node runs: the prompt's prefill
and then each decoding step, with the same KV cache. So "add the direction
at layer 14 and read what the model then writes" is this node with one
item, and a `sweep` over strengths is the same node producing one set of
samples per factor, `factor` a coordinate on every item, which
`text/measure`, `eval/judge` and `records/summarize` group by unchanged.
Factor `0` (the `control`, on by default) is plain sampling: under the same
seed it reproduces the un-intervened sample byte for byte.
""",
    inputs=(
        In("records", "records/record",
           "Chat-shaped records: `user` (required), `system` and `prefill` "
           "(optional; the prefill is read only with `continue_prefill`), "
           "an `id`, and optionally `coords`.", many=True),
        INTERVENTION,
        _DIRECTION_PORT,
        _SOURCE_PORT,
        ADAPTER,
    ),
    output=Output('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text`, `coords` (the record's, plus `sample: k`), `metadata.sampling` (with `ended`, and the `prefill` and `stop` when used), and the wire form of the model. At trace fidelity each item also has `trace` (`token_ids`, `text`, `offsets`, `generation_spans`) and `segmentations`. The header carries `fidelity`. Under an intervention, ids are `<record id>-s<k>-f<factor>`, every item carries `factor` in its `coords`, and the header carries `spec` (the items as run, objects replaced by their provenance), `weights` (parameter edits, when any) and `sweep` (the factors, `0.0` first when a control was added)."),
    params=(
        P("spec", "list[object]",
          "An intervention's items, applied at every forward pass — the "
          "prompt, then every decoding step — in the grammar `intervene/apply` "
          "documents under *Spec items*: `positions: \"last\"` is the token "
          "being produced, `\"all\"` every token so far, `\"generated\"` what "
          "the model has said. Or the items arrive as an `intervene/spec` on "
          "the `intervention` port. Without either, plain sampling.",
          None, fields=_SPEC_FIELDS),
        *_SWEEP_PARAMS,
        P("n", "int", "How many completions to sample per record.", 1),
        P("start", "int",
          "The first sample index. Indices run `start` … `start + n − 1`; "
          "a later node with a later `start` extends the corpus without "
          "re-sampling what exists.",
          0),
        P("temperature", "float",
          "Sampling temperature. `0` would be greedy; the default is a "
          "creative-writing setting.",
          0.9),
        P("top_p", "float",
          "Nucleus sampling: only the smallest set of tokens whose "
          "probabilities sum to `top_p` is sampled from.",
          0.95),
        P("max_tokens", "int", "The longest completion, in tokens.", 256),
        P("continue_prefill", "bool",
          "Begin each sample's assistant turn with the record's `prefill`, and "
          "keep it at the front of the item's `text`.",
          False),
        P("stop", "list[string]",
          "Strings at which a sample stops; the marker itself is not part of "
          "the text.",
          None),
        P("fidelity", "string",
          "`\"text\"` keeps the completion text; `\"trace\"` also keeps token "
          "ids, offsets and spans, so the collection can be scored token by "
          "token.",
          "text", choices=("text", "trace")),
    ),
    example={
        "model": {"$param": "model"},
        "n": 4,
        "temperature": 0.9,
        "max_tokens": 200,
        "fidelity": "trace",
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}},
)


def run(ctx, inputs, params):
    """The Generate model block: per condition record, sample n
    completions into a text-fidelity DocumentCollection. Range
    rule (epic 000258 amendment 4): each sample's rng derives from
    (seed, record id, index), indices [start, start+n) — growing a
    corpus is the same node over a later range plus Union. The
    prompt is prefilled once per record and the KV cache copied
    per sample."""
    import numpy as _np

    from mechbench_compute.distill import prefill_decision, render
    from mechbench_compute.generate import sample_completion_cached
    from mechbench_compute.seeds import item_seed

    model = ctx.model(params.get("model"))
    tok = model.tokenizer
    records = lexicon.items_of(inputs.get("records") or [])
    if not records:
        raise ValueError(
            "text/generate: no records to run over — wire records to "
            "the `records` port")
    n = int(params.get("n", 1))
    start = int(params.get("start", 0))
    seed = params.get("seed", 0)
    temperature = float(params.get("temperature", 0.9))
    top_p = float(params.get("top_p", 0.95))
    max_tokens = int(params.get("max_tokens", 256))
    fidelity = params.get("fidelity", "text")
    if fidelity not in ("text", "trace"):
        raise ValueError(f"generate: unsupported fidelity {fidelity!r}")
    # A record's prefill begins the assistant's turn when asked
    # (000549): the samples then continue exactly the envelope a
    # decision read and a training step condition on, and `stop`
    # ends them where the answer does. Off, the prefill is dropped,
    # as it always was.
    continue_prefill = bool(params.get("continue_prefill", False))
    stop_strings = tuple(s for s in (params.get("stop") or ()) if s)

    from mechbench_compute import intervene as intervene_mod
    from mechbench_compute.generate import offsets_by_cumulative_decode

    # An intervention (000601) — inline items or an intervene/spec on
    # the port — makes the node a sweep: one set of samples per
    # cell, its axes coordinates, weight edits scoped per strength.
    # Without one, `cells` is a single None and the loop below is the
    # one it always was, byte for byte.
    plan = intervene_mod.plan(model, params, inputs)
    cells: list = plan.cells if plan else [None]

    if ctx.on_start:
        ctx.on_start(len(records) * n * len(cells))
    items = []
    for cell in cells:
        with intervene_mod.edit_weights(model, plan.weight_items if plan else (),
                                        cell.factor if plan else 0.0):
            for rec in records:
                lead = str(rec.get("prefill") or "") if continue_prefill else ""
                r = render(model, dict(rec, prefill=lead))
                rendered, ids = r.text, r.ids
                # The prefill's hooks see the prompt's tokens; each
                # sample then gets its own live intervention, over a
                # token list its decoder grows.
                prompt_tokens = [tok.decode([int(t)]) for t in ids] if plan else []
                prefill = (prefill_decision(model, ids, interventions=plan.live(cell, prompt_tokens, rec))
                           if plan else prefill_decision(model, ids))
                for k in range(start, start + n):
                    key = f"{rec['id']}:{k}" + (f":{cell.slug}" if plan else "")
                    if ctx.resume_items and key in ctx.resume_items:
                        # Reproducible (epic 000320): this item is a pure
                        # function of its key; the spooled copy IS what
                        # this loop would produce. Same position, same
                        # bytes.
                        items.append(ctx.resume_items[key])
                        if ctx.on_item:
                            ctx.on_item(key, ctx.resume_items[key], True)
                        continue
                    # The item rule (000258 am. 4), now in seeds.py (000402):
                    # a leaf's seed comes from its key, never its position.
                    rng = _np.random.default_rng(item_seed(seed, rec["id"], k))
                    text, out_ids = sample_completion_cached(
                        model, ids, max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p, rng=rng,
                        prefill=prefill, return_ids=True,
                        stop_strings=stop_strings,
                        **({"interventions": plan.live(cell, prompt_tokens, rec)} if plan else {}))
                    if stop_strings and any(s in tok.decode(out_ids) for s in stop_strings):
                        ended = "stop"
                    elif len(out_ids) >= max_tokens:
                        ended = "max_tokens"
                    else:
                        ended = "end"
                    coords = {**rec.get("coords", {}), "sample": k}
                    if plan:
                        coords.update(cell.axes)
                    item = {
                        "id": f"{rec['id']}-s{k}" + (f"-{cell.slug}" if plan else ""),
                        "kind": "text/document",
                        # The assistant's turn as it reads: the prefill it was
                        # begun with, then what the model wrote.
                        "text": lead + text,
                        # A document is a record: coords on the item, and
                        # under metadata where older readers look.
                        "coords": coords,
                        "metadata": {
                            "coords": dict(coords),
                            "sampling": {"temperature": temperature,
                                         "top_p": top_p, "seed": seed,
                                         "index": k, "ended": ended,
                                         **({"prefill": lead} if lead else {}),
                                         **({"stop": list(stop_strings)} if stop_strings else {})},
                            # The wire form, never the resolved object: the
                            # object carries the adapter bytes (000488).
                            "model": serialize_model(params.get("model")),
                        },
                    }
                    if fidelity == "trace":
                        full_ids = list(ids) + list(out_ids)
                        offs, full_text = offsets_by_cumulative_decode(
                            tok, full_ids)
                        item["trace"] = {
                            "token_ids": [int(t) for t in full_ids],
                            "tokenizer": read_tokenizer_id(params.get("model")),
                            "text": full_text,
                            "offsets": [[int(a), int(b)] for a, b in offs],
                            "generation_spans": [{
                                "token_start": len(ids),
                                "token_end": len(full_ids),
                                "model": serialize_model(params.get("model")),
                                "temperature": temperature,
                                "top_p": top_p,
                                "seed": k,
                            }],
                        }
                        item["segmentations"] = [{
                            "schema_name": "envelope",
                            "segments": [
                                {"role": "prompt", "token_start": 0,
                                 "token_end": len(ids)},
                                {"role": "body", "token_start": len(ids),
                                 "token_end": len(full_ids)},
                            ],
                        }]
                        # Where the model reasoned, when its own vocabulary
                        # declares reasoning delimiters (task 000592). A
                        # second segmentation beside the envelope, so a
                        # reader that knows only `prompt`/`body` is
                        # unaffected and a position selector can name the
                        # thinking span.
                        reasoning = THINK.segmentation(
                            full_ids, start=len(ids), pair=THINK.delimiter_ids(tok))
                        if reasoning is not None:
                            item["segmentations"].append(reasoning)
                    items.append(item)
                    if ctx.on_item:
                        ctx.on_item(key, item)
    return lexicon.collection(
        "text/document", items,
        name=params.get("name", "generated"),
        description=params.get("description", ""),
        fidelity=fidelity,
        **(plan.header() if plan else {}))
