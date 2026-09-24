from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute import thinking as THINK
from mechbench_compute.chat.constants import LOCAL
from mechbench_compute.chat.count_endings import count_endings
from mechbench_compute.chat.describe_reasoning_only import describe_reasoning_only
from mechbench_compute.chat.read_local_ending import read_local_ending
from mechbench_compute.chat.split_reasoning import find_delimiters, split_reasoning
from mechbench_compute.lexicon._base import In, Op, Output, P
from mechbench_compute.protocol.read_tokenizer_id import read_tokenizer_id
from mechbench_compute.protocol.serialize_model import serialize_model
from mechbench_compute.providers import messages as pm

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
closed is distinguishable from one that did, and the header's `ended`
counts the items by ending, so a glance says whether any was cut off.

A model whose vocabulary declares reasoning markup (Gemma 4's thinking
channel, `<|channel>thought\\n…<channel|>`) can write a thought before its
reply, even with thinking off. The thought is never `text`: it goes to
the item's `reasoning`, as `text/chat` does, and `text` keeps only the
prose after it. A thought never closed runs to the end of the sample and
is reasoning; an item that is reasoning alone has an empty `text`,
`ended: "empty"` and `metadata.empty` with cause `reasoning`. A sample
with no markup, and every sample of a model that declares none, is kept
exactly as the model wrote it. The trace, at trace fidelity, is the raw
token stream, thought included.

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
        In("intervention", "intervene/spec",
           "An intervention declared as an object — its `items` are spec "
           "items in the grammar `intervene/apply` documents — applied "
           "during every forward pass this node runs. A `direction` or "
           "`source` an item needs arrives on the port of that name. "
           "Where the node also has an inline `spec` or `intervention` "
           "param, the param wins when both are given.",
           required=False),
        In("direction", "direction/vector",
           "A direction that fills any spec item without one.", required=False),
        In("source", "activations/vector | intervene/readout",
           "A collection of `activations/vector` — a capture, intervened or "
           "not — that fills any `mean`/`resample`/`patch` item without one. "
           "A capture readout stored before 0.110.0 is read too.",
           many=True, required=False),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only — "
           "from an `adapter/train` node, a `{\"$ref\": {\"hf_adapter\": {\"repo\": …}}}` "
           "reference, or a stored adapter. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` "
           "scales this one.",
           required=False),
    ),
    output=Output('text/document', collection=True, doc="`n` items per record, ids `<record id>-s<k>`: `text` (prose only), `reasoning` (only when the model wrote a thought: a list of `{text, provider: \"local\", model}`, in order), `coords` (the record's, plus `sample: k`), `metadata.sampling` (with `ended`, and the `prefill` and `stop` when used), `metadata.empty` when the item is reasoning alone, and the wire form of the model. At trace fidelity each item also has `trace` (`token_ids`, `text`, `offsets`, `generation_spans`) and `segmentations`. The header carries `fidelity` and `ended`: the items counted by `metadata.sampling.ended`, every ending `text/chat` names present and zero when none (`{\"end\": 3, \"stop\": 0, \"max_tokens\": 1, …}`), so `max_tokens` above zero means samples were cut off at the limit. A header without `ended` was stored before the count existed: its items carry `metadata.sampling.ended` from 0.99.0 on, and the count is theirs to take. Under an intervention, ids are `<record id>-s<k>-f<factor>`, every item carries `factor` in its `coords`, and the header carries `spec` (the items as run, objects replaced by their provenance), `weights` (parameter edits, when any) and `sweep` (the factors, `0.0` first when a control was added)."),
    params=(
        P("spec", "list[object]",
          "An intervention's items, applied at every forward pass — the "
          "prompt, then every decoding step — in the grammar `intervene/apply` "
          "documents under *Spec items*: `positions: \"last\"` is the token "
          "being produced, `\"all\"` every token so far, `\"generated\"` what "
          "the model has said. Or the items arrive as an `intervene/spec` on "
          "the `intervention` port. Without either, plain sampling.",
          None, fields=(
              P("point", "string", "Where in the forward pass to act.", "resid_post", value="point"),
              P("parameter", "string",
                "Edit this weight instead of an activation, named as the module tree names it; "
                "`*` stands for one segment.",
                None),
              P("layers", "int | list[int] | \"all\"", "Which layers the item applies to.", "all"),
              P("positions", "selector", "Which token positions.", "last"),
              P("heads", "int | list[int]", "Only these attention heads, at a point with a head axis.", None),
              P("neurons", "int | list[int]", "Only these indices along the feature axis.", None),
              P("op", "string", "What to do there — the table above lists each op and what it needs.", "zero",
                choices=("zero", "mean", "resample", "patch", "add", "scale", "clamp", "project_out",
                         "rotate", "truncate")),
              P("strength", "float", "The item's magnitude, multiplied by each sweep factor.", 1.0),
              P("direction", "json",
                "The direction, usually a stored one (`{\"$ref\": …}`); or it arrives on the node's `direction` port.", None),
              P("direction2", "json", "For `rotate`: the second axis of the plane.", None),
              P("source", "json",
                "For `mean`, `resample` and `patch`: the replacement activations, or they arrive on the "
                "node's `source` port.",
                None),
              P("row", "object", "For `patch`: which row of `source` to write in.", None,
                fields=(P("index", "int", "The row's index.", 0),)),
              P("condition", "object",
                "Act only where the activation projects onto a direction above (or below) a threshold.", None,
                fields=(
                    P("direction", "json", "The direction projected onto."),
                    P("threshold", "float", "The projection's threshold.", 0.0),
                    P("above", "bool", "Act above the threshold; `false` acts below it.", True),
                )),
              P("except", "bool",
                "Invert the sets this item names — every layer, head or neuron BUT "
                "those — which measures a circuit's completeness where the direct "
                "ablation measures its faithfulness.",
                False),
              P("from", "object",
                "For `patch`: where the row is read, when that is not where it is "
                "written — the patchscope's move.", None, fields=(
                    P("layer", "int", "The source layer."),
                    P("point", "string", "The source point; the item's own by default.", None, value="point"),
                )),
              P("pattern", "object",
                "At `attn.scores` or `attn.weights`: the attention EDGE to act on — "
                "which source positions the selected destinations may attend to.",
                None, fields=(
                    P("from", "selector", "The source (key) positions."),
                    P("to", "selector", "The destination (query) positions; the item's `positions` by default.", None),
                )),
              P("renormalize", "bool",
                "At `attn.weights`, after zeroing: rescale the rows that lost mass so "
                "they sum to one again. The rows that lost none are left as they are.",
                True),
              P("seed", "int", "The seed `resample` draws with; the node's `seed` by default.", None),
              P("sweep_over", "list[string]",
                "Which of the node's sweep axes vary THIS item — `[\"layers\"]` sweeps "
                "this item's layers and leaves its strength alone. Every axis, by "
                "default.",
                None, choices=("strength", "layers", "heads", "positions", "neurons")),
              P("side", "string",
                "For a weight's `project_out`: the side facing the residual stream, where the module's "
                "name does not imply it.",
                None, choices=("in", "out")),
              P("rank", "int", "For a weight's `truncate`: how many singular directions to keep.", None),
          )),
        P("sweep", "object",
          "The axes to vary, each a list of values the spec items' field of "
          "that name takes in turn: `{\"strength\": [0.5, 1.0, 2.0]}` scales "
          "every item's `strength`; `{\"layers\": [0, 1, 2]}` runs the spec at "
          "each layer; `heads`, `positions` and `neurons` likewise. Several "
          "axes are a cartesian product, run with `strength` outermost, and "
          "each becomes a coordinate on every row — `factor`, `layer`, `head`, "
          "`position`, `neuron` — so a sweep is summarised, compared and "
          "plotted on the axis it varied.",
          {"strength": [1.0]}, fields=(
              P("strength", "list[float]", "The factors; `0` is the untouched model.", [1.0]),
              P("layers", "list[json]",
                "The layers, one cell each: `[0, 1, 2]`, or `[[0,1],[2,3]]` for "
                "groups. The coordinate is the layer, or `0+1` for a group.", None),
              P("heads", "list[json]", "The attention heads, one cell each.", None),
              P("positions", "list[selector]", "The position selectors, one cell each.", None),
              P("neurons", "list[json]", "The feature indices, one cell each.", None),
          )),
        P("control", "bool",
          "Add a factor‑0 run — the model untouched — to the sweep, so every "
          "record has a baseline (`factor: 0.0`). One run however many axes "
          "the sweep has: an unintervened pass does not depend on the layer "
          "the intervention would have named. Set `false` when the sweep "
          "already contains a strength of `0` or no baseline is wanted.",
          True),
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
    continue_prefill = bool(params.get("continue_prefill", False))
    stop_strings = tuple(s for s in (params.get("stop") or ()) if s)
    delimiters = find_delimiters(tok)
    model_name = str(getattr(params.get("model"), "base", params.get("model")))

    from mechbench_compute import intervene as intervene_mod
    from mechbench_compute.generate import offsets_by_cumulative_decode

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
                prompt_tokens = [tok.decode([int(t)]) for t in ids] if plan else []
                prefill = (prefill_decision(model, ids, interventions=plan.live(cell, prompt_tokens, rec))
                           if plan else prefill_decision(model, ids))
                for k in range(start, start + n):
                    key = f"{rec['id']}:{k}" + (f":{cell.slug}" if plan else "")
                    if ctx.resume_items and key in ctx.resume_items:
                        items.append(ctx.resume_items[key])
                        if ctx.on_item:
                            ctx.on_item(key, ctx.resume_items[key], True)
                        continue
                    rng = _np.random.default_rng(item_seed(seed, rec["id"], k))
                    text, out_ids = sample_completion_cached(
                        model, ids, max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p, rng=rng,
                        prefill=prefill, return_ids=True,
                        stop_strings=stop_strings,
                        **({"interventions": plan.live(cell, prompt_tokens, rec)} if plan else {}))
                    ended = read_local_ending(tok, out_ids, stop_strings=stop_strings,
                                              max_tokens=max_tokens)
                    thought, text = split_reasoning(text, delimiters)
                    if thought and not text:
                        ended = "empty"
                    coords = {**rec.get("coords", {}), "sample": k}
                    if plan:
                        coords.update(cell.axes)
                    item = {
                        "id": f"{rec['id']}-s{k}" + (f"-{cell.slug}" if plan else ""),
                        "kind": "text/document",
                        "text": lead + text,
                        "coords": coords,
                        "metadata": {
                            "coords": dict(coords),
                            "sampling": {"temperature": temperature,
                                         "top_p": top_p, "seed": seed,
                                         "index": k, "ended": ended,
                                         **({"prefill": lead} if lead else {}),
                                         **({"stop": list(stop_strings)} if stop_strings else {})},
                            "model": serialize_model(params.get("model")),
                        },
                    }
                    if thought:
                        item["reasoning"] = pm.read_reasoning(
                            pm.ReasoningPart(text=t, provider=LOCAL, model=model_name)
                            for t in thought)
                        if not text:
                            item["metadata"]["empty"] = describe_reasoning_only(
                                model_name, max_tokens)
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
        ended=count_endings(items),
        **(plan.header() if plan else {}))
