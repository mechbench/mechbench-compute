from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="adapter/train",
    requires="mlx-local",
    summary=(
        "Train a LoRA adapter that shapes what the model says at a decision "
        "point toward a target distribution over outcomes — and emit the "
        "adapter as an object whose lineage is the training's methods "
        "section."
    ),
    description="""\
Training data are chat-shaped prompt records; at the point where the
assistant's turn begins (after any `prefill`), the model is trained toward
a **target distribution** over outcome strings rather than toward one
answer — soft-target cross-entropy. `target` describes that distribution:
`{"uniform": [outcomes]}`, or `{"weights": {outcome: weight}}` (raw corpus
frequencies, say), optionally reshaped by a `transform` chain — `sqrt`,
`pow`, `temper`, `temper_to_entropy`, `mix_uniform`, `top_k`, `normalize` —
so one frequency table can be trained flat, tempered or inverted by
declaration.

An outcome may be many tokens long ("Science Fiction Fantasy"). The
default items train the first token as a soft row and one second token
per outcome, which is exact for outcomes of one or two tokens. `path`
items train the **whole trie**: each step draws outcomes by their target
mass and trains a soft row at every token of each one, the `closer`
included, so every token is trained toward the distribution of what can
follow it, in proportion to the mass that reaches it. The closer is how an
outcome ends: after "Mystery" the row holds both the closing quote and
" Thriller".

With `depth` > 1 the outcome is a *sequence* of slots (a list of three
colours), each slot with its own target (`per_slot`) or all sharing one;
`join` and `closer` are the text between and after them. Sequences are
sampled fresh every step, and `replace: false` draws them without
replacement: a slot draws only from the outcomes not yet drawn, their
weights renormalized, so no outcome repeats.

A slot is one token (`unit: "token"`) or a whole outcome (`unit:
"item"`). With token slots the **naturalism gate** checks first that every
sampled sequence tokenizes to exactly one token per slot, so slot *i* is
token *i*: an adapter trained on a vocabulary the tokenizer splits would
be training on noise. With item slots each step trains `path` items
through the list: a soft row at every token, each over the outcomes still
available to that slot, so the rows themselves carry the no-repeats rule.
An outcome's first-slot tokens differ from its tokens after the join
("Science" against " Science"), and the gate checks every outcome in
both places.

`anchors` are prompts with a known correct `answer`, mixed into each batch
so the adapter learns the distribution without forgetting how to answer.

If the model reference already carries adapters, training happens on the
fused stack — the new round learns a delta on top. Long runs checkpoint
every `checkpoint_every` steps and resume from the same trajectory.

`seed` fixes the whole run: the adapter's initial weights as well as the
sampling order, so two runs of the same node on the same machine produce
a byte-identical adapter. Change the seed to see the spread a different
draw gives.
""",
    inputs=(
        In("records", "records/record",
           "The training prompts: chat-shaped records with `user`, and "
           "optionally `system` and `prefill`.", many=True),
        In("anchors", "records/record",
           "Prompt records with a known `answer`, mixed into each batch.",
           many=True, required=False),
    ),
    output=Output('adapter/lora', collection=False, doc="`data` (safetensors bytes), `format`, `base_model`, `trained_on` (the base and any prior adapters), `lora` (rank, alpha, scale, target modules, parameter count) and `train` (steps, lr, seed, batch, final loss, the target spec, depth, unit, replace, positions, counts). Wire it into a later node's `adapter` port, or `adapter/publish`."),
    params=(
        P("target", "object",
          "The target distribution — `{\"uniform\": [...]}` or "
          "`{\"weights\": {...}}`, with optional `transform` steps, "
          "`depth`, `join`, `per_slot`. See above.", fields=(
              P("uniform", "list[string]",
                "The outcomes, weighted equally. Wins over `weights` when both "
                "are given.",
                None),
              P("weights", "map[string, float]",
                "Outcome → weight, each finite and at least 0: raw corpus "
                "frequencies, say. A stored word list may be given by reference.",
                None, stored="text/word-list"),
              P("transform", "list[object]",
                "Steps that reshape the distribution, applied in order; the "
                "result is always normalised.",
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
                      "For `temper_to_entropy`: the entropy to reach, in bits.",
                      None),
                    P("tolerance", "float",
                      "For `temper_to_entropy`: how close is close enough, in "
                      "bits.",
                      0.0001),
                    P("epsilon", "float",
                      "For `mix_uniform`: the share of uniform mixed in, from 0 "
                      "to 1.",
                      None),
                    P("k", "int",
                      "For `top_k`: how many of the heaviest outcomes to keep.",
                      None),
                )),
              P("depth", "int",
                "How many slots an outcome has. Above 1, each outcome is a sequence sampled fresh every step.", 1),
              P("join", "string", "For depth > 1: the text between slots.", ""),
              P("per_slot", "list[object]",
                "For depth > 1: one target per slot, as many as `depth`. Without it every slot shares this one.",
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
                )),
              P("unit", "string",
                "For depth > 1: what a slot is — one token, or a whole outcome of any length, "
                "trained with `path` items.",
                "token", choices=("token", "item")),
              P("replace", "bool",
                "For depth > 1: whether an outcome can be drawn again in a later slot. "
                "`false` draws without replacement.",
                True),
          )),
        P("steps", "int", "Training steps.", 250),
        P("lr", "float", "Learning rate.", 1e-4),
        P("lora", "object",
          "`{\"rank\": 8, \"alpha\": 16, \"target_modules\": [\"q_proj\", "
          "\"v_proj\"]}` — the adapter's shape.",
          {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}, fields=(
              P("rank", "int", "The adapter's rank.", 8),
              P("alpha", "float", "The scaling numerator: the update is scaled by `alpha / rank`.", 16),
              P("target_modules", "list[string]", "The projections the adapter is trained on.",
                ["q_proj", "v_proj"],
                choices=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")),
          )),
        P("batch", "object",
          "Items per step by kind: depth 1 `{\"target\": 3, \"anchor\": 1, "
          "\"continuation\": 2}`, or `{\"path\": 3, \"anchor\": 1}` for the "
          "whole trie; token slots `{\"sequence\": 3, \"target\": 1, "
          "\"anchor\": 1}`; item slots `{\"path\": 3, \"anchor\": 1}`. A kind "
          "the target's shape does not build is refused.",
          None, fields=(
              P("target", "int", "Target items per step: at depth > 1, the first slot's marginal rows.", None),
              P("anchor", "int", "Anchor items per step.", None),
              P("continuation", "int", "Continuation items per step (depth 1).", None),
              P("sequence", "int", "Sampled sequences per step (depth > 1, token slots).", None),
              P("path", "int",
                "Outcomes (or, with item slots, whole lists) drawn per step and trained "
                "as soft rows at every token.", None),
          )),
        P("closer", "string",
          "The text after the outcome that closes the decision — `\" }\"` "
          "for depth 1, `'\"'` for deeper tries. It is how an outcome that "
          "begins another (\"Mystery\", \"Mystery Thriller\") ends, so item "
          "slots require one.",
          None),
        P("positions", "\"all\" | \"skip_first\" | list[int]",
          "For token slots: which slots are trained. `\"skip_first\"` leaves "
          "the first slot untrained.",
          "all"),
        P("marginal", "bool",
          "For token slots: also train the first slot's marginal distribution "
          "as its own item. Turn off with `positions: \"skip_first\"`.",
          True),
        P("naturalism", "bool | object",
          "Run the one-token-per-slot gate before training; `{\"samples\": "
          "40}` sets how many sequences it checks.",
          True, fields=(
              P("samples", "int", "How many sampled sequences the gate checks.", 40),
          )),
        P("checkpoint_every", "int",
          "Save resumable training state every this many steps.",
          50),
    ),
    example={
        "model": {"$param": "model"},
        "target": {"weights": {"$ref": {"bench": "you/lab/frequencies"}},
                   "transform": [{"op": "sqrt"}, {"op": "normalize"}]},
        "steps": 300,
        "lora": {"rank": 8, "alpha": 16},
        "seed": 7,
    },
    example_inputs={"records": {"$ref": {"bench": "you/lab/prompts"}}, "anchors": {"$ref": {"bench": "you/lab/anchors"}}},
)


def run(ctx, inputs, params):
    import os
    import tempfile

    from mechbench_compute.distill import encode, render
    from mechbench_compute.finetune import (
        batch_for,
        build_anchor_items,
        build_item_path_factory,
        build_marginal_items,
        build_path_factory,
        build_sequence_factory,
        build_target_items,
        check_enough_to_draw,
        compile_tries,
        naturalism_gate,
        resolve_slot_targets,
        train_soft_ce,
    )
    from mechbench_compute.lora import (
        apply_lora,
        fuse_adapter_stack,
        save_adapter,
    )

    model = ctx.model(params.get("model"))
    tok = model.tokenizer

    mref = params.get("model")
    if hasattr(mref, "adapter_payloads") and mref.adapter_payloads:
        fuse_adapter_stack(model.lm, list(mref.adapter_payloads))

    records = lexicon.items_of(inputs.get("records") or [])
    def rendered_of(rec):
        return render(model, rec).text

    target_spec = params.get("target")
    if not target_spec:
        raise ValueError("finetune/lora: params.target is required")
    depth = int(target_spec.get("depth", 1))
    join = str(target_spec.get("join", ""))
    unit = str(target_spec.get("unit", "token"))
    replace = bool(target_spec.get("replace", True))
    if unit not in ("token", "item"):
        raise ValueError(f"adapter/train: target.unit is 'token' or 'item', not {unit!r}")
    if depth <= 1 and (unit != "token" or not replace):
        raise ValueError(
            "adapter/train: target.unit and target.replace describe slots — "
            "set target.depth above 1")
    batch = batch_for(depth, unit, params.get("batch"))
    rendered_all = [rendered_of(r) for r in records]
    factories = {}
    marginals: list = []
    continuations: list = []
    if depth <= 1:
        from mechbench_compute.finetune import target_map_from_spec

        target = target_map_from_spec(target_spec)
        closer = params.get("closer", " }")
        tries = compile_tries(tok, target, rendered_all, closer)
        if batch.get("target", 0) or batch.get("continuation", 0):
            marginals, continuations = build_target_items(
                tok, target, rendered_all, closer=closer, tries=tries)
        if batch.get("path", 0):
            factories["path"] = build_path_factory(tries)
    elif unit == "item":
        if params.get("positions", "all") != "all" or params.get("marginal", True) is not True:
            raise ValueError(
                "adapter/train: `positions` and `marginal` shape token "
                "slots; with target.unit 'item' every path trains every "
                "position")
        slot_targets = resolve_slot_targets(target_spec, depth)
        closer = params.get("closer", '"')
        gate = params.get("naturalism", True)
        factories["path"] = build_item_path_factory(
            tok, slot_targets, rendered_all, join=join, closer=closer,
            replace=replace, gate=bool(gate))
    else:
        slot_targets = resolve_slot_targets(target_spec, depth)
        if not replace:
            check_enough_to_draw(slot_targets)
        closer = params.get("closer", '"')
        positions = params.get("positions", "all")
        gate = params.get("naturalism", True)
        if gate:
            naturalism_gate(
                tok, slot_targets, rendered_all,
                [encode(tok, r) for r in rendered_all],
                join=join, closer=closer,
                samples=int(gate.get("samples", 40))
                if isinstance(gate, dict) else 40,
                replace=replace)
        factories["sequence"] = build_sequence_factory(
            tok, slot_targets, rendered_all,
            join=join, closer=closer, positions=positions,
            replace=replace)
        marginals = (build_marginal_items(tok, slot_targets[0],
                                          rendered_all)
                     if params.get("marginal", True) else [])

    anchor_records = lexicon.items_of(inputs.get("anchors") or [])
    anchors = build_anchor_items(
        tok, [(rendered_of(r), r["answer"]) for r in anchor_records])

    lora_cfg = params.get("lora") or {}
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    target_modules = tuple(lora_cfg.get("target_modules")
                           or ("q_proj", "v_proj"))
    steps = int(params.get("steps", 250))
    lr = float(params.get("lr", 1e-4))
    seed = int(params.get("seed", 7))

    n_lora = apply_lora(model.lm, rank, alpha, targets=target_modules,
                        seed=seed)
    if ctx.on_start:
        ctx.on_start(steps)
    checkpoint_every = int(params.get("checkpoint_every", 50))
    resumed_from = int(ctx.resume_state["step"]) if ctx.resume_state else 0
    if resumed_from and ctx.on_item:
        for _ in range(resumed_from):
            ctx.on_item(None, None, True)
    final_loss = train_soft_ce(
        model.lm,
        {"target": marginals, "anchor": anchors,
         "continuation": continuations},
        batch, steps=steps, lr=lr, seed=seed,
        factories=factories,
        on_step=(lambda s, l: ctx.on_item()) if ctx.on_item else None,
        checkpoint_every=checkpoint_every if ctx.on_checkpoint else 0,
        on_checkpoint=ctx.on_checkpoint,
        resume_state=ctx.resume_state)

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        save_adapter(model.lm, path)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        os.unlink(path)

    ctx.executor._model = None
    ctx.executor._model_id = None

    base_ref = params.get("model")
    trained_on = (
        {"base": base_ref.base, "adapters": list(base_ref.adapter_labels)}
        if hasattr(base_ref, "adapter_labels")
        else {"base": base_ref, "adapters": []}
    )
    return {
        "kind": "adapter/lora",
        "format": "safetensors",
        "base_model": trained_on["base"],
        "trained_on": trained_on,
        "lora": {"rank": rank, "alpha": alpha,
                  "scale": alpha / rank,
                  "target_modules": list(target_modules),
                  "params": n_lora},
        "train": {"steps": steps, "lr": lr, "seed": seed,
                   "batch": batch, "final_loss": round(final_loss, 4),
                   "n_prompts": len(records),
                   "n_anchors": len(anchor_records),
                   "closer": closer,
                   "target": target_spec,
                   "depth": depth,
                   "unit": unit,
                   "replace": replace,
                   "positions": params.get("positions", "all"),
                   "marginal": bool(params.get("marginal", True))},
        "data": data,
    }
