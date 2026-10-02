from __future__ import annotations

from mechbench_compute import lexicon
from mechbench_compute.lexicon._base import (
    DEFAULT_OUTPUT,
    In,
    Op,
    Otherwise,
    Output,
    P,
    Resume,
)

OP = Op(
    name="adapter/train",
    needs=frozenset({"model.backward", "model.forward"}),
    resume=Resume("state-restorable", items=True),
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

With `keep_checkpoints` the run also keeps the adapter as it stood every
`checkpoint_every` steps and at the last, and emits them on its
`checkpoints` output: a collection of `adapter/lora`, one whole adapter
per kept step, `coords.step` its step and `loss` the training loss
there. The final adapter on the default output is unchanged, and the
last item's weights are its weights byte for byte. Training time becomes
a coordinate: `records/map` over the collection, with the item fed to a
model's `adapter` port, sweeps any operation over steps, and
`adapter/measure` reads the whole collection at once. A run interrupted
after a kept step resumes with the items it already kept, not
recomputed. The collection is one stored object, so the adapters' bytes
times their count must fit the platform's 64 MiB object limit; the run
refuses before training when it would not.

`seed` fixes the whole run: the adapter's initial weights as well as the
sampling order, so two runs of the same node on the same machine produce
a byte-identical adapter. Change the seed to see the spread a different
draw gives.

### Training an operator instead

With `operator` the node trains a function of what a mask selects on the
residual stream after the named layers (`resid_post`) instead of a LoRA.
Its parameters are the only ones trained; the base model is frozen and
comes back bit for bit. It trains toward the same target with the same
loop, at one position: `target` and `anchor` items, the rows the decision
writes. `function` is its form, and every form starts as the identity:
`affine` is a·x + b per masked coordinate, or, on a learned subspace
(`mask: {"rank": r}`), ReFT's h + Rᵀ(W·h + b − R·h) with R's rows kept
orthonormal; `polynomial` is a polynomial of `degree` per coordinate;
`mlp` is x plus one hidden layer of `width` (ReLU) on the masked slice.
`mask` is a list of dimensions, `{"rank": r}`, or absent for every
coordinate. `positions: "last"` acts at the last position of every
forward pass (the decision, then each token as it is generated) and
`"all"` everywhere; `gate: true` adds one logistic unit on the residual
whose output scales the edit at every position, so where it acts is
learned, not given. `penalty: {"l1": λ}` pulls each parameter toward the
identity with a proximal step after every update, by λ·lr/(√v + ε) with v
Adam's running mean of its squared gradient, so a value the target does
not need lands on the identity exactly; `effective` counts the rest. Operators
want a larger `lr` than a LoRA (0.01 to 0.1), and a polynomial, whose high
powers of raw activations are steep, a smaller one.

The output is then an `adapter/operator`: its `parameters` by layer,
`operator` (its form, `d`, `params` and `effective`) and `train`. A model
reference carries it beside LoRA adapters, and every node that loads the
model attaches it; `layers` does not apply to it, and it cannot be merged
into a checkpoint. `depth` above 1, `continuation` and `path` items and
`keep_checkpoints` are refused with an operator.
""",
    inputs=(
        In("records", "records/record",
           "The training prompts: chat-shaped records with `user`, and "
           "optionally `system` and `prefill`.", many=True),
        In("anchors", "records/record",
           "Prompt records with a known `answer`, mixed into each batch.",
           many=True, required=False),
    ),
    output=Output('adapter/lora', collection=False, doc="`data` (safetensors bytes), `format`, `base_model`, `trained_on` (the base and any prior adapters), `lora` (rank, alpha, scale, target modules, parameter count) and `train` (steps, lr, seed, batch, final loss, the target spec, depth, unit, replace, positions, counts). Wire it into a later node's `adapter` port, or `adapter/publish`. With `operator`, an `adapter/operator` instead: `parameters` by layer, `operator` (the form, `d`, `params`, `effective`), `base_model`, `trained_on` and `train`; a model reference carries it.",
                  otherwise=tuple(Otherwise("adapter/operator", param="operator.function", equals=f)
                                  for f in ("affine", "polynomial", "mlp"))),
    outputs={"checkpoints": Output(
        "adapter/lora", collection=True,
        doc="With `keep_checkpoints`: one item per kept step — every `checkpoint_every` steps and the last — "
            "each a whole adapter with `id` `step-<n>`, `coords.step` (an integer), `loss` (the training loss at "
            "that step) and the final adapter's `format`, `base_model`, `trained_on`, `lora` and `train` (without "
            "`final_loss`). The header carries `base_model`, `trained_on`, `lora`, `train` as the final adapter "
            "has them, and `checkpoint_every`. Read it with `{\"node\": …, \"output\": \"checkpoints\"}`; "
            "without `keep_checkpoints` there is none, and an edge from it fails.")},
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
        P("operator", "object",
          "Train an operator instead of a LoRA — see *Training an operator instead*. Its output "
          "is an `adapter/operator`.", None, fields=(
              P("layers", "int | list[int]", "The layers it acts after, each with its own parameters."),
              P("point", "string", "Where it acts: `resid_post`, the residual stream after the layer, "
                "the one point an operator takes.", "resid_post"),
              P("positions", "string", "`last`, the last position of every forward pass, or `all`.", "last",
                choices=("last", "all")),
              P("gate", "bool", "Learn where it acts with one logistic unit on the residual; its "
                "positions are then `all`.", False),
              P("mask", "list[int] | object", "A list of dimensions, `{\"rank\": r}` for a learned "
                "subspace, or absent for every coordinate.", None,
                fields=(P("rank", "int", "The learned subspace's rank."),)),
              P("function", "string", "The form it is fitted in.", choices=("affine", "polynomial", "mlp")),
              P("degree", "int", "For `polynomial`: its degree.", 2),
              P("width", "int", "For `mlp`: the hidden layer's width.", 16),
              P("penalty", "object", "Pull the parameters toward the identity.", None,
                fields=(P("l1", "float", "λ, the weight of the L1 pull."),)),
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
          "Save resumable training state every this many steps; with "
          "`keep_checkpoints`, also the cadence the adapters are kept at.",
          50),
        P("keep_checkpoints", "bool",
          "Also emit the adapter as it stood every `checkpoint_every` steps "
          "and at the last, as the `checkpoints` output: a collection of "
          "adapters over `coords.step`.",
          False),
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
        read_adapter_bytes,
    )

    model = ctx.model(params.get("model"))
    tok = model.tokenizer

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
    operator = params.get("operator")
    batch = (read_operator_batch(params, depth) if operator is not None
             else batch_for(depth, unit, params.get("batch")))
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

    base_ref = params.get("model")
    trained_on = (
        {"base": base_ref.base,
         "adapters": [label if layers is None else {"bench": label, "layers": list(layers)}
                      for label, layers in zip(base_ref.adapter_labels, base_ref.adapter_layers)]}
        if hasattr(base_ref, "adapter_labels")
        else {"base": base_ref, "adapters": []}
    )
    methods = {"steps": steps, "lr": lr, "seed": seed, "batch": batch,
               "n_prompts": len(records), "n_anchors": len(anchor_records),
               "closer": closer, "target": target_spec, "depth": depth,
               "unit": unit, "replace": replace,
               "positions": params.get("positions", "all"),
               "marginal": bool(params.get("marginal", True))}
    if operator is not None:
        from mechbench_compute.adapters.train_operator import train_operator

        return {DEFAULT_OUTPUT: train_operator(
            ctx, model, operator, {"target": marginals, "anchor": anchors}, batch,
            trained_on=trained_on, methods=methods,
            checkpoint_every=int(params.get("checkpoint_every", 50)))}

    n_lora = apply_lora(model.lm, rank, alpha, targets=target_modules,
                        seed=seed, keys=model.architecture.adapter_keys)
    lora = {"rank": rank, "alpha": alpha, "scale": alpha / rank,
            "target_modules": list(target_modules), "params": n_lora}
    lineage = {"kind": "adapter/lora", "format": "safetensors",
               "base_model": trained_on["base"], "trained_on": trained_on,
               "lora": lora}

    checkpoint_every = int(params.get("checkpoint_every", 50))
    kept_steps = read_kept_steps(params, steps, checkpoint_every)
    if kept_steps:
        check_kept_size(len(read_adapter_bytes(model.lm)), len(kept_steps))

    def keep(step, loss, data):
        return {"id": f"step-{step}", "coords": {"step": step}, **lineage,
                "train": methods, "loss": float(loss), "data": data}

    if ctx.on_start:
        ctx.on_start(steps)
    resumed_from = int(ctx.resume_state["step"]) if ctx.resume_state else 0
    kept = restore_kept(kept_steps, resumed_from, ctx.resume_items)
    if resumed_from and ctx.on_item:
        for step in range(1, resumed_from + 1):
            if step in kept:
                ctx.on_item(kept[step]["id"], kept[step], True)
            else:
                ctx.on_item(None, None, True)

    def on_step(step, loss):
        if step in kept_steps and step < steps:
            kept[step] = keep(step, loss, read_adapter_bytes(model.lm))
            if ctx.on_item:
                ctx.on_item(kept[step]["id"], kept[step], False)
        elif ctx.on_item:
            ctx.on_item()

    final_loss = train_soft_ce(
        model.lm,
        {"target": marginals, "anchor": anchors,
         "continuation": continuations},
        batch, steps=steps, lr=lr, seed=seed,
        factories=factories,
        on_step=on_step if (ctx.on_item or kept_steps) else None,
        checkpoint_every=checkpoint_every if ctx.on_checkpoint else 0,
        on_checkpoint=ctx.on_checkpoint,
        resume_state=ctx.resume_state)

    data = read_adapter_bytes(model.lm)
    ctx.evict_model()

    adapter = {**lineage,
               "train": {**methods, "final_loss": round(final_loss, 4)},
               "data": data}
    if not kept_steps:
        return {DEFAULT_OUTPUT: adapter}
    kept[steps] = keep(steps, final_loss, data)
    return {DEFAULT_OUTPUT: adapter,
            "checkpoints": lexicon.collection(
                "adapter/lora", [kept[s] for s in kept_steps],
                base_model=adapter["base_model"], trained_on=trained_on,
                lora=lora, train=adapter["train"],
                checkpoint_every=checkpoint_every)}


def read_operator_batch(params, depth: int) -> dict[str, int]:
    for name, why in (("lora", "trains a LoRA, and `operator` an operator: give one"),
                      ("keep_checkpoints", "keeps LoRA adapters; an operator's run keeps none")):
        if params.get(name):
            raise ValueError(f"adapter/train: `{name}` {why}")
    if depth > 1:
        raise ValueError("adapter/train: an operator trains at one position; set target.depth to 1")
    batch = dict(params.get("batch") or {"target": 3, "anchor": 1})
    other = sorted(k for k, n in batch.items() if int(n) > 0 and k not in ("target", "anchor"))
    if other:
        raise ValueError(f"adapter/train: an operator trains at one position, on `target` and "
                         f"`anchor` items; not {other}")
    return batch


def read_kept_steps(params, steps: int, every: int) -> list[int]:
    if not params.get("keep_checkpoints", False):
        return []
    if every <= 0:
        raise ValueError(
            "adapter/train: keep_checkpoints keeps an adapter every "
            "`checkpoint_every` steps; set it above 0")
    return [*range(every, steps, every), steps]


def check_kept_size(adapter_bytes: int, count: int) -> None:
    from mechbench_compute.bench import MAX_OBJECT_BYTES

    if adapter_bytes * count > MAX_OBJECT_BYTES:
        raise ValueError(
            f"adapter/train: {count} kept checkpoints of {adapter_bytes:,} bytes "
            f"each are over the {MAX_OBJECT_BYTES:,}-byte object limit the "
            f"collection is stored under; keep fewer with a larger "
            f"`checkpoint_every` (at most {max(1, MAX_OBJECT_BYTES // adapter_bytes)} fit)")


def restore_kept(kept_steps: list[int], resumed_from: int,
                 resume_items) -> dict[int, dict]:
    kept: dict[int, dict] = {}
    for step in (s for s in kept_steps if s <= resumed_from):
        item = (resume_items or {}).get(f"step-{step}")
        if item is None:
            raise ValueError(
                f"adapter/train: resuming at step {resumed_from}, but the "
                f"adapter kept at step {step} is not among the run's kept "
                f"items; restart the node")
        kept[step] = item
    return kept
