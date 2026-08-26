"""Interpretability primitives as protocol operations (the
mechbench-experiments port).

The original step-XX scripts each hand-rolled a loop around the same
three moves: run the model with an intervention, read something out of
the residual stream, compare. The intervention layer (interventions.py)
already made those moves declarative; this module makes them PROTOCOL
BLOCKS, so the experiments become graphs anyone can run, re-run, and
diff on the platform:

- ``ablate_layers``     — steps 02/04/34/35 and the legacy flat kind:
                          per-layer (or per-sublayer) Δ log p sweeps.
- ``residual_vectors``  — steps 01/08/10/11/12's shared substrate:
                          residual-stream vectors at (layer, position)
                          per condition, as data other blocks consume.
- ``residual_divergence`` — the matched-pair mechanism (000050/052):
                          run a pair of prompts, cosine-compare the
                          residual streams per (layer, position).
- ``vector_similarity`` — steps 10/11/28's readout (pure, no model):
                          cosine matrix + separation metrics over
                          labeled vectors. Lives in blocks.PURE_BLOCKS.

Prompts default to RAW tokenization (no chat template): the original
experiments measure completion behavior, and a chat wrapper changes
both the positions and the task. `template: "chat"` opts in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from mechbench_compute.interventions import Ablate, Capture

#: Refuse vector payloads past this many floats — a mistyped layer list
#: must not emit a gigabyte of CBOR. ~16 MB of float64 at the cap.
MAX_VECTOR_FLOATS = 2_000_000

_COMPONENTS: dict[str, Callable[[int], Any]] = {
    "block": Ablate.layer,
    "attention": Ablate.attention,
    "mlp": Ablate.mlp,
    # MatFormer's per-layer-input gate (step 03's side channel). On a
    # non-MatFormer model the hook name does not exist and the run
    # refuses with the arch's own error.
    "gate": Ablate.side_channel,
}


def _tokenize(model, prompt: str, template: str) -> mx.array:
    return model.tokenize(prompt, chat_template=(template == "chat"))


def _last_logp(logits: mx.array) -> np.ndarray:
    last = logits[0, -1, :].astype(mx.float32)
    lp = last - mx.logsumexp(last)
    mx.eval(lp)
    return np.array(lp)


def _resolve_layers(spec: Any, n_layers: int) -> list[int]:
    """"all", an int, or a list of ints — validated against the arch."""
    if spec in (None, "all"):
        return list(range(n_layers))
    layers = [int(spec)] if isinstance(spec, int) else [int(x) for x in spec]
    bad = [i for i in layers if not 0 <= i < n_layers]
    if bad:
        raise ValueError(
            f"layers {bad} out of range for this model (n_layers={n_layers})"
        )
    return layers


def _prompt_of(record: Mapping[str, Any]) -> str:
    p = record.get("user") or record.get("prompt")
    if not isinstance(p, str) or not p:
        raise ValueError(
            f"record {record.get('id')!r} has no prompt: expected `user` "
            "(condition-set convention) or `prompt`"
        )
    return p


def _target_token_id(model, target: str) -> int:
    """The first token of `target`, tokenized raw as a continuation."""
    ids = model.tokenize(target, chat_template=False)
    flat = [int(t) for t in np.array(ids).reshape(-1)]
    # skip BOS-like specials the tokenizer prepends
    specials = set(getattr(model.tokenizer, "all_special_ids", []) or [])
    for t in flat:
        if t not in specials:
            return t
    raise ValueError(f"target {target!r} tokenized to specials only")


def ablate_layers(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Per-layer ablation sweep: for each condition, skip (or zero one
    sublayer of) each layer in turn and measure Δ log p of the target —
    the condition's `target` string's first token, or the baseline's
    top-1 when no target is named."""
    component = str(params.get("component", "block"))
    if component not in _COMPONENTS:
        raise ValueError(
            f"unknown component {component!r}: one of {sorted(_COMPONENTS)}"
        )
    template = str(params.get("template", "raw"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("ablate/layers needs at least one condition")
    if on_start:
        on_start(len(records) * (len(layers) + 1))

    intervene = _COMPONENTS[component]
    rows: list[dict[str, Any]] = []
    damage_by_layer: dict[int, list[float]] = {i: [] for i in layers}
    for record in records:
        prompt = _prompt_of(record)
        ids = _tokenize(model, prompt, template)
        base_lp = _last_logp(model.run(ids).logits)
        if on_item:
            on_item()
        target = record.get("target") or params.get("target")
        if target:
            tok = _target_token_id(model, str(target))
        else:
            tok = int(np.argmax(base_lp))
        baseline = float(base_lp[tok])
        token_str = model.tokenizer.decode([tok])
        for layer in layers:
            ids = _tokenize(model, prompt, template)
            lp = _last_logp(model.run(ids, interventions=[intervene(layer)]).logits)
            delta = float(lp[tok]) - baseline
            damage_by_layer[layer].append(delta)
            rows.append({
                "id": record.get("id"),
                "layer": layer,
                "delta_logp": round(delta, 4),
            })
            if on_item:
                on_item()
        rows.append({
            "id": record.get("id"),
            "layer": None,
            "baseline_logp": round(baseline, 4),
            "target_token": token_str,
            "target_id": tok,
        })

    return {
        "kind": "ablation_sweep",
        "component": component,
        "template": template,
        "layers": layers,
        "n_conditions": len(records),
        "rows": rows,
        "aggregates": {
            "mean_delta": [
                round(float(np.mean(damage_by_layer[i])), 4) for i in layers
            ],
            "median_delta": [
                round(float(np.median(damage_by_layer[i])), 4) for i in layers
            ],
        },
        "description": (
            f"Δ log p of the target token when each layer's {component} "
            "contribution is removed; more negative = more load-bearing."
        ),
    }


def _position_index(model, ids: mx.array, record: Mapping[str, Any],
                    position: Any) -> int:
    if isinstance(position, int):
        return position
    if position in (None, "final"):
        return int(np.array(ids).shape[-1]) - 1
    if position == "subject":
        subject = record.get("subject")
        if not isinstance(subject, str) or not subject:
            raise ValueError(
                f"record {record.get('id')!r}: position 'subject' needs a "
                "`subject` field naming a substring of the prompt"
            )
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        # Among tokens whose text appears in the subject, prefer the
        # LONGEST (ties -> latest): 'casa' must beat a stray 'a' later
        # in the sentence. The subject's own final piece carries the
        # representation (the original geometry convention).
        hits = [(len(t.strip()), i) for i, t in enumerate(tokens)
                if t.strip() and t.strip() in subject]
        if not hits:
            raise ValueError(
                f"record {record.get('id')!r}: subject {subject!r} not "
                "found among the prompt's tokens"
            )
        return max(hits)[1]
    raise ValueError(f"unknown position {position!r}")


def residual_vectors(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Residual-stream vectors at (layers × position) per condition —
    the substrate every geometry experiment reads. Labels ride along
    (`label` field, or the coords key named by params.label_coord) so
    similarity blocks can group without re-parsing ids."""
    template = str(params.get("template", "raw"))
    point = str(params.get("point", "post"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    position = params.get("position", "final")
    label_coord = params.get("label_coord")
    if not records:
        raise ValueError("residuals/vectors needs at least one condition")
    total = len(records) * len(layers) * model.arch.d_model
    if total > MAX_VECTOR_FLOATS:
        raise ValueError(
            f"{len(records)} conditions × {len(layers)} layers × "
            f"d_model {model.arch.d_model} = {total} floats exceeds the "
            f"{MAX_VECTOR_FLOATS} cap — capture fewer layers or conditions"
        )
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point=point)
    rows: list[dict[str, Any]] = []
    for record in records:
        ids = _tokenize(model, _prompt_of(record), template)
        result = model.run(ids, interventions=[cap])
        pos = _position_index(model, ids, record, position)
        label = record.get("label")
        if label is None and label_coord:
            label = (record.get("coords") or {}).get(label_coord)
        for layer in layers:
            v = result.cache[f"blocks.{layer}.resid_{point}"][0, pos, :]
            v = v.astype(mx.float32)
            mx.eval(v)
            rows.append({
                "id": record.get("id"),
                "label": label,
                "layer": layer,
                "vector": [round(float(x), 5) for x in np.array(v)],
            })
        if on_item:
            on_item()
    return {
        "kind": "residual_vectors",
        "point": point,
        "position": str(position),
        "layers": layers,
        "d_model": model.arch.d_model,
        "template": template,
        "rows": rows,
    }


def residual_divergence(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Matched-pair divergence: run prompts `a` and `b`, and per
    (layer, position) report 1 − cosine of the residual streams. The
    map shows WHERE a one-token change ripples (000050/000052).

    Pairs must tokenize to equal lengths — that is what 'matched'
    means; unequal pairs are reported as errors, not silently aligned.
    """
    template = str(params.get("template", "raw"))
    point = str(params.get("point", "post"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("residuals/divergence needs at least one pair")
    if on_start:
        on_start(len(records) * 2)

    cap = Capture.residual(layers, point=point)
    pairs: list[dict[str, Any]] = []
    for record in records:
        a, b = record.get("a"), record.get("b")
        if not (isinstance(a, str) and isinstance(b, str) and a and b):
            raise ValueError(
                f"pair {record.get('id')!r} needs prompt fields `a` and `b`"
            )
        ids_a = _tokenize(model, a, template)
        ids_b = _tokenize(model, b, template)
        len_a = int(np.array(ids_a).shape[-1])
        len_b = int(np.array(ids_b).shape[-1])
        if len_a != len_b:
            pairs.append({
                "id": record.get("id"),
                "error": (
                    f"prompts tokenize to different lengths ({len_a} vs "
                    f"{len_b}) — a matched pair must match"
                ),
            })
            if on_item:
                on_item()
                on_item()
            continue
        run_a = model.run(ids_a, interventions=[cap])
        if on_item:
            on_item()
        run_b = model.run(ids_b, interventions=[cap])
        if on_item:
            on_item()
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids_a).reshape(-1)]
        matrix: list[list[float]] = []
        for layer in layers:
            va = np.array(run_a.cache[f"blocks.{layer}.resid_{point}"][0]
                          .astype(mx.float32))
            vb = np.array(run_b.cache[f"blocks.{layer}.resid_{point}"][0]
                          .astype(mx.float32))
            na = np.linalg.norm(va, axis=-1)
            nb = np.linalg.norm(vb, axis=-1)
            cos = (va * vb).sum(axis=-1) / np.maximum(na * nb, 1e-9)
            matrix.append([round(float(1.0 - c), 5) for c in cos])
        pairs.append({
            "id": record.get("id"),
            "tokens": tokens,
            "divergence": matrix,  # [layer][position], 1 - cosine
        })
    return {
        "kind": "divergence_map",
        "point": point,
        "layers": layers,
        "template": template,
        "pairs": pairs,
        "description": (
            "1 − cosine similarity of the two residual streams per "
            "(layer, position). 0 = identical; the map shows where a "
            "one-token change ripples."
        ),
    }


def lens_positions(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Step 08 as a block: project every layer's residual through the
    unembedding at every position and follow one target token — where
    in the sequence, and at what depth, does the answer become
    visible? Rank 0 means the target is that position's top readout."""
    from mechbench_compute import lens

    template = str(params.get("template", "raw"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("lens/positions needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.residual(layers, point="post")
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = _prompt_of(record)
        ids = _tokenize(model, prompt, template)
        result = model.run(ids, interventions=[cap])
        target = record.get("target") or params.get("target")
        if target:
            tok = _target_token_id(model, str(target))
        else:
            tok = int(np.argmax(_last_logp(result.logits)))
        ranks, logprobs = lens.logit_lens_per_position(
            model, result.cache, tok, layers=layers)
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        rows.append({
            "id": record.get("id"),
            "tokens": tokens,
            "target_token": model.tokenizer.decode([tok]),
            "target_id": tok,
            "logprob": [[round(float(x), 4) for x in r] for r in logprobs],
            "rank": [[int(x) for x in r] for r in ranks],
        })
        if on_item:
            on_item()
    return {
        "kind": "lens_map",
        "layers": layers,
        "template": template,
        "rows": rows,
        "description": (
            "Logit-lens readout of the target token at every (layer, "
            "position): log p and rank of the target when each layer's "
            "residual is projected straight through the unembedding."
        ),
    }


def patch_trace(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Causal tracing (step 09): run CLEAN capturing every layer, run
    CORRUPT for the baseline, then patch the clean residual into the
    corrupt run one (layer, position) at a time and measure how much of
    the clean answer's probability comes back. The map localizes WHERE
    the fact lives. Progress ticks per layer row (a full row of
    positions is one unit)."""
    from mechbench_compute.interventions import Patch

    template = str(params.get("template", "raw"))
    point = str(params.get("point", "resid_post"))
    metric = str(params.get("metric", "logprob"))
    if metric not in ("logprob", "prob"):
        raise ValueError(f"unknown metric {metric!r}: 'logprob' or 'prob'")
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if not records:
        raise ValueError("patch/trace needs at least one pair")
    if on_start:
        on_start(len(records) * len(layers))

    cap = Capture.residual(layers, point=point.removeprefix("resid_"))
    pairs: list[dict[str, Any]] = []
    for record in records:
        clean, corrupt = record.get("clean"), record.get("corrupt")
        if not (isinstance(clean, str) and isinstance(corrupt, str)
                and clean and corrupt):
            raise ValueError(
                f"pair {record.get('id')!r} needs prompt fields "
                "`clean` and `corrupt`"
            )
        ids_clean = _tokenize(model, clean, template)
        ids_corrupt = _tokenize(model, corrupt, template)
        n_clean = int(np.array(ids_clean).shape[-1])
        n_corrupt = int(np.array(ids_corrupt).shape[-1])
        if n_clean != n_corrupt:
            pairs.append({
                "id": record.get("id"),
                "error": (
                    f"prompts tokenize to different lengths ({n_clean} vs "
                    f"{n_corrupt}) — positions cannot align under patching"
                ),
            })
            if on_item:
                for _ in layers:
                    on_item()
            continue
        clean_run = model.run(ids_clean, interventions=[cap])
        clean_lp = _last_logp(clean_run.logits)
        target = record.get("target") or params.get("target")
        tok = (_target_token_id(model, str(target)) if target
               else int(np.argmax(clean_lp)))
        # 'prob' only registers when the clean prompt puts real mass on
        # the target (the original step 09 used the clean top-1, which
        # guarantees it); 'logprob' registers recovery at ANY mass —
        # explicit rare targets measured exactly nothing in prob space
        # on the first prod run.
        def read(lp: np.ndarray, tok: int = tok) -> float:
            return (float(np.exp(lp[tok])) if metric == "prob"
                    else float(lp[tok]))

        p_clean_in_clean = read(clean_lp)
        corrupt_lp = _last_logp(model.run(ids_corrupt).logits)
        baseline = read(corrupt_lp)

        recovery: list[list[float]] = []
        seq = n_corrupt
        for layer in layers:
            row: list[float] = []
            for pos in range(seq):
                patch = Patch.position(
                    layer=layer, position=pos, source=clean_run.cache,
                    point=point)
                lp = _last_logp(
                    model.run(ids_corrupt, interventions=[patch]).logits)
                row.append(round(read(lp) - baseline, 5))
            recovery.append(row)
            if on_item:
                on_item()
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids_corrupt).reshape(-1)]
        pairs.append({
            "id": record.get("id"),
            "tokens": tokens,
            "target_token": model.tokenizer.decode([tok]),
            "metric": metric,
            "p_target_clean": round(p_clean_in_clean, 5),
            "p_target_corrupt": round(baseline, 5),
            "recovery": recovery,  # [layer][pos]: Δ(metric) of the target
        })
    return {
        "kind": "patch_trace",
        "point": point,
        "metric": metric,
        "layers": layers,
        "template": template,
        "pairs": pairs,
        "description": (
            "Activation patching: p(clean answer) recovered when the "
            "clean residual is patched into the corrupt run at each "
            "(layer, position). Bright cells are where the fact lives."
        ),
    }


#: Attention matrices are quadratic in sequence length; refuse a
#: capture that would emit more than this many floats.
MAX_ATTN_FLOATS = 2_000_000


def attention_patterns(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steps 05/06: post-softmax attention weights per head at chosen
    layers. Layers must be named explicitly — every layer of every
    head of a long prompt is a picture nobody asked for."""
    template = str(params.get("template", "raw"))
    spec = params.get("layers")
    if spec in (None, "all"):
        raise ValueError(
            "attention/patterns needs an explicit layers list — "
            "attention weights are per-head and quadratic in prompt "
            "length, so name the layers you want to look at"
        )
    layers = _resolve_layers(spec, model.arch.n_layers)
    if not records:
        raise ValueError("attention/patterns needs at least one condition")
    if on_start:
        on_start(len(records))

    cap = Capture.attn_weights(layers)
    rows: list[dict[str, Any]] = []
    total_floats = 0
    for record in records:
        ids = _tokenize(model, _prompt_of(record), template)
        result = model.run(ids, interventions=[cap])
        tokens = [model.tokenizer.decode([int(t)])
                  for t in np.array(ids).reshape(-1)]
        seq = len(tokens)
        total_floats += len(layers) * model.arch.n_heads * seq * seq
        if total_floats > MAX_ATTN_FLOATS:
            raise ValueError(
                f"attention capture would exceed {MAX_ATTN_FLOATS} floats "
                "— fewer layers, shorter prompts, or fewer conditions"
            )
        per_layer = []
        for layer in layers:
            w = result.cache[f"blocks.{layer}.attn.weights"]
            arr = np.array(w.astype(mx.float32))[0]  # [heads, L, S]
            per_layer.append({
                "layer": layer,
                "heads": [[[round(float(x), 4) for x in r] for r in h]
                          for h in arr],
            })
        rows.append({"id": record.get("id"), "tokens": tokens,
                     "layers": per_layer})
        if on_item:
            on_item()
    return {
        "kind": "attention_patterns",
        "n_heads": model.arch.n_heads,
        "layers": layers,
        "template": template,
        "rows": rows,
        "description": (
            "Post-softmax attention weights per head: row = the "
            "attending position, column = the attended-to position."
        ),
    }


def ablate_heads(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Step 07: zero one head at a time across (layers × heads) and
    measure Δ log p of the target — the head-level version of the
    layer sweep. Progress ticks per (condition, layer)."""
    template = str(params.get("template", "raw"))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    n_heads = model.arch.n_heads
    if not records:
        raise ValueError("ablate/heads needs at least one condition")
    if on_start:
        on_start(len(records) * (len(layers) + 1))

    sums = np.zeros((len(layers), n_heads), dtype=np.float64)
    metas: list[dict[str, Any]] = []
    for record in records:
        prompt = _prompt_of(record)
        ids = _tokenize(model, prompt, template)
        base_lp = _last_logp(model.run(ids).logits)
        if on_item:
            on_item()
        target = record.get("target") or params.get("target")
        tok = (_target_token_id(model, str(target)) if target
               else int(np.argmax(base_lp)))
        baseline = float(base_lp[tok])
        metas.append({
            "id": record.get("id"),
            "target_token": model.tokenizer.decode([tok]),
            "baseline_logp": round(baseline, 4),
        })
        for li, layer in enumerate(layers):
            for head in range(n_heads):
                lp = _last_logp(model.run(
                    ids, interventions=[Ablate.head(layer, head)]).logits)
                sums[li, head] += float(lp[tok]) - baseline
            if on_item:
                on_item()
    mean = sums / len(records)
    return {
        "kind": "head_ablation",
        "layers": layers,
        "n_heads": n_heads,
        "n_conditions": len(records),
        "conditions": metas,
        "mean_delta": [[round(float(x), 4) for x in row] for row in mean],
        "template": template,
        "description": (
            "Mean Δ log p of the target with each single head zeroed — "
            "rows are layers, columns are heads; dark cells are heads "
            "the answer runs through."
        ),
    }


def logit_attribution(
    model,
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    on_item: Callable[[], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Steps 32/33 as a block: direct logit attribution per layer.

    One forward per condition captures every layer's residual plus the
    final norm's scale; the residual stream is decomposed into exactly
    additive components (embedding, then each layer's delta), and each
    component's contribution to the target logit is read through the
    norm-folded unembed (apply_ln, task 000142).

    SELF-VALIDATING: every row reports its additivity residual — the
    summed contributions minus the model's true final logit. A reader
    never has to take the decomposition on faith.
    """
    from mechbench_compute import attribution

    template = str(params.get("template", "raw"))
    apply_ln = bool(params.get("apply_ln", True))
    layers = _resolve_layers(params.get("layers"), model.arch.n_layers)
    if layers != list(range(model.arch.n_layers)):
        raise ValueError(
            "attribution/logits decomposes the WHOLE stream — additivity "
            "only holds over all layers, so `layers` must be \"all\""
        )
    if not records:
        raise ValueError("attribution/logits needs at least one condition")
    if on_start:
        on_start(len(records))

    from mechbench_compute.interventions import Capture as Cap

    interventions = [
        Cap.residual(layers, point="post"),
        Cap.residual([0], point="pre"),
        Cap.final_norm_scale(),
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        ids = _tokenize(model, _prompt_of(record), template)
        result = model.run(ids, interventions=interventions)
        lp = _last_logp(result.logits)
        target = record.get("target") or params.get("target")
        tok = (_target_token_id(model, str(target)) if target
               else int(np.argmax(lp)))
        contrast = record.get("contrast")
        ctok = _target_token_id(model, str(contrast)) if contrast else None

        acc = attribution.accumulated_resid(result.cache, include_pre=True)
        components = np.diff(acc, axis=0, prepend=np.zeros_like(acc[:1]))
        # components[0] = embedding stream, components[i] = layer i-1's delta
        ln_scale = np.array(
            mx.array(result.cache["final_norm.scale"]).astype(mx.float32)
        ).reshape(-1)
        targets = [tok] if ctok is None else [tok, ctok]
        attrs = attribution.logit_attrs(
            model, components, targets,
            apply_ln=apply_ln, ln_scale=ln_scale)
        contrib = attrs[:, 0] if ctok is None else attrs[:, 0] - attrs[:, 1]

        # The honesty number: does the decomposition sum to the truth?
        # The comparison lives in PRE-softcap space — the decomposition
        # is linear and the cap is not, so a capped "true" logit would
        # disagree structurally (gemma4 caps; e2b's first run showed it).
        last = result.logits[0, -1, :].astype(mx.float32)
        mx.eval(last)
        last_np = np.array(last, dtype=np.float64)
        cap = getattr(
            getattr(getattr(model, "_model", None), "language_model", None),
            "final_logit_softcapping", None)
        if cap:
            c = float(cap)
            last_np = c * np.arctanh(np.clip(last_np / c, -0.999999, 0.999999))
        true_logit = float(last_np[tok])
        if ctok is not None:
            true_logit -= float(last_np[ctok])
        summed = float(attrs.sum(axis=0)[0]) if ctok is None else float(
            (attrs[:, 0] - attrs[:, 1]).sum())
        rows.append({
            "id": record.get("id"),
            "target_token": model.tokenizer.decode([tok]),
            "contrast_token": (model.tokenizer.decode([ctok])
                               if ctok is not None else None),
            "contributions": [round(float(x), 4) for x in contrib],
            "additivity": {
                "summed": round(summed, 3),
                "true_logit": round(true_logit, 3),
                "residual": round(summed - true_logit, 3),
            },
        })
        if on_item:
            on_item()
    return {
        "kind": "logit_attribution",
        "apply_ln": apply_ln,
        "layers": layers,
        "template": template,
        "components": ["embed", *[f"L{i}" for i in layers]],
        "rows": rows,
        "description": (
            "Direct logit attribution: each component's contribution to "
            "the target logit (embedding first, then every layer's "
            "delta), norm-folded so the bars sum to the model's true "
            "final logit — each row carries its own additivity residual."
        ),
    }


def vector_similarity(inputs: Mapping[str, Any],
                      params: Mapping[str, Any]) -> dict[str, Any]:
    """Pure readout over a residual_vectors record: per layer, the
    cosine matrix plus the separation metrics the original geometry
    steps reported (intra/inter cosine, nearest-neighbor purity,
    silhouette when labels exist)."""
    from mechbench_compute import geometry

    src = inputs.get("vectors") or params.get("vectors")
    if not isinstance(src, Mapping) or src.get("kind") != "residual_vectors":
        raise ValueError(
            "vectors/similarity needs a residual_vectors record on its "
            "`vectors` port"
        )
    rows = src.get("rows") or []
    layers = src.get("layers") or sorted({r.get("layer") for r in rows})
    out_layers: list[dict[str, Any]] = []
    for layer in layers:
        layer_rows = [r for r in rows if r.get("layer") == layer]
        if not layer_rows:
            continue
        ids = [r.get("id") for r in layer_rows]
        labels = [r.get("label") for r in layer_rows]
        vectors = np.array([r["vector"] for r in layer_rows], dtype=np.float32)
        matrix = geometry.cosine_matrix(vectors)
        entry: dict[str, Any] = {
            "layer": layer,
            "ids": ids,
            "labels": labels,
            "matrix": [[round(float(x), 4) for x in row] for row in matrix],
        }
        labeled = all(lb is not None for lb in labels) and len(set(labels)) > 1
        if labeled:
            intra, inter, gap = geometry.intra_inter_separation(vectors, labels)
            purity, _hits = geometry.nearest_neighbor_purity(vectors, labels)
            entry["separation"] = {
                "intra_cosine": round(float(intra), 4),
                "inter_cosine": round(float(inter), 4),
                "gap": round(float(gap), 4),
            }
            entry["nn_purity"] = round(float(purity), 4)
            import contextlib

            # sklearn absent, or a degenerate labeling: the silhouette
            # is optional garnish, never worth failing the readout.
            with contextlib.suppress(Exception):
                entry["silhouette"] = round(
                    float(geometry.silhouette_cosine(vectors, labels)), 4)
        out_layers.append(entry)
    return {
        "kind": "similarity_matrix",
        "position": src.get("position"),
        "point": src.get("point"),
        "layers": out_layers,
        "description": (
            "Pairwise cosine similarity of residual vectors per layer, "
            "with label-separation metrics where labels exist."
        ),
    }
