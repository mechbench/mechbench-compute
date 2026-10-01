from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mechbench_compute.api import (
    In,
    Op,
    Output,
    P,
    Resume,
    collection,
    encode_features,
    items_of,
    read_dictionary_activations,
    read_dictionary_weights,
    read_header,
    render,
)

POSITIONS = ("all", "last")

MAX_ROWS = 2_000_000

OP = Op(
    name="dictionary/encode",
    needs=frozenset({"model.forward", "executor.sub"}),
    resume=Resume("restart"),
    summary=(
        "Read a model's activations on records through a sparse dictionary: which features fire at "
        "each position and how strongly, and how much of the activations the dictionary reconstructs."
    ),
    description="""\
One forward pass per record. At the point and layer the dictionary reads,
every position's activation is encoded — `b_enc` plus the activation
times the encoder, through the dictionary's nonlinearity (a JumpReLU
keeps a feature only above its own threshold) — and each feature that
fired becomes one item: the record, the position, its token, the
feature and its value.

The header says how well the dictionary reconstructs these activations,
measured on them and not taken from where it was published: the
decoder's reconstruction (the features times their decoder rows, plus
`b_dec`) is compared with the activation at every position read.
`variance_explained` is one minus the summed squared error over the
activations' summed squared distance from their mean; `l0` is the mean
number of features firing per position; `inactive` counts the features
that never fired. A dictionary trained on another model, or on this
one before an adapter changed it, can reconstruct poorly here, and then
its features describe the reconstruction rather than the model.

With an adapter on `adapted`, the records are read twice: once on the
node's model (`coords.model` `base`) and once with the adapter fused on
top (`adapted`), and the header carries both fidelities and the drop
between them, so a feature's change under the adapter is read beside
how far the dictionary can be trusted on each side.

The first position is left out when it is the beginning-of-sequence
token, whose activation is an outlier dictionaries are not trained on
(`skip_bos`). `features` keeps only the listed features' items; the
fidelity is over every feature either way.
""",
    inputs=(
        In("records", "records/record",
           "The prompts (`user`, `prompt` or `text`), each optionally with `coords`. A model reads each "
           "record as [its kind](/kinds/records/record/) says: `user` in the chat template, `prompt` and "
           "`text` raw.", many=True),
        In("dictionary", "direction/dictionary",
           "The dictionary to read through, from `dictionary/load`."),
        In("adapter", "adapter/lora",
           "A LoRA adapter to fuse on top of the model for this node only. Fuses last, on top of any "
           "adapters the model reference itself carries; `adapter_scale` scales this one.",
           required=False),
        In("adapted", "adapter/lora",
           "An adapter to read the records under as well, fused on top of the node's model for the "
           "second reading only. The items then carry `coords.model` `base` or `adapted`, and the header "
           "both fidelities.",
           required=False),
    ),
    output=Output("activations/feature", collection=True,
                  doc="One item per (record, position, feature) that fired: `{id, coords, position, token, "
                      "feature, value}`, `coords.model` naming the reading. The header carries `dictionary`, "
                      "`model`, `point`, `positions`, `features` when given, and `fidelity` "
                      "(`variance_explained`, `l0`, `mse`, `positions`, `inactive`, `published_l0`); with an "
                      "adapter on `adapted`, also `adapted_fidelity` and `fidelity_drop`."),
    params=(
        P("position", "string",
          "Which positions to read: every position, or only the last.",
          "all", choices=POSITIONS),
        P("skip_bos", "bool",
          "Leave out the first position when it is the beginning-of-sequence token.",
          True),
        P("features", "list[int]",
          "Keep only these features' items. The fidelity is over every feature either way.",
          None),
    ),
    example={"model": {"$param": "model"}, "position": "all"},
    example_inputs={"records": {"$ref": {"bench": "you/lab/stories"}},
                    "dictionary": {"$ref": {"bench": "you/lab/gemma-scope-resid-17"}}},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    records = items_of(inputs.get("records") or [])
    readout = encode_records(model, records, inputs["dictionary"], params,
                             on_item=ctx.on_item, on_start=ctx.on_start)
    if inputs.get("adapted") is None:
        return readout
    second = ctx.sub("dictionary/encode",
                     {"records": inputs["records"], "dictionary": inputs["dictionary"],
                      "adapter": inputs["adapted"]},
                     params)
    return join_readings(readout, second)


def read_model_id(model: Any) -> str | None:
    for attr in ("requested_ref", "requested", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    v = getattr(getattr(model, "arch", None), "model_id", None)
    return v if isinstance(v, str) and v else None


def encode_records(model: Any, records: Sequence[Mapping[str, Any]], dictionary: Any,
                   params: Mapping[str, Any], *, reading: str = "base",
                   on_item: Callable[[], None] | None = None,
                   on_start: Callable[[int], None] | None = None) -> dict[str, Any]:
    weights = read_dictionary_weights(dictionary)
    space = weights["space"]
    point, layer = str(space["point"]), int(space["layer"])
    position = str(params.get("position") or "all")
    if position not in POSITIONS:
        raise ValueError(f"dictionary/encode: position {position!r} is not one of {', '.join(POSITIONS)}")
    skip_bos = bool(params.get("skip_bos", True))
    keep = params.get("features")
    keep_set = None if keep is None else {int(i) for i in keep}
    n_layers = int(model.arch.n_layers)
    if not 0 <= layer < n_layers:
        raise ValueError(f"dictionary/encode: the dictionary reads layer {layer}; the model has {n_layers}")
    if not records:
        raise ValueError("dictionary/encode needs at least one record")
    tokenizer = model.tokenizer
    bos = getattr(tokenizer, "bos_token_id", None)
    width, d_in = weights["w_enc"].shape[1], weights["w_enc"].shape[0]
    if on_start:
        on_start(len(records))
    rows: list[dict[str, Any]] = []
    sse = total_sq = 0.0
    sums = np.zeros(d_in, dtype=np.float64)
    n = active = 0
    fired = np.zeros(width, dtype=bool)
    for record in records:
        rendered = render(model, record)
        acts = read_dictionary_activations(model, rendered.array, point, layer)
        if acts.shape[1] != d_in:
            raise ValueError(f"dictionary/encode: the dictionary reads {d_in}-wide activations, and "
                             f"{point} at layer {layer} of this model is {acts.shape[1]} wide")
        at = list(range(len(rendered.ids))) if position == "all" else [len(rendered.ids) - 1]
        if skip_bos and bos is not None:
            at = [p for p in at if not (p == 0 and rendered.ids[0] == bos)]
        x = acts[at]
        f = encode_features(x, weights)
        recon = f @ weights["w_dec"] + weights["b_dec"]
        x64 = x.astype(np.float64)
        sse += float(np.sum((x64 - recon.astype(np.float64)) ** 2))
        total_sq += float(np.sum(x64 ** 2))
        sums += x64.sum(axis=0)
        n += len(at)
        nonzero = f != 0
        active += int(nonzero.sum())
        fired |= nonzero.any(axis=0)
        coords = {**dict(record.get("coords") or {}), "model": reading}
        for row, p in enumerate(at):
            tok = int(rendered.ids[p])
            token = {"id": tok, "text": tokenizer.decode([tok])}
            for j in np.flatnonzero(nonzero[row]):
                if keep_set is not None and int(j) not in keep_set:
                    continue
                rows.append({"id": str(record.get("id")), "coords": coords, "position": int(p),
                             "token": token, "feature": int(j), "value": float(f[row, j])})
        if len(rows) > MAX_ROWS:
            raise ValueError(f"dictionary/encode: more than {MAX_ROWS} feature activations; read fewer "
                             "records, only the last position, or only some `features`")
        if on_item:
            on_item()
    if n == 0:
        raise ValueError("dictionary/encode: no positions were read")
    spread = total_sq - float(np.sum(sums ** 2)) / n
    header = weights["header"]
    fidelity = {
        "variance_explained": round(1.0 - sse / spread, 6) if spread > 0 else None,
        "l0": round(active / n, 6),
        "mse": round(sse / (n * d_in), 9),
        "positions": n,
        "inactive": int(width - fired.sum()),
        "published_l0": (header.get("published") or {}).get("l0"),
    }
    return collection(
        "activations/feature", rows,
        dictionary={k: header.get(k) for k in ("derivation", "reads", "writes", "model", "source", "width",
                                               "activation")},
        model=read_model_id(model),
        point={"point": point, "layer": layer},
        positions={"position": position, "skip_bos": skip_bos},
        features=None if keep is None else sorted(keep_set or ()),
        fidelity=fidelity,
    )


def join_readings(base: Mapping[str, Any], adapted: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(items_of(base))
    rows += [{**r, "coords": {**r["coords"], "model": "adapted"}} for r in items_of(adapted)]
    header = {k: v for k, v in read_header(base).items() if k not in ("kind", "item_kind", "key")}
    after = read_header(adapted)["fidelity"]
    before = header["fidelity"]
    drop = (round(before["variance_explained"] - after["variance_explained"], 6)
            if before["variance_explained"] is not None and after["variance_explained"] is not None else None)
    return collection("activations/feature", rows, **{**header, "adapted_fidelity": after,
                                                      "fidelity_drop": drop})
