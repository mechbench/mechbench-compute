from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import Any

from mechbench_compute.distill import encode, suffix_tokens
from mechbench_compute.lexicon._base import In, Op, Output, P

OP = Op(
    name="text/tokenize",
    requires="mlx-local",
    summary=(
        "Measure how a tokenizer splits a set of items — as continuations of "
        "a prefix — with a depth histogram, fragmentation, script "
        "composition, and an optional pass/fail gate on expected depth."
    ),
    description="""\
A tokenizer is part of the model, and its behaviour bears on results before
any forward pass: an adapter trained to answer in "one token per slot" only
means something if the vocabulary tokenizes that way *inside its envelope*.
So the block measures items as continuations of a `prefix` — the boundary
matters, because BPE merges across it — which is how training and readout
both see them.

For each item it records the number of tokens it contributes after the
prefix (its **depth**), its pieces, and tokens per whitespace word. Over the
set: the depth histogram, the fraction that are a single token, mean
fragmentation, the Unicode-script composition of the text, and — when
`expect_depth` is given — a gate listing every item that does not tokenize
to exactly that depth.
""",
    inputs=(
        In("vocabulary", "text/word-list",
           "The items to measure: a word list (`words`), or a frequency table "
           "or training target whose `weights` keys are the items. A bare "
           "list of strings inline is read as the words. One of `vocabulary` "
           "and `records` is required.", required=False),
        In("records", "records/record",
           "Records whose `text` (else `user` or `prompt`) is measured, when "
           "no `vocabulary` is given.", many=True, required=False),
    ),
    output=Output('text/tokenization', collection=False, doc="`n_items`, `mean_depth`, `min_depth`, `max_depth`, `single_token_fraction`, `mean_tokens_per_word`, `fragmented_fraction`, `rows` (the histogram: `{depth, count, share}`), `script_composition`, `boundary_failures` (items that changed the prefix's own tokenization), `gate` (`{expected_depth, pass, n_violations, violations}` or null), `most_fragmented`, and `items` when kept."),
    params=(
        P("prefix", "string",
          "The envelope each item is tokenized after — `'{ \"name\": \"'` "
          "for a JSON value. Empty measures items on their own.",
          ""),
        P("expect_depth", "int | string",
          "Turn on the gate: every item must tokenize to exactly this many "
          "tokens after the prefix. `\"\"` or `\"none\"` (as a string param) "
          "means no gate.",
          None),
        P("top_fragmented", "int",
          "How many of the most fragmented items to list under "
          "`most_fragmented`. `0` omits the list.",
          10),
        P("keep_items", "bool",
          "Also emit every item's own measurement (`depth`, `pieces`, "
          "`tokens_per_word`) under `items`.",
          False),
    ),
    example={
        "model": {"$param": "model"},
        "prefix": "{ \"color\": \"",
        "expect_depth": 1,
    },
    example_inputs={"vocabulary": ["red", "blue", "green", "turquoise"]},
)


def run(ctx, inputs, params):
    model = ctx.model(params.get("model"))
    return measure_model_tokenizer(model, inputs, params)


def _read_strings(inputs: Mapping[str, Any]) -> list[str]:
    items = None
    obj = inputs.get("vocabulary")
    if isinstance(obj, Mapping):
        if isinstance(obj.get("weights"), Mapping):
            items = list(obj["weights"].keys())
        elif isinstance(obj.get("words"), list):
            items = list(obj["words"])
    elif isinstance(obj, list):
        items = obj
    if items is None and inputs.get("records") is not None:
        from mechbench_compute.lexicon import kinds as K

        items = []
        for r in K.items_of(inputs["records"]):
            v = (r.get("text") or r.get("user") or r.get("prompt")) if isinstance(r, Mapping) else r
            if isinstance(v, str) and v.strip():
                items.append(v)
    if not items:
        raise ValueError(
            "text/tokenize needs a `vocabulary` (a word list, a weights map, "
            "or a list of strings) or `records` with text")
    return [str(x) for x in items]


def _classify_char(ch: str) -> str:
    if ch.isspace():
        return "space"
    if ch.isdigit():
        return "digit"
    cat = unicodedata.category(ch)
    if cat[0] in ("P", "S"):
        return "punct"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    head = name.split(" ")[0]
    if head in ("LATIN",):
        return "latin"
    if head in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"):
        return "cjk"
    if head in ("CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI",
                "THAI", "BENGALI", "TAMIL"):
        return head.lower()
    return "other"


def measure_tokenizer(tokenizer, tokenizer_id: str, inputs: Mapping[str, Any],
                      params: Mapping[str, Any]) -> dict[str, Any]:
    items = _read_strings(inputs)
    prefix = str(params.get("prefix", "") or "")
    prefix_ids = encode(tokenizer, prefix) if prefix else []
    expect = params.get("expect_depth")
    if isinstance(expect, str):
        expect = None if expect.strip().lower() in ("", "none") else expect
    keep = bool(params.get("keep_items", False))
    top_n = int(params.get("top_fragmented", 10) or 0)

    per: list[dict[str, Any]] = []
    boundary_failures: list[str] = []
    for it in items:
        if prefix:
            try:
                seq = suffix_tokens(tokenizer, prefix, prefix_ids, it)
            except ValueError:
                boundary_failures.append(it)
                seq = encode(tokenizer, it)
        else:
            seq = encode(tokenizer, it)
        words = it.split()
        pieces = [tokenizer.decode([int(t)]) for t in seq]
        per.append({
            "item": it,
            "depth": len(seq),
            "words": len(words),
            "tokens_per_word": (round(len(seq) / len(words), 4)
                                if words else None),
            "pieces": pieces,
        })

    depths = [p["depth"] for p in per]
    n = len(per)
    hist: dict[int, int] = {}
    for d in depths:
        hist[d] = hist.get(d, 0) + 1
    tpw = [p["tokens_per_word"] for p in per if p["tokens_per_word"] is not None]

    chars = "".join(items)
    scripts: dict[str, int] = {}
    for ch in chars:
        s = _classify_char(ch)
        scripts[s] = scripts.get(s, 0) + 1
    total_chars = max(1, len(chars))

    gate = None
    if expect is not None:
        expect = int(expect)
        bad = [{"item": p["item"], "depth": p["depth"], "pieces": p["pieces"]}
               for p in per if p["depth"] != expect]
        gate = {"expected_depth": expect, "pass": not bad,
                "n_violations": len(bad), "violations": bad[:50]}

    out: dict[str, Any] = {
        "kind": "text/tokenization",
        "tokenizer": tokenizer_id,
        "prefix": prefix,
        "n_items": n,
        "mean_depth": round(sum(depths) / n, 4),
        "max_depth": max(depths),
        "min_depth": min(depths),
        "single_token_fraction": round(sum(1 for d in depths if d == 1) / n, 4),
        "mean_tokens_per_word": round(sum(tpw) / len(tpw), 4) if tpw else None,
        "fragmented_fraction": round(
            sum(1 for p in per if p["words"] and p["depth"] > p["words"]) / n, 4),
        "rows": [{"depth": d, "count": c, "share": round(c / n, 4)}
                 for d, c in sorted(hist.items())],
        "script_composition": {k: round(v / total_chars, 4)
                               for k, v in sorted(scripts.items(),
                                                  key=lambda kv: -kv[1])},
        "boundary_failures": boundary_failures[:50],
        "gate": gate,
    }
    if top_n:
        worst = sorted(per, key=lambda p: -(p["tokens_per_word"] or 0))[:top_n]
        out["most_fragmented"] = [
            {"item": p["item"], "depth": p["depth"], "pieces": p["pieces"]}
            for p in worst]
    if keep:
        out["items"] = per
    return out


def measure_model_tokenizer(model, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    tid = (getattr(model, "model_id", None) or getattr(model, "id", None)
           or str(params.get("model") or ""))
    return measure_tokenizer(model.tokenizer, str(tid), inputs, params)
