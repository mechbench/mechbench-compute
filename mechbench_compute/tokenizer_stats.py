"""Tokenizer diagnostics as a block (task 000377, lexicon epic 000364).

A tokenizer is a constituent of the model as much as a weight matrix
is, and its properties bear on results before any forward pass: an
adapter trained on "one token per slot" is only meaningful if the
vocabulary tokenizes that way in its envelope (the naturalism gate),
and experiment 015's first question about trie depth — how deep were
the sequences an adapter was actually trained on? — is a property of
the (vocabulary, tokenizer, envelope) triple, not of any activation.

`~canonical/ops/tokenize/stats/1` measures a set of items AS
CONTINUATIONS OF A PREFIX (the envelope: `'{ "name": "'`), because the
boundary matters — BPE merges across it — and that is how training
and readout see them:

    depth            tokens the item contributes after the prefix
    depth_hist       how many items at each depth (the trie inventory)
    single-token     fraction of items that are one token
    tokens_per_word  fragmentation against whitespace words
    scripts          Unicode script composition of the item text
    gate             pass/fail against an expected depth, with the
                     violating items — finetune's naturalism check as
                     a standalone readout

Items come from `items` (strings), a `target_map`'s weight keys (the
vocabulary object a trainer consumes), or records' text field.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.distill import encode, suffix_tokens


def _items_of(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> list[str]:
    items = params.get("items")
    if items is None:
        obj = inputs.get("vocabulary") or params.get("vocabulary")
        if isinstance(obj, Mapping) and isinstance(obj.get("weights"), Mapping):
            items = list(obj["weights"].keys())
        elif isinstance(obj, list):
            items = obj
    if items is None:
        recs = inputs.get("records") or params.get("records")
        if isinstance(recs, Mapping):
            recs = recs.get("records") or recs.get("items") or recs.get("rows")
        if isinstance(recs, list):
            field = str(params.get("field", "text"))
            items = []
            for r in recs:
                v = (r.get(field) or r.get("text") or r.get("user")
                     or r.get("prompt")) if isinstance(r, Mapping) else r
                if isinstance(v, str) and v.strip():
                    items.append(v)
    if not items:
        raise ValueError(
            "tokenize/stats needs `items`, a `vocabulary` (target_map or "
            "list), or records with text")
    return [str(x) for x in items]


def _script_of(ch: str) -> str:
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


def tokenizer_stats(tokenizer, tokenizer_id: str, inputs: Mapping[str, Any],
                    params: Mapping[str, Any]) -> dict[str, Any]:
    """The measurement, over a tokenizer object; the block wrapper in
    protocol.py supplies the bound model's tokenizer."""
    items = _items_of(inputs, params)
    prefix = str(params.get("prefix", "") or "")
    prefix_ids = encode(tokenizer, prefix) if prefix else []
    expect = params.get("expect_depth")
    keep = bool(params.get("keep_items", False))
    top_n = int(params.get("top_fragmented", 10) or 0)

    per: list[dict[str, Any]] = []
    boundary_failures: list[str] = []
    for it in items:
        if prefix:
            try:
                seq = suffix_tokens(tokenizer, prefix, prefix_ids, it)
            except ValueError:
                # The item changed the prefix's own tokenization: the
                # envelope boundary is unstable for it. Count it, and
                # measure it raw so the histogram still has a row.
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
        s = _script_of(ch)
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
        "kind": "tokenizer_stats",
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
        # The trie inventory, as rows a table renderer draws.
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


def block(model, inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """The executor's entry: the bound model's tokenizer, named by the
    binding so the record says which vocabulary it measured."""
    tid = (getattr(model, "model_id", None) or getattr(model, "id", None)
           or str(params.get("model") or ""))
    return tokenizer_stats(model.tokenizer, str(tid), inputs, params)


__all__: Sequence[str] = ("block", "tokenizer_stats")
