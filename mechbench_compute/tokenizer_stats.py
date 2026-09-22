"""Tokenizer diagnostics as a block.

A tokenizer is a constituent of the model as much as a weight matrix
is, and its properties bear on results before any forward pass: an
adapter trained on "one token per slot" is only meaningful if the
vocabulary tokenizes that way in its envelope (the naturalism gate).
Trie depth is a property of the (vocabulary, tokenizer, envelope)
triple, not of any activation.

`text/tokenize` measures a set of items AS
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

from collections.abc import Sequence

__all__: Sequence[str] = ("measure_model_tokenizer", "measure_tokenizer")
