from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from mechbench_compute.adapters.tokenize_conversation import read_turns
from mechbench_compute.distill import encode, render_chat

CURSOR = 4


class Source(Protocol):
    def next_doc(self) -> Mapping[str, Any]: ...

    def cursor(self) -> list[int]: ...

    def seek(self, cursor: Sequence[int]) -> None: ...

    def describe(self) -> dict[str, Any]: ...


def order(seed: int, *key: int, n: int) -> np.ndarray:
    return np.random.default_rng([seed, *key]).permutation(n)


class RecordsSource:
    def __init__(self, records: Sequence[Mapping[str, Any]], seed: int, index: int) -> None:
        if not records:
            raise ValueError("dictionary/train: the `records` port holds no records to train on")
        self.records = list(records)
        self.seed, self.index = seed, index
        self.epoch = self.pos = 0
        self.perm = order(seed, index, 0, n=len(self.records))

    def next_doc(self) -> Mapping[str, Any]:
        if self.pos >= len(self.records):
            self.epoch, self.pos = self.epoch + 1, 0
            self.perm = order(self.seed, self.index, self.epoch, n=len(self.records))
        doc = self.records[int(self.perm[self.pos])]
        self.pos += 1
        return doc

    def cursor(self) -> list[int]:
        return [self.epoch, self.pos, 0, 0]

    def seek(self, cursor: Sequence[int]) -> None:
        self.epoch, self.pos = int(cursor[0]), int(cursor[1])
        self.perm = order(self.seed, self.index, self.epoch, n=len(self.records))

    def describe(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for r in self.records:
            digest.update(repr(sorted((k, str(v)) for k, v in r.items())).encode())
        return {"records": {"count": len(self.records), "sha256": digest.hexdigest()},
                "epochs": self.epoch}


def tokenize_document(tokenizer: Any, doc: Mapping[str, Any]) -> list[int]:
    if doc.get("messages") is not None:
        text = tokenizer.apply_chat_template(read_turns(doc), tokenize=False)
    elif isinstance(doc.get("text"), str):
        text = doc["text"]
    elif isinstance(doc.get("user"), str):
        text = render_chat(tokenizer, str(doc.get("system") or ""), doc["user"])
    else:
        raise ValueError(f"dictionary/train: the document {doc.get('id')!r} has no `text`, `messages` "
                         "or `user` to read")
    ids = encode(tokenizer, text)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and ids[:1] != [bos]:
        ids = [bos, *ids]
    return ids


class TokenStream:
    def __init__(self, sources: Sequence[Source], weights: Sequence[float], tokenizer: Any,
                 context: int) -> None:
        self.sources, self.weights = list(sources), [float(w) for w in weights]
        if any(not w > 0 for w in self.weights):
            raise ValueError("dictionary/train: every corpus source's `weight` is above 0")
        self.tokenizer, self.context = tokenizer, context
        self.counts = [0] * len(self.sources)
        self.carry: list[int] = []

    def pick(self) -> int:
        return min(range(len(self.sources)), key=lambda i: ((self.counts[i] + 1) / self.weights[i], i))

    def next_window(self) -> list[int]:
        while len(self.carry) < self.context:
            i = self.pick()
            self.carry += tokenize_document(self.tokenizer, self.sources[i].next_doc())
            self.counts[i] += 1
        window, self.carry = self.carry[:self.context], self.carry[self.context:]
        return window

    def state(self) -> dict[str, np.ndarray]:
        return {"cursors": np.array([s.cursor() for s in self.sources], dtype=np.int64).reshape(-1, CURSOR),
                "counts": np.array(self.counts, dtype=np.int64),
                "carry": np.array(self.carry, dtype=np.int64)}

    def restore(self, state: Mapping[str, Any]) -> None:
        cursors = np.asarray(state["cursors"], dtype=np.int64).reshape(-1, CURSOR)
        if len(cursors) != len(self.sources):
            raise ValueError(f"dictionary/train: the saved training state reads {len(cursors)} corpus "
                             f"sources and this node {len(self.sources)}; restart the node")
        for source, cursor in zip(self.sources, cursors, strict=True):
            source.seek([int(c) for c in cursor])
        self.counts = [int(c) for c in np.asarray(state["counts"]).reshape(-1)]
        self.carry = [int(t) for t in np.asarray(state["carry"]).reshape(-1)]

    def describe(self) -> list[dict[str, Any]]:
        return [{**s.describe(), "weight": w, "documents": n}
                for s, w, n in zip(self.sources, self.weights, self.counts, strict=True)]


def open_stream(corpus: Any, records: Sequence[Mapping[str, Any]], tokenizer: Any, *, seed: int,
                context: int) -> TokenStream:
    if records and corpus is not None:
        raise ValueError("dictionary/train: the corpus is the `records` port or the `corpus` param; "
                         "this node gives both")
    if records:
        return TokenStream([RecordsSource(records, seed, 0)], [1.0], tokenizer, context)
    if corpus is None:
        raise ValueError("dictionary/train: give a `corpus` (a Hugging Face dataset, "
                         '`{"hub": {"repo": …, "split": …, "field": "text"}}`) or bind records to `records`')
    specs = list(corpus) if isinstance(corpus, list) else [corpus]
    if not specs:
        raise ValueError("dictionary/train: `corpus` lists no sources")
    from mechbench_compute.dictionaries.train_hub import HubSource

    sources, weights = [], []
    for i, spec in enumerate(specs):
        if not isinstance(spec, Mapping) or not isinstance(spec.get("hub"), Mapping):
            raise ValueError(f"dictionary/train: corpus source {i} is "
                             '`{"hub": {"repo": …, "config": …, "split": …, "revision": …, "field": …}, '
                             f'"weight": 1}}`, not {spec!r}')
        sources.append(HubSource(spec["hub"], seed, i))
        weights.append(spec.get("weight", 1))
    return TokenStream(sources, weights, tokenizer, context)
