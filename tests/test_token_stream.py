from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import generate
from mechbench_compute.token_readout import TokenReadout

VOCAB = 8


class _Tok:
    eos_token_id = 0

    def decode(self, ids):
        return "".join("ab"[i % 2] if i < 4 else "é" for i in ids)


class _Arch:
    d_model = 3
    n_layers = 4


class _Result:
    def __init__(self, logits, cache):
        self.logits, self.cache = logits, cache


class _Model:
    tokenizer = _Tok()
    arch = _Arch()

    def _logits(self, t):
        row = np.full(VOCAB, -9.0, dtype=np.float32)
        row[(t + 1) if t + 1 < 6 else 0] = 9.0
        return mx.array(row.reshape(1, 1, VOCAB))

    def run(self, input_ids, *, interventions=None, kv_cache=None, capture=None, hooks=None):
        t = int(np.array(input_ids)[0, -1])
        cache = {name: mx.array(np.array([[[float(t), 1.0, 0.0]]], dtype=np.float32)).astype(mx.bfloat16)
                 for name in (capture or [])}
        return _Result(self._logits(t), cache)

    def lm(self, input_ids, cache=None):
        return self._logits(int(np.array(input_ids)[0, -1]))


DIRECTION = {"kind": "direction/vector", "vector": [1.0, 0.0, 0.0], "space": {"model": None, "layer": 2, "point": "resid_post", "head": None, "d": 3}}


def _run(**kw):
    model = _Model()
    prefill = ([], model._logits(1)[0, -1, :])
    return model, generate.sample_completion_cached(model, [1], max_tokens=10, temperature=0, prefill=prefill,
                                                    return_ids=True, **kw)


class TestAReplyStreams:
    def test_each_token_arrives_as_it_is_made_and_the_pieces_join_to_the_reply(self):
        seen = []
        pieces: list[str] = []
        _, (text, ids) = _run(on_token=seen.append, pieces_out=pieces)
        assert ids == [2, 3, 4, 5]
        assert [e["index"] for e in seen] == [0, 1, 2, 3]
        assert "".join(e["text"] for e in seen) == text == "".join(pieces)
        assert all("coord" not in e for e in seen)

    def test_a_readout_projects_each_token_onto_the_direction_as_it_goes(self):
        model = _Model()
        readout = TokenReadout(model, DIRECTION)
        seen = []
        prefill = ([], model._logits(1)[0, -1, :])
        generate.sample_completion_cached(model, [1], max_tokens=10, temperature=0, prefill=prefill,
                                          on_token=seen.append, readout=readout)
        assert [e["coord"] for e in seen] == [2.0, 3.0, 4.0, 5.0] == readout.coords
        rec = readout.record([e["text"] for e in seen])
        assert rec["coords"] == [2.0, 3.0, 4.0, 5.0] and len(rec["tokens"]) == 4
        assert rec["direction"]["space"]["layer"] == 2

    def test_a_direction_without_a_layer_is_refused(self):
        with pytest.raises(ValueError, match="no layer"):
            TokenReadout(_Model(), {"kind": "direction/vector", "vector": [1.0, 0.0, 0.0], "space": {"layer": None, "point": "resid_post", "d": 3}})

    def test_a_direction_of_another_width_is_refused(self):
        with pytest.raises(ValueError, match="dims"):
            TokenReadout(_Model(), {**DIRECTION, "vector": [1.0, 0.0]})
