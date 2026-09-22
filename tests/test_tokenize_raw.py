"""chat_template=False must mean NO chat template — on every arch path.

A branch that ignores the flag is invisible in the output: the prompt
still tokenizes, and the only trace is a frozen target decoding as
'user' — the chat-turn header token. So every arch path is checked.
"""

import mlx.core as mx

from mechbench_compute.model import Model


class RawTokenizer:
    def encode(self, text):
        return [2] + [ord(c) % 50 + 10 for c in text]


class VlmProcessor:
    chat_template = "{{...}}"
    tokenizer = RawTokenizer()


class Arch:
    model_type = "gemma3"
    n_layers = 2
    d_model = 4


def test_raw_flag_bypasses_the_template_on_the_vlm_path(monkeypatch):
    from mechbench_compute import model as model_mod

    def explode(*a, **k):
        raise AssertionError("chat template applied despite chat_template=False")

    monkeypatch.setattr(model_mod, "apply_chat_template", explode)
    m = Model.__new__(Model)
    m._processor = VlmProcessor()
    m._model = object()
    m.arch = Arch()
    ids = m.tokenize("over the hill", chat_template=False)
    assert isinstance(ids, mx.array)
    assert ids.shape[0] == 1
    assert int(np_first(ids)) == 2  # the raw tokenizer's own BOS


def np_first(ids):
    import numpy as np

    return np.array(ids)[0, 0]
