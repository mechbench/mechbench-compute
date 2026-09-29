import mlx.core as mx
import numpy as np

from mechbench_compute.architectures import for_type
from mechbench_compute.model import Model


class RawTokenizer:
    def encode(self, text):
        return [2] + [ord(c) % 50 + 10 for c in text]


class VlmProcessor:
    chat_template = "{{...}}"
    tokenizer = RawTokenizer()


def test_raw_flag_bypasses_the_template_on_the_vlm_path(monkeypatch):
    import mlx_vlm.prompt_utils

    def explode(*a, **k):
        raise AssertionError("chat template applied despite chat_template=False")

    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", explode)
    m = Model.__new__(Model)
    m._processor = VlmProcessor()
    m._model = object()
    m.architecture = for_type("gemma3")
    ids = m.tokenize("over the hill", chat_template=False)
    assert isinstance(ids, mx.array)
    assert ids.shape[0] == 1
    assert int(np.array(ids)[0, 0]) == 2
