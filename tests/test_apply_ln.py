"""apply_ln for DLA (task 000142): folding the captured rms scale and
the norm's gain into each component makes the decomposition sum to the
model's true final logits — the whole point of the mode.

Exact at math level with a tiny synthetic unembed; the bfloat16 cast
inside logit_attrs sets the tolerance.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute.attribution import logit_attrs

D, V, S = 8, 10, 5


class TinyHead:
    def __init__(self, w):
        self.w = mx.array(w.astype(np.float32))

    def __call__(self, v):
        return v.astype(mx.float32) @ self.w.T


class StubModel:
    class arch:
        model_type = "llama"

    def __init__(self, wu, gain):
        class Args:
            tie_word_embeddings = False

        class Norm:
            weight = mx.array(gain.astype(np.float32))

        class Inner:
            norm = Norm()

        class M:
            args = Args()
            model = Inner()

        M.lm_head = TinyHead(wu)
        self._model = M


def test_components_sum_to_the_true_final_logits():
    rng = np.random.RandomState(7)
    wu = rng.randn(V, D).astype(np.float32)
    gain = (0.5 + rng.rand(D)).astype(np.float32)
    c1 = rng.randn(S, D).astype(np.float32)
    c2 = rng.randn(S, D).astype(np.float32)
    h = c1 + c2
    eps = 1e-6
    rms = np.sqrt((h * h).mean(axis=-1) + eps)  # [S]
    true_final = ((h[-1] / rms[-1]) * gain) @ wu.T  # [V]

    model = StubModel(wu, gain)
    attrs = logit_attrs(
        model, np.stack([c1, c2]), list(range(V)),
        apply_ln=True, ln_scale=rms,
    )
    summed = attrs.sum(axis=0)  # over components
    # float32 accumulation now: the tolerance is numerical, not
    # representational
    assert np.allclose(summed, true_final, rtol=1e-3, atol=1e-3)


def test_without_the_scale_it_refuses():
    model = StubModel(np.eye(V, D, dtype=np.float32),
                      np.ones(D, dtype=np.float32))
    with pytest.raises(ValueError, match="final_norm.scale"):
        logit_attrs(model, np.zeros((1, S, D)), [0], apply_ln=True)


def test_plain_mode_is_unchanged_by_the_new_kwargs():
    rng = np.random.RandomState(3)
    wu = rng.randn(V, D).astype(np.float32)
    model = StubModel(wu, np.ones(D, dtype=np.float32))
    c = rng.randn(1, S, D).astype(np.float32)
    a = logit_attrs(model, c, [1, 2])
    b = logit_attrs(model, c, [1, 2], apply_ln=False)
    assert np.array_equal(a, b)
