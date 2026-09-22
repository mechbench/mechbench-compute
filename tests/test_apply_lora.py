"""apply_lora on non-uniform towers (the E2B prod failure).

gemma4's attention projections are conditional per layer — KV-shared
tail layers lack k_proj/v_proj, k-eq-v layers lack v_proj. The wrap
must follow the architecture, and a target that matches nowhere must
refuse rather than silently train nothing.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mechbench_compute import lora


class FakeAttn(nn.Module):
    def __init__(self, with_v: bool):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        if with_v:
            self.v_proj = nn.Linear(8, 8, bias=False)


class FakeLayer(nn.Module):
    def __init__(self, with_v: bool):
        super().__init__()
        self.self_attn = FakeAttn(with_v)


class FakeInner(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.layers = [FakeLayer(v) for v in pattern]


class FakeLM(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.model = FakeInner(pattern)


class TestConditionalProjections:
    def test_wraps_v_proj_only_where_it_exists(self):
        lm = FakeLM([True, False, True])
        n = lora.apply_lora(lm, rank=2, alpha=4)
        # q on all three layers, v on two — and nothing crashed on the
        # layer that has no v_proj.
        wrapped = [
            isinstance(getattr(layer.self_attn, "v_proj", None), lora.LoRALinear)
            for layer in lm.model.layers
        ]
        assert wrapped == [True, False, True]
        assert all(
            isinstance(layer.self_attn.q_proj, lora.LoRALinear)
            for layer in lm.model.layers
        )
        assert n > 0

    def test_a_target_matching_no_layer_refuses(self):
        lm = FakeLM([False, False])
        with pytest.raises(ValueError, match="v_proj"):
            lora.apply_lora(lm, rank=2, alpha=4)

    def test_unknown_target_still_names_the_known_set(self):
        lm = FakeLM([True])
        with pytest.raises(ValueError, match="known"):
            lora.apply_lora(lm, rank=2, alpha=4, targets=("frobnicate",))


class TestSeededInitialization:
    """The `A` draw is seeded. `B` starts at zero, so `A` changes
    nothing at step 0 and everything after it: unseeded, two runs of one
    protocol at one seed train different adapters, by more than the
    drift an experiment compares releases by."""

    def _a_matrices(self, lm):
        return [np.array(layer.self_attn.q_proj.lora_a.astype(mx.float32))
                for layer in lm.model.layers]

    def test_the_same_seed_starts_from_the_same_adapter(self):
        first, second = FakeLM([True, True]), FakeLM([True, True])
        lora.apply_lora(first, rank=2, alpha=4, seed=7)
        lora.apply_lora(second, rank=2, alpha=4, seed=7)
        for a, b in zip(self._a_matrices(first), self._a_matrices(second)):
            assert np.array_equal(a, b)

    def test_a_different_seed_starts_somewhere_else(self):
        first, second = FakeLM([True, True]), FakeLM([True, True])
        lora.apply_lora(first, rank=2, alpha=4, seed=7)
        lora.apply_lora(second, rank=2, alpha=4, seed=8)
        assert not np.array_equal(self._a_matrices(first)[0],
                                  self._a_matrices(second)[0])

    def test_each_projection_draws_its_own(self):
        # One key reused across layers would start every layer from the
        # same matrix — a symmetry the gradients would never break.
        lm = FakeLM([True, True, True])
        lora.apply_lora(lm, rank=2, alpha=4, seed=7)
        mats = self._a_matrices(lm)
        assert not np.array_equal(mats[0], mats[1])
        assert not np.array_equal(mats[1], mats[2])

    def test_without_a_seed_the_global_generator_still_answers(self):
        # The bare call keeps its old behaviour for a caller that wants
        # an unrepeatable draw; the block always passes a seed.
        first, second = FakeLM([True]), FakeLM([True])
        lora.apply_lora(first, rank=2, alpha=4)
        lora.apply_lora(second, rank=2, alpha=4)
        assert not np.array_equal(self._a_matrices(first)[0],
                                  self._a_matrices(second)[0])

    def test_the_seed_leaves_the_global_generator_alone(self):
        # Seeding by key rather than by `mx.random.seed` — a block that
        # seeds its own init must not silently re-seed the process.
        lm = FakeLM([True])  # built first: nn.Linear draws from the global one
        mx.random.seed(11)
        expected = mx.random.normal((4,))
        mx.random.seed(11)
        lora.apply_lora(lm, rank=2, alpha=4, seed=7)
        assert np.array_equal(np.array(mx.random.normal((4,))),
                              np.array(expected))
