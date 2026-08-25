"""apply_lora on non-uniform towers (the E2B prod failure).

gemma4's attention projections are conditional per layer — KV-shared
tail layers lack k_proj/v_proj, k-eq-v layers lack v_proj. The wrap
must follow the architecture, and a target that matches nowhere must
refuse rather than silently train nothing.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

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
