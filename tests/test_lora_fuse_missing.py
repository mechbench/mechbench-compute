from __future__ import annotations

import mlx.core as mx
import pytest

from mechbench_compute import lora


class _Mod:
    def __init__(self, d):
        self.weight = mx.zeros((d, d))


class _Attn:
    def __init__(self, d, with_v):
        self.q_proj = _Mod(d)
        if with_v:
            self.v_proj = _Mod(d)


class _Layer:
    def __init__(self, d, with_v):
        self.self_attn = _Attn(d, with_v)


class _Inner:
    def __init__(self, layers):
        self.layers = layers


class _LM:
    def __init__(self, d=4, n=4, v_until=2):
        self.model = _Inner([_Layer(d, i < v_until) for i in range(n)])


def _weights(d=4, n=4):
    w = {}
    for i in range(n):
        for proj in ("q_proj", "v_proj"):
            w[f"model.layers.{i}.self_attn.{proj}.lora_a"] = mx.ones((2, d))
            w[f"model.layers.{i}.self_attn.{proj}.lora_b"] = mx.ones((d, 2))
    return w


class TestMissingModules:
    def test_refuses_by_default_and_names_them(self):
        with pytest.raises(ValueError, match=r"self_attn\.v_proj on layers 2\.\.3 \(2\)"):
            lora.fuse(_LM(), _weights(), scale=2.0)

    def test_skip_missing_fuses_the_rest_and_reports(self):
        lm = _LM()
        skipped: list[str] = []
        handle = lora.fuse(lm, _weights(), scale=2.0, skip_missing=True,
                           skipped=skipped)
        assert skipped == ["2.self_attn.v_proj", "3.self_attn.v_proj"]
        assert sorted(handle) == [(0, "self_attn", "q_proj"), (0, "self_attn", "v_proj"),
                                  (1, "self_attn", "q_proj"), (1, "self_attn", "v_proj"),
                                  (2, "self_attn", "q_proj"), (3, "self_attn", "q_proj")]
        assert float(lm.model.layers[3].self_attn.q_proj.weight[0, 0]) == 4.0
        lora.restore(lm, handle)
        assert float(lm.model.layers[3].self_attn.q_proj.weight[0, 0]) == 0.0

    def test_an_adapter_that_fits_reports_nothing(self):
        lm = _LM(v_until=4)
        skipped: list[str] = []
        lora.fuse(lm, _weights(), scale=2.0, skip_missing=True, skipped=skipped)
        assert skipped == []
