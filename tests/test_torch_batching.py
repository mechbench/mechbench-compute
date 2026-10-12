from types import SimpleNamespace

import pytest

from mechbench_compute.torch_backend.batching import read_kv_widths


class PerLayerConfig(SimpleNamespace):
    def __getattribute__(self, name):
        if name in ("head_dim", "num_key_value_heads"):
            raise RuntimeError(f"{name!r} is a per-layer attribute")
        return super().__getattribute__(name)


def test_a_uniform_config_gives_one_width_per_layer():
    cfg = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                          hidden_size=512, num_hidden_layers=3)
    assert read_kv_widths(cfg) == [128, 128, 128]


def test_a_config_without_head_dim_divides_the_hidden_size():
    cfg = SimpleNamespace(num_attention_heads=8, hidden_size=512, num_hidden_layers=2)
    assert read_kv_widths(cfg) == [512, 512]


def test_a_per_layer_config_is_read_layer_by_layer_without_the_ambiguous_attributes():
    sliding = SimpleNamespace(num_key_value_heads=16, head_dim=256)
    full = SimpleNamespace(num_key_value_heads=4, head_dim=512)
    cfg = PerLayerConfig(num_attention_heads=32, hidden_size=5376, num_hidden_layers=3,
                         per_layer_config=[sliding, sliding, full])
    with pytest.raises(RuntimeError):
        cfg.head_dim
    assert read_kv_widths(cfg) == [4096, 4096, 2048]
