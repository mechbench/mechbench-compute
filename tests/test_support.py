from __future__ import annotations

import json
import types

import mlx.core as mx
import pytest

from mechbench_compute import _arch, architectures, support
from mechbench_compute.architectures import ARCHITECTURES, BY_MODEL_TYPE
from mechbench_compute.model import Model
from mechbench_compute.providers import pricing
from mechbench_compute.providers.registry import registry
from tests.tiny_models import MODEL_TYPES


def test_the_declared_architectures_are_the_ones_the_tiny_models_cover() -> None:
    assert set(BY_MODEL_TYPE) == set(MODEL_TYPES)


@pytest.mark.parametrize("arch", ARCHITECTURES, ids=lambda a: a.model_type)
def test_the_hook_gate_reads_the_declaration(arch: support.Architecture) -> None:
    for point in _arch.LAYER_HOOK_POINTS:
        assert _arch.family_supports(arch.model_type, point, layer_scoped=True) == (
            point in arch.layer_points)
    for point in _arch.GLOBAL_HOOK_POINTS:
        assert _arch.family_supports(arch.model_type, point, layer_scoped=False) == (
            point in arch.global_points)


def test_levels_follow_the_points() -> None:
    assert BY_MODEL_TYPE["gemma4"].level == "full"
    for mt in ("gemma3", "qwen2", "llama"):
        assert BY_MODEL_TYPE[mt].level == "core"
    assert {a.level for a in ARCHITECTURES} <= set(support.LEVELS)
    assert {a.loader for a in ARCHITECTURES} <= set(support.LOADERS)


def test_an_architecture_is_found_from_a_loaded_model_or_refused_by_name() -> None:
    for a in ARCHITECTURES:
        if a.loader != "mlx-lm":
            continue
        args = types.SimpleNamespace(
            model_type=a.model_type, num_hidden_layers=2, hidden_size=8,
            num_attention_heads=2, num_key_value_heads=1, vocab_size=16)
        loaded = types.SimpleNamespace(args=args)
        assert architectures.for_model(loaded) is a
        assert _arch.Arch.from_mlx_model(loaded).model_type == a.model_type
    with pytest.raises(NotImplementedError, match="qwen3"):
        architectures.for_model(types.SimpleNamespace(
            args=types.SimpleNamespace(model_type="qwen3", num_hidden_layers=1)))
    assert architectures.for_type("mamba") is None


def test_an_architecture_lives_at_its_model_type() -> None:
    for a in ARCHITECTURES:
        module = __import__(f"mechbench_compute.architectures.{a.model_type}",
                            fromlist=["ARCH"])
        assert module.ARCH is a


def test_refusal_names_an_unknown_model_type_and_a_refusing_key() -> None:
    assert support.refusal({"model_type": "Qwen2"}) is None
    assert "qwen3" in (support.refusal({"model_type": "qwen3"}) or "")
    assert support.refusal({"model_type": "gemma4",
                            "text_config": {"enable_moe_block": False}}) is None
    assert "mixture-of-experts" in (support.refusal(
        {"model_type": "gemma4", "text_config": {"enable_moe_block": True}}) or "")


@pytest.mark.parametrize("config", [
    {"model_type": "qwen3"},
    {"model_type": "gemma4", "text_config": {"enable_moe_block": True}},
])
def test_load_refuses_what_the_declaration_refuses_before_loading(tmp_path, config) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="cannot load"):
        Model.load(str(tmp_path))


def test_a_gemma4_moe_layer_is_refused_by_the_forward_too() -> None:
    from mlx_vlm.models.gemma4 import config, gemma4

    text = config.TextConfig(hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             global_head_dim=8, vocab_size=64, vocab_size_per_layer_input=64,
                             hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                             sliding_window=3, sliding_window_pattern=2,
                             enable_moe_block=True, moe_intermediate_size=16,
                             num_experts=4, top_k_experts=2)
    vision = config.VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                                 global_head_dim=8, position_embedding_size=16)
    wrapped = gemma4.Model(config.ModelConfig(text_config=text, vision_config=vision,
                                              vocab_size=64, image_token_id=62,
                                              audio_token_id=63))
    model = Model(wrapped, None)
    assert support.refusal({"model_type": model.arch.model_type,
                            "text_config": {"enable_moe_block": True}}) is not None
    with pytest.raises(NotImplementedError, match="MoE"):
        model.run(mx.array([[1, 2, 3]]))


def test_provider_models_are_the_price_table_of_every_real_provider() -> None:
    rows = support.provider_models()
    real = {n for n, spec in registry().items() if spec.base_url}
    assert {r["provider"] for r in rows} == {p for p in real if pricing.PRICES.get(p)}
    assert "mock" not in {r["provider"] for r in rows}
    for r in rows:
        price = pricing.PRICES[r["provider"]][r["model"]]
        assert (r["inputPerMillion"], r["outputPerMillion"]) == (price.input, price.output)
        assert (r["cacheReadPerMillion"], r["cacheWritePerMillion"]) == (
            price.cache_read, price.cache_write)
        caps = registry()[r["provider"]].capabilities
        assert (r["tools"], r["reasoning"]) == (caps.tools, caps.reasoning)
