from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from mechbench_compute import attribution
from mechbench_compute.model import Model

FAMILIES = ["gemma4", "gemma3", "qwen2", "llama"]
WINDOW = 3


def _build_mlx_lm(family):
    from mlx_lm.models import llama, qwen2

    module = {"qwen2": qwen2, "llama": llama}[family]
    kw = dict(model_type=family, hidden_size=32, num_hidden_layers=4,
              intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
              rms_norm_eps=1e-6, vocab_size=64, tie_word_embeddings=False)
    if family == "llama":
        kw.update(layer_types=["sliding_attention", "full_attention"] * 2,
                  sliding_window=WINDOW)
    model = module.Model(module.ModelArgs(**kw))
    return model, model


def _build_gemma3():
    from mlx_vlm.models.gemma3 import config, language

    cfg = config.TextConfig(model_type="gemma3_text", hidden_size=32, num_hidden_layers=4,
                            intermediate_size=64, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, vocab_size=64,
                            sliding_window=WINDOW, sliding_window_pattern=2,
                            query_pre_attn_scalar=8)
    lm = language.LanguageModel(cfg)
    return SimpleNamespace(language_model=lm, config=SimpleNamespace(text_config=cfg)), lm


def _build_gemma4():
    from mlx_vlm.models.gemma4 import config, gemma4

    text = config.TextConfig(hidden_size=32, num_hidden_layers=4, intermediate_size=64,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             global_head_dim=8, vocab_size=64, vocab_size_per_layer_input=64,
                             hidden_size_per_layer_input=8, num_kv_shared_layers=0,
                             sliding_window=WINDOW, sliding_window_pattern=2)
    vision = config.VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                                 global_head_dim=8, position_embedding_size=16)
    model = gemma4.Model(config.ModelConfig(text_config=text, vision_config=vision,
                                            vocab_size=64, image_token_id=62,
                                            audio_token_id=63))
    return model, model.language_model


def _build_tiny_model(family) -> Model:
    if family in ("qwen2", "llama"):
        wrapped, lm = _build_mlx_lm(family)
    else:
        wrapped, lm = {"gemma3": _build_gemma3, "gemma4": _build_gemma4}[family]()
    keys = iter(mx.random.split(mx.random.key(11), 1000))
    lm.update(tree_map(lambda p: mx.random.normal(p.shape, key=next(keys)) * 0.5,
                       lm.parameters()))
    mx.eval(lm.parameters())
    model = Model(wrapped, None)
    assert model.arch.model_type == family
    return model


@pytest.fixture(scope="module", params=FAMILIES)
def tiny(request):
    return _build_tiny_model(request.param)


def _read_logits(result) -> np.ndarray:
    return np.array(result.logits.astype(mx.float32))


@pytest.mark.parametrize("n_tokens", [WINDOW, 3 * WINDOW + 1])
def test_capturing_attention_internals_at_every_layer_leaves_the_logits_alone(tiny, n_tokens):
    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 8, 6][:n_tokens]])
    plain = _read_logits(tiny.run(ids))
    probed = tiny.run(ids, capture=[f"blocks.{i}.attn.weights"
                                    for i in range(tiny.arch.n_layers)])
    assert np.abs(plain).max() > 1.0
    assert np.allclose(_read_logits(probed), plain, atol=1e-4, rtol=1e-4)
    for i in range(tiny.arch.n_layers):
        weights = np.array(probed.cache[f"blocks.{i}.attn.weights"].astype(mx.float32))[0]
        assert np.allclose(np.triu(weights, k=1), 0.0)


def test_direct_logit_attribution_with_the_final_norm_sums_to_the_true_logit(tiny):
    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4]])
    n = tiny.arch.n_layers
    result = tiny.run(ids, capture=[*(f"blocks.{i}.resid_post" for i in range(n)),
                                    "blocks.0.resid_pre", "final_norm.scale"])
    last = _read_logits(result)[0, -1].astype(np.float64)
    cap = getattr(tiny.lm, "final_logit_softcapping", None)
    if cap:
        last = cap * np.arctanh(last / cap)
    acc = attribution.accumulated_resid(result.cache, include_pre=True)
    components = np.diff(acc, axis=0, prepend=np.zeros_like(acc[:1]))
    ln_scale = np.array(result.cache["final_norm.scale"].astype(mx.float32)).reshape(-1)
    targets = [int(np.argmax(last)), int(np.argmin(last)), 17]
    attrs = attribution.logit_attrs(tiny, components, targets, apply_ln=True,
                                    ln_scale=ln_scale)
    assert np.allclose(attrs.sum(axis=0), last[targets], atol=2e-3, rtol=1e-3)
