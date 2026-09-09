"""Task 000365: the forward-pass grammar's new points — registry,
selectors and validation without a model; an opt-in smoke test on the
real Gemma 4 E2B when it is cached locally (MECHBENCH_MODEL_TESTS=1)."""

from __future__ import annotations

import os

import pytest

from mechbench_compute import _arch
from mechbench_compute.errors import InvalidHookName
from mechbench_compute.hooks import (
    attn_internal_layers,
    mlp_internal_layers,
    parse_hook_name,
)

NEW_LAYER = [
    "attn.in_norm", "attn.q_pre_norm", "attn.k_pre_norm", "attn.q_pre_rope",
    "attn.k_pre_rope", "attn.scores", "attn.o_in",
    "mlp.in_norm", "mlp.gate", "mlp.up", "mlp.act", "mlp.down_in",
]
NEW_GLOBAL = ["embed", "final_norm", "logits"]


def _arch_e2b_like(model_type="gemma4"):
    return _arch.Arch(
        model_id="fake/e2b", n_layers=35, d_model=1536, n_heads=8, n_kv_heads=1,
        vocab_size=262144, hidden_size_per_layer_input=256,
        global_layers=(4, 9, 14, 19, 24, 29, 34), first_kv_shared_layer=15,
        model_type=model_type,
    )


class TestRegistry:
    def test_new_points_are_registered(self):
        for p in NEW_LAYER:
            assert p in _arch.LAYER_HOOK_POINTS
        for p in NEW_GLOBAL:
            assert p in _arch.GLOBAL_HOOK_POINTS
        assert "final_norm.scale" in _arch.GLOBAL_HOOK_POINTS  # untouched

    def test_names_parse(self):
        a = _arch_e2b_like()
        for p in NEW_LAYER:
            info = parse_hook_name(f"blocks.3.{p}", arch=a)
            assert info.layer == 3 and info.point == p
        for p in NEW_GLOBAL:
            info = parse_hook_name(p, arch=a)
            assert info.layer is None and info.point == p

    def test_selectors(self):
        a = _arch_e2b_like()
        names = {"blocks.2.attn.scores", "blocks.5.mlp.act", "blocks.7.resid_post",
                 "blocks.9.attn.k_pre_rope", "logits"}
        assert attn_internal_layers(names, arch=a) == {2, 9}
        assert mlp_internal_layers(names, arch=a) == {5}

    def test_internal_sets(self):
        assert {"attn.scores", "attn.o_in", "attn.q_pre_rope"} <= _arch.ATTN_INTERNAL_POINTS
        assert _arch.MLP_INTERNAL_POINTS == {"mlp.gate", "mlp.up", "mlp.act", "mlp.down_in"}
        assert "mlp.in_norm" not in _arch.MLP_INTERNAL_POINTS  # both paths dispatch it


class TestFamilySupport:
    def test_canonical_family_supports_everything(self):
        for p in NEW_LAYER:
            assert _arch.family_supports("gemma4", p, layer_scoped=True)
        for p in NEW_GLOBAL:
            assert _arch.family_supports("gemma4", p, layer_scoped=False)

    def test_legacy_families_refuse_the_new_points(self):
        for fam in ("gemma3", "qwen2", "llama"):
            assert _arch.family_supports(fam, "resid_post", layer_scoped=True)
            assert _arch.family_supports(fam, "attn.weights", layer_scoped=True)
            assert not _arch.family_supports(fam, "mlp.act", layer_scoped=True)
            assert not _arch.family_supports(fam, "attn.scores", layer_scoped=True)
            assert not _arch.family_supports(fam, "logits", layer_scoped=False)
            assert not _arch.family_supports(fam, "gate_out", layer_scoped=True)


class _FakeModelForValidation:
    """Just enough of Model to exercise `_validate_hook_names`."""

    def __init__(self, arch):
        self.arch = arch

    from mechbench_compute.model import Model as _M

    _validate_hook_names = _M._validate_hook_names


class TestValidation:
    def test_shared_layer_pre_key_points_are_refused_loudly(self):
        m = _FakeModelForValidation(_arch_e2b_like())
        m._validate_hook_names({"blocks.14.attn.k_pre_rope"})  # last fresh layer: fine
        with pytest.raises(InvalidHookName) as e:
            m._validate_hook_names({"blocks.15.attn.k_pre_rope"})
        assert "KV-shared" in str(e.value)
        with pytest.raises(InvalidHookName):
            m._validate_hook_names({"blocks.20.attn.k_pre_norm"})
        # queries are computed at every layer: allowed on shared layers
        m._validate_hook_names({"blocks.20.attn.q_pre_rope"})

    def test_unimplemented_family_point_is_refused(self):
        m = _FakeModelForValidation(_arch_e2b_like(model_type="qwen2"))
        m._validate_hook_names({"blocks.3.resid_post", "final_norm.scale"})
        with pytest.raises(InvalidHookName) as e:
            m._validate_hook_names({"blocks.3.mlp.act"})
        assert "not implemented" in str(e.value)


# --- opt-in: the real forward ---------------------------------------------------

E2B = "mlx-community/gemma-4-e2b-it-bf16"


def _e2b_cached() -> bool:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(hub, "models--" + E2B.replace("/", "--")))


_REAL = pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1" or not _e2b_cached(),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached to run the real forward",
)


@pytest.fixture(scope="module")
def model():
    from mechbench_compute import Model

    return Model.load(E2B)


@_REAL
class TestRealForward:
    def test_every_new_point_captures_with_the_right_shape(self, model):
        import mlx.core as mx

        from mechbench_compute.distill import encode

        ids = mx.array([encode(model.tokenizer, "The old lighthouse keeper")])
        a = model.arch
        layer = a.last_fresh_kv_global  # a fresh-KV global layer: every point exists
        capture = [f"blocks.{layer}.{p}" for p in NEW_LAYER] + NEW_GLOBAL
        res = model.run(ids, capture=capture)
        L = int(ids.shape[-1])
        c = res.cache
        assert c[f"blocks.{layer}.attn.in_norm"].shape == (1, L, a.d_model)
        assert c[f"blocks.{layer}.attn.q_pre_rope"].shape[0:3] == (1, a.n_heads, L)
        assert c[f"blocks.{layer}.attn.scores"].shape[:3] == (1, a.n_heads, L)
        assert c[f"blocks.{layer}.attn.o_in"].shape[:2] == (1, L)
        F = c[f"blocks.{layer}.mlp.gate"].shape[-1]
        for p in ("mlp.up", "mlp.act", "mlp.down_in"):
            assert c[f"blocks.{layer}.{p}"].shape == (1, L, F)
        assert c["embed"].shape == (1, L, a.d_model)
        assert c["final_norm"].shape == (1, L, a.d_model)
        assert c["logits"].shape == (1, L, a.vocab_size)

    def test_overrides_change_the_output_and_the_untouched_path_is_identical(self, model):
        import mlx.core as mx

        from mechbench_compute.distill import encode

        ids = mx.array([encode(model.tokenizer, "The old lighthouse keeper")])
        base = model.run(ids).logits
        # Capturing an MLP-interior point switches that layer to the manual
        # path: the logits must still match the compiled path closely.
        layer = model.arch.last_fresh_kv_global
        cap = model.run(ids, capture=[f"blocks.{layer}.mlp.act"]).logits
        assert float(mx.abs(cap - base).max()) < 5e-2  # bf16 last-bit class
        # A zero override at mlp.down_in changes the logits.
        zero = model.run(ids, hooks={f"blocks.{layer}.mlp.down_in": lambda x, i: mx.zeros_like(x)}).logits
        assert float(mx.abs(zero - base).max()) > 1e-2
        # A logits override is the last word.
        const = model.run(ids, hooks={"logits": lambda x, i: mx.zeros_like(x)}).logits
        assert float(mx.abs(const).max()) == 0.0
