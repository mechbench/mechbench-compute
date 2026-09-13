"""The declarative intervene block (task 000366): the hook math on fake
activations, spec parsing, the readout plumbing on a fake model, and
an opt-in real-E2B check."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import directions as dirs
from mechbench_compute import intervene as iv


def _dir(vec, layer=2, point="resid_post"):
    return dirs.make(vec, layer=layer, point=point, method="t")


def _spec(**kw):
    item = {"point": "resid_post", "layers": [2], "positions": "last", "op": "zero"}
    item.update(kw)
    return iv.Spec(item, n_layers=4, seed=0)


def _act(B=1, L=3, D=4, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.normal(size=(B, L, D)).astype(np.float32))


def _apply(spec, act, tokens=("a", "b", "c"), layer=2):
    fn = spec.build(layer, list(tokens))
    out = fn(act, None)
    mx.eval(out)
    return np.array(out)


class TestOps:
    def test_zero_last_position_only(self):
        a = _act()
        out = _apply(_spec(op="zero"), a)
        assert np.all(out[0, -1] == 0) and np.allclose(out[0, :-1], np.array(a)[0, :-1])

    def test_scale_all(self):
        a = _act()
        out = _apply(_spec(op="scale", strength=2.0, positions="all"), a)
        assert np.allclose(out, 2 * np.array(a))

    def test_add_direction(self):
        a = _act()
        d = _dir([1, 0, 0, 0])
        out = _apply(_spec(op="add", strength=3.0, direction=d), a)
        assert np.allclose(out[0, -1], np.array(a)[0, -1] + [3, 0, 0, 0], atol=1e-5)

    def test_project_out_kills_the_component(self):
        a = _act()
        d = _dir([0, 1, 0, 0])
        out = _apply(_spec(op="project_out", direction=d), a)
        assert abs(out[0, -1, 1]) < 1e-6 and np.allclose(out[0, -1, [0, 2, 3]], np.array(a)[0, -1, [0, 2, 3]])

    def test_clamp_bounds_the_projection(self):
        a = mx.array(np.array([[[5.0, 1.0, 0.0, 0.0], [-5.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]], np.float32))
        d = _dir([1, 0, 0, 0])
        out = _apply(_spec(op="clamp", strength=1.0, direction=d, positions="all"), a)
        assert np.allclose(out[0, :, 0], [1.0, -1.0, 0.5])
        assert out[0, 0, 1] == 1.0  # orthogonal component untouched

    def test_rotate_quarter_turn_in_the_plane(self):
        a = mx.array(np.array([[[1.0, 0.0, 0.0, 0.0]]], np.float32))
        d1, d2 = _dir([1, 0, 0, 0]), _dir([0, 1, 0, 0])
        out = _apply(_spec(op="rotate", strength=np.pi / 2, direction=d1, direction2=d2), a)
        assert np.allclose(out[0, 0], [0.0, 1.0, 0.0, 0.0], atol=1e-6)

    def test_mean_and_resample_from_source(self):
        rows = [{"layer": 2, "vector": [1, 1, 1, 1]}, {"layer": 2, "vector": [3, 3, 3, 3]}]
        src = {"kind": "residual_vectors", "rows": rows}
        a = _act()
        out = _apply(_spec(op="mean", source=src), a)
        assert np.allclose(out[0, -1], [2, 2, 2, 2])
        out2 = _apply(_spec(op="resample", source=src, seed=3), a)
        assert np.allclose(out2[0, -1], [1, 1, 1, 1]) or np.allclose(out2[0, -1], [3, 3, 3, 3])

    def test_neurons_and_heads_select_features(self):
        a = _act(D=4)
        out = _apply(_spec(op="zero", positions="all", neurons=[1, 3]), a)
        assert np.all(out[0, :, [1, 3]] == 0) and np.allclose(out[0, :, [0, 2]], np.array(a)[0, :, [0, 2]])
        heads_act = mx.array(np.ones((1, 2, 3, 4), np.float32))  # [B, n_heads, L, hd]
        spec = _spec(point="attn.per_head_out", op="zero", positions="all", heads=[1])
        out = _apply(spec, heads_act)
        assert np.all(out[0, 1] == 0) and np.all(out[0, 0] == 1)

    def test_condition_gates_on_projection(self):
        a = mx.array(np.array([[[2.0, 0, 0, 0], [-2.0, 0, 0, 0], [2.0, 0, 0, 0]]], np.float32))
        cond = {"direction": _dir([1, 0, 0, 0]), "threshold": 0.0, "above": True}
        out = _apply(_spec(op="zero", positions="all", condition=cond), a)
        assert np.all(out[0, 0] == 0) and np.all(out[0, 2] == 0) and out[0, 1, 0] == -2.0

    def test_token_positions(self):
        a = _act(L=3)
        out = _apply(_spec(op="zero", positions={"tokens": ["b"]}), a, tokens=("a", " b", "c"))
        assert np.all(out[0, 1] == 0) and np.allclose(out[0, 0], np.array(a)[0, 0])


class TestSpecParsing:
    def test_bad_point_op_layer(self):
        with pytest.raises(iv.SpecError):
            _spec(point="resid_nope")
        with pytest.raises(iv.SpecError):
            _spec(op="explode")
        with pytest.raises(iv.SpecError):
            _spec(layers=[9])

    def test_direction_ops_need_a_direction(self):
        with pytest.raises(iv.SpecError):
            _spec(op="add")

    def test_global_points_have_no_layers(self):
        s = _spec(point="logits", op="zero")
        assert s.layers == [None] and s.hook_names() == ["logits"]

    def test_hook_names(self):
        assert _spec(layers=[1, 3]).hook_names() == ["blocks.1.resid_post", "blocks.3.resid_post"]

    def test_width_mismatch_is_loud_at_call_time(self):
        with pytest.raises(iv.SpecError):
            _apply(_spec(op="add", direction=_dir([1, 0])), _act(D=4))


class _FakeTok:
    def decode(self, ids):
        return {1: "The", 2: " old", 3: " light"}.get(int(ids[0]), f"t{ids[0]}")


class _FakeArch:
    n_layers = 4


class _FakeModel:
    """A model whose 'logits' are a fixed function of the residual at
    blocks.2.resid_post, so an intervention changes the readout."""

    tokenizer = _FakeTok()
    arch = _FakeArch()

    def tokenize(self, prompt, chat_template=False):
        return mx.array([[1, 2, 3]])

    def run(self, ids, hooks=None, capture=None, interventions=None):
        from mechbench_compute.interventions import compose

        hooks_d, caps = compose(interventions, hooks=hooks, capture=capture)
        L = int(ids.shape[-1])
        h = mx.array(np.tile(np.arange(4, dtype=np.float32), (1, L, 1)))  # [1, L, 4]
        fn = hooks_d.get("blocks.2.resid_post")
        if fn is not None:
            out = fn(h, None)
            h = out if out is not None else h
        # logits: 6-token vocab, a linear read of the residual
        W = mx.array(np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], np.float32))
        logits = h @ W
        cache = {}
        if "blocks.2.resid_post" in caps:
            cache["blocks.2.resid_post"] = h

        class R:
            pass

        r = R(); r.logits = logits; r.cache = cache
        mx.eval(logits)
        return r


class TestRunReadout:
    def test_sweep_with_control_and_items(self):
        model = _FakeModel()
        d = _dir([1, 0, 0, 0])
        items = []
        out = iv.run(model, [{"id": "r1", "user": "hi"}],
                     {"spec": [{"point": "resid_post", "layers": [2], "op": "add",
                                "strength": 10.0, "direction": d}],
                      "sweep": {"strength": [1.0]}, "top_k": 2},
                     on_item=lambda k, row: items.append(k))
        assert out["kind"] == "intervene_readout" and out["sweep"] == [0.0, 1.0]
        assert items == ["r1:0.0", "r1:1.0"]
        ctrl, steered = out["rows"]
        assert ctrl["factor"] == 0.0 and steered["factor"] == 1.0
        # the control residual is [0, 1, 2, 3], so its top token is id 3
        # (" light" in the fake tokenizer); adding +10 along coordinate 0
        # makes id 0 ("t0") the top token
        assert ctrl["top"][0]["token"] == " light"
        assert steered["top"][0]["token"] == "t0"
        assert steered["entropy_bits"] < ctrl["entropy_bits"]
        assert out["spec"][0]["direction"]["derivation"]["method"] == "t"  # wire form, no vector

    def test_capture_readout(self):
        model = _FakeModel()
        d = _dir([0, 1, 0, 0])
        out = iv.run(model, [{"id": "r1", "user": "hi"}],
                     {"spec": [{"point": "resid_post", "layers": [2], "op": "project_out", "direction": d}],
                      "readout": {"kind": "capture", "points": ["blocks.2.resid_post"]}})
        ctrl, done = out["rows"]
        assert ctrl["captures"]["blocks.2.resid_post"][1] == 1.0
        assert abs(done["captures"]["blocks.2.resid_post"][1]) < 1e-6

    def test_direction_by_port_fills_the_spec(self):
        model = _FakeModel()
        out = iv.run(model, [{"id": "r1", "user": "hi"}],
                     {"spec": [{"point": "resid_post", "layers": [2], "op": "add", "strength": 5.0}],
                      "control": False},
                     inputs={"direction": _dir([1, 0, 0, 0])})
        assert out["rows"][0]["top"][0]["token"] == "t0"

    def test_empty_spec_refused(self):
        with pytest.raises(iv.SpecError):
            iv.run(_FakeModel(), [{"id": "r1", "user": "hi"}], {"spec": []})


E2B = "mlx-community/gemma-4-e2b-it-bf16"


@pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1"
    or not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--" + E2B.replace("/", "--"))),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached",
)
def test_real_project_out_zeroes_the_projection_at_the_point():
    from mechbench_compute import Model

    model = Model.load(E2B)
    layer = model.arch.last_fresh_kv_global
    # a direction from the model's own residual: capture, then project it out
    base = model.run(model.tokenize("The old lighthouse keeper", chat_template=False),
                     capture=[f"blocks.{layer}.resid_post"])
    v = np.array(base.cache[f"blocks.{layer}.resid_post"][0, -1].astype(mx.float32))
    d = dirs.make(v, layer=layer, point="resid_post", method="self")
    out = iv.run(model, [{"id": "lh", "user": "The old lighthouse keeper"}],
                 {"spec": [{"point": "resid_post", "layers": [layer], "op": "project_out", "direction": d}],
                  "readout": {"kind": "capture", "points": [f"blocks.{layer}.resid_post"]}})
    ctrl, done = out["rows"]
    u = np.array(d["vector"], np.float32)
    proj_ctrl = float(np.array(ctrl["captures"][f"blocks.{layer}.resid_post"]) @ u)
    proj_done = float(np.array(done["captures"][f"blocks.{layer}.resid_post"]) @ u)
    assert abs(proj_ctrl) > 1.0 and abs(proj_done) < 0.05 * abs(proj_ctrl)
