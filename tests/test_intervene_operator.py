from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import directions as dirs
from mechbench_compute import generate as generate_mod
from mechbench_compute import intervene as iv
from mechbench_compute import shapes as S
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.expr import engine as expr_engine
from mechbench_compute.hooks import HookInfo
from mechbench_compute.intervene.compile_operator import compile_operator
from mechbench_compute.intervene.operator_refused import OperatorRefused
from mechbench_compute.interventions import Ablate, _PositionAdd
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops import Context
from mechbench_compute.ops.intervene.apply import run_intervene
from mechbench_compute.ops.text import generate as generate_op
from tests.tiny_models import MODEL_TYPES, build_tiny_model

IDS = mx.array([[1, 5, 9, 2, 7, 3, 11, 4]])

TOKENS = [f"t{i}" for i in range(IDS.shape[1])]

POINT = "blocks.1.resid_post"

RECORD = {"id": "r", "user": "the cat sat on a mat"}


@pytest.fixture(scope="module", params=MODEL_TYPES)
def tiny(request):
    return build_tiny_model(request.param, BY_MODEL_TYPE[request.param])


def run_items(model, items, capture=(), inputs=None):
    compiled = iv.compile(model, items, inputs=inputs)
    return model.run(IDS, interventions=[iv.SpecIntervention(compiled.specs, TOKENS)],
                     capture=list(capture))


def read_rows(result, name=POINT) -> np.ndarray:
    return np.array(result.cache[name].astype(mx.float32))[0]


def make_direction(vec, layer=1, point="resid_post"):
    return dirs.make(vec, S.space(model=None, layer=layer, point=point, d=len(vec)), method="t")


def read_width(model) -> int:
    return int(model.arch.d_model)


def refuse(code, run):
    with pytest.raises(OperatorRefused) as e:
        run()
    assert e.value.code == code and str(e.value).startswith(f"{code}: ")
    return e.value


class TestTheFixedOpsAreItsCases:
    def test_zero_on_a_layers_attn_out_is_ablate_attention(self, tiny):
        mine = run_items(tiny, [{"point": "attn_out", "layers": [1], "positions": "all", "f": "0"}],
                         capture=["blocks.1.attn_out", "blocks.3.resid_post"])
        theirs = tiny.run(IDS, interventions=[Ablate.attention(1)],
                          capture=["blocks.1.attn_out", "blocks.3.resid_post"])
        assert mx.array_equal(mine.logits, theirs.logits)
        assert mx.array_equal(mine.cache["blocks.3.resid_post"], theirs.cache["blocks.3.resid_post"])
        assert not np.any(read_rows(mine, "blocks.1.attn_out"))

    def test_x_plus_v_at_one_position_is_the_position_add(self, tiny):
        v = np.random.default_rng(3).normal(size=read_width(tiny)).astype(np.float32)
        mine = run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": [3], "f": "x + v",
                                 "constants": {"v": v.tolist()}}], capture=[POINT])
        theirs = tiny.run(IDS, interventions=[_PositionAdd(1, 3, mx.array(v), 1.0, "resid_post")],
                          capture=[POINT])
        assert mx.array_equal(mine.logits, theirs.logits)
        assert mx.array_equal(mine.cache[POINT], theirs.cache[POINT])

    def test_zero_on_a_heads_output_is_ablate_head(self, tiny):
        mine = run_items(tiny, [{"point": "attn.per_head_out", "layers": [1], "heads": [2],
                                 "positions": "all", "f": "0"}])
        theirs = tiny.run(IDS, interventions=[Ablate.head(1, 2)])
        assert mx.array_equal(mine.logits, theirs.logits)

    def test_a_constant_bound_from_a_source_mean_is_mean_ablation(self, tiny):
        rng = np.random.default_rng(5)
        rows = [S.vector(rng.normal(size=read_width(tiny)).astype(np.float32),
                         S.space(model="tiny", layer=1, point="resid_post", d=read_width(tiny)), id=f"r{i}")
                for i in range(3)]
        source = K.collection("activations/vector", rows)
        mine = run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "last", "f": "m",
                                 "constants": {"m": {"source": "mean"}}}], inputs={"source": source})
        theirs = run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "last",
                                   "op": "mean"}], inputs={"source": source})
        assert mx.array_equal(mine.logits, theirs.logits)


class TestWhatTheMaskSelects:
    def test_k_times_x_on_dimensions_changes_only_those_and_by_the_factor(self, tiny):
        dims = [1, 5, 17]
        clean = read_rows(tiny.run(IDS, capture=[POINT]))
        out = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "all",
                                          "mask": dims, "f": "k * x", "constants": {"k": 2.5}}],
                                  capture=[POINT]))
        rest = [i for i in range(read_width(tiny)) if i not in dims]
        assert np.array_equal(out[:, dims], clean[:, dims] * np.float32(2.5))
        assert np.array_equal(out[:, rest], clean[:, rest])

    def test_x_squared_plus_2x_on_a_mask_is_the_hand_computation(self, tiny):
        dims = [0, 3, 7]
        clean = read_rows(tiny.run(IDS, capture=[POINT]))
        out = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "all",
                                          "mask": dims, "f": "x**2 + 2*x"}], capture=[POINT]))
        x = clean[:, dims]
        assert np.array_equal(out[:, dims], x * x + np.float32(2) * x)
        assert np.array_equal(np.delete(out, dims, axis=1), np.delete(clean, dims, axis=1))

    def test_min_of_x_and_c_clamps_from_above(self, tiny):
        clean = read_rows(tiny.run(IDS, capture=[POINT]))
        c = float(np.median(clean[:, 4]))
        out = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "all",
                                          "mask": [4], "f": "min(x, c)", "constants": {"c": c}}],
                                  capture=[POINT]))
        assert np.array_equal(out[:, 4], np.minimum(clean[:, 4], np.float32(c)))

    def test_a_direction_mask_scales_only_the_projection(self, tiny):
        direction = make_direction(np.random.default_rng(7).normal(size=read_width(tiny)))
        clean = read_rows(tiny.run(IDS, capture=[POINT])).astype(np.float64)
        out = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "all",
                                          "mask": direction, "f": "k * x",
                                          "constants": {"k": 3.0}}], capture=[POINT])).astype(np.float64)
        u64 = np.asarray(direction["vector"], np.float64)
        u64 /= np.linalg.norm(u64)
        p, q = clean @ u64, out @ u64
        tol = 1e-5 * max(1.0, float(np.abs(clean).max()))
        assert np.allclose(q, 3.0 * p, rtol=1e-5, atol=tol)
        assert np.allclose(out - np.outer(q, u64), clean - np.outer(p, u64), rtol=0, atol=tol)

    def test_a_frame_replaces_its_coordinates_and_passes_the_rest(self, tiny):
        rng = np.random.default_rng(9)
        d = read_width(tiny)
        b1, b2 = rng.normal(size=d).astype(np.float32), rng.normal(size=d).astype(np.float32)
        frame = K.collection("direction/vector", [make_direction(b1), make_direction(b1 + b2)])
        clean = read_rows(tiny.run(IDS, capture=[POINT])).astype(np.float64)
        same = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "all",
                                           "mask": frame, "f": "x"}], capture=[POINT]))
        assert np.array_equal(same, clean.astype(np.float32))
        pair = [make_direction(b1), make_direction(b2)]
        gone = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": "all",
                                           "mask": pair, "f": "0"}], capture=[POINT])).astype(np.float64)
        basis = np.stack([np.asarray(d["vector"], np.float64) for d in pair])
        tol = 1e-5 * max(1.0, float(np.abs(clean).max()))
        assert np.allclose(gone @ basis.T, 0.0, atol=tol)
        coords = np.linalg.lstsq(basis.T, clean.T, rcond=None)[0].T
        assert np.allclose(gone, clean - coords @ basis, atol=tol)

    def test_except_inverts_a_dimension_mask_as_it_inverts_the_named_layers(self):
        spec = iv.Spec({"point": "resid_post", "layers": [1], "positions": "all", "mask": [2],
                        "except": True, "f": "0"}, n_layers=4, seed=0)
        act = mx.array(np.arange(1, 13, dtype=np.float32).reshape(1, 3, 4))
        out = np.array(spec.build(0, ["a", "b", "c"])(act, None))
        assert spec.layers == [0, 2, 3]
        assert np.array_equal(out[0, :, 2], np.array(act)[0, :, 2])
        assert not np.any(np.delete(out[0], 2, axis=1))


class TestPositionsAreHonoured:
    def test_a_prompt_position_is_the_only_one_touched(self, tiny):
        clean = read_rows(tiny.run(IDS, capture=[POINT]))
        out = read_rows(run_items(tiny, [{"point": "resid_post", "layers": [1], "positions": [2],
                                          "f": "-x"}], capture=[POINT]))
        assert np.array_equal(out[2], -clean[2])
        assert np.array_equal(np.delete(out, 2, axis=0), np.delete(clean, 2, axis=0))

    def test_a_decoding_step_is_the_one_acted_at(self, tiny, monkeypatch):
        monkeypatch.setattr(generate_mod, "_stop_ids", lambda tokenizer: set())

        def generate(item):
            spec = {"spec": [item], "control": False} if item else {}
            out = generate_op.run(Context(loaded=tiny), {"records": [RECORD]},
                                  {"model": "tiny", "temperature": 0.0, "max_tokens": 6,
                                   "fidelity": "trace", **spec})
            row = out["items"][0]
            start = row["trace"]["generation_spans"][0]["token_start"]
            return row, row["trace"]["token_ids"][start:]

        _, plain = generate(None)
        row, flipped = generate({"point": "logits", "positions": {"step": 2}, "f": "-x"})
        _, sugar = generate({"point": "logits", "positions": {"step": 2}, "op": "scale",
                             "strength": -1.0})
        assert flipped == sugar
        assert flipped[:2] == plain[:2] and flipped[2] != plain[2]
        assert row["metadata"]["intervention"] == {"steps": [2]}


class TestARefusalNamesItsCode:
    def test_an_unknown_function(self, tiny):
        e = refuse("OPERATOR_FUNCTION_UNKNOWN",
                   lambda: iv.compile(tiny, [{"point": "resid_post", "layers": [1], "f": "foo(x)"}]))
        assert e.construct == "foo" and "abs, ceil, exp" in str(e)
        assert "a if c else b" in str(refuse("OPERATOR_FUNCTION_UNKNOWN", lambda: compile_operator(
            "where(x > 0, x, 0)")))

    def test_a_dimension_beyond_the_width(self, tiny):
        d = read_width(tiny)
        e = refuse("MASK_OUT_OF_RANGE", lambda: run_items(
            tiny, [{"point": "resid_post", "layers": [1], "mask": [0, d], "f": "0"}]))
        assert e.construct == str(d) and f"0 through {d - 1}" in str(e)

    def test_a_caret_is_not_power(self):
        e = refuse("OPERATOR_SYNTAX", lambda: compile_operator("x^2 + 2*x"))
        assert e.construct == "^" and "power is `**`" in str(e)

    @pytest.mark.parametrize(("code", "item"), [
        ("OPERATOR_UNSUPPORTED", {"f": "x.y"}),
        ("OPERATOR_UNSUPPORTED", {"f": "x if x in [1, 2] else 0"}),
        ("OPERATOR_UNSUPPORTED", {"f": "min(x)"}),
        ("OPERATOR_TYPE", {"f": "x > 0"}),
        ("OPERATOR_TYPE", {"f": "exp(x, 2)"}),
        ("OPERATOR_NAME_UNBOUND", {"f": "k * x"}),
        ("OPERATOR_FIELDS", {"f": "0", "op": "zero"}),
        ("OPERATOR_FIELDS", {"f": "0", "neurons": [1]}),
        ("OPERATOR_FIELDS", {"mask": [1]}),
        ("OPERATOR_SYNTAX", {"f": ["x"]}),
        ("CONSTANT_INVALID", {"f": "x + k", "constants": {"k": "two"}}),
        ("CONSTANT_INVALID", {"f": "x + k", "constants": {"k": {"source": "mean"}}}),
        ("CONSTANT_INVALID", {"f": "x", "constants": {"x": 1.0}}),
        ("MASK_INVALID", {"f": "0", "mask": []}),
        ("MASK_INVALID", {"f": "0", "mask": [1, 1]}),
        ("MASK_INVALID", {"f": "0", "mask": "direction"}),
        ("MASK_DEGENERATE", {"f": "0", "mask": [make_direction([1.0, 0, 0]),
                                                make_direction([2.0, 0, 0])]}),
    ])
    def test_at_compile(self, code, item):
        refuse(code, lambda: iv.Spec({"point": "resid_post", "layers": [1], **item}, n_layers=4, seed=0))

    def test_an_operator_does_not_edit_a_weight(self, tiny):
        refuse("OPERATOR_FIELDS", lambda: iv.compile(
            tiny, [{"parameter": "layers.0.self_attn.o_proj", "f": "0"}]))

    def test_at_run(self, tiny):
        d = read_width(tiny)
        refuse("MASK_WIDTH_MISMATCH", lambda: run_items(
            tiny, [{"point": "resid_post", "layers": [1], "mask": make_direction([1.0] * (d + 1)),
                    "f": "0"}]))
        refuse("CONSTANT_INVALID", lambda: run_items(
            tiny, [{"point": "resid_post", "layers": [1], "mask": [1, 2], "f": "x + v",
                    "constants": {"v": [1.0, 2.0, 3.0]}}]))
        e = refuse("OPERATOR_UNDEFINED", lambda: run_items(
            tiny, [{"point": "resid_post", "layers": [1], "positions": "all", "f": "log(x)"}]))
        assert e.construct == "log(x)"


class TestTheRecord:
    def test_the_header_and_the_sentence_record_the_operator(self, tiny):
        u = np.zeros(read_width(tiny), np.float32)
        u[0] = 1.0
        out = run_intervene(tiny, [RECORD], {"spec": [
            {"point": "resid_post", "layers": [1], "mask": [5], "f": "k*x", "constants": {"k": 2}},
            {"point": "attn_out", "layers": [2], "mask": make_direction(u, layer=2, point="attn_out"),
             "f": "min(x,c)", "constants": {"c": 0.5}}]})
        first, second = out["spec"]
        assert first["f"] == "k * x" and first["mask"] == [5] and first["constants"] == {"k": 2}
        assert second["f"] == "min(x, c)"
        assert second["mask"]["kind"] == "direction/vector" and "vector" not in second["mask"]
        assert "f(x) = k * x at resid_post" in out["description"]
        assert "f(x) = min(x, c) at attn_out" in out["description"]

    def test_the_direction_port_fills_a_mask_that_asks_for_it(self, tiny):
        u = np.random.default_rng(1).normal(size=read_width(tiny)).astype(np.float32)
        item = {"point": "resid_post", "layers": [1], "positions": "all", "f": "0"}
        port = run_items(tiny, [{**item, "mask": "direction"}], inputs={"direction": make_direction(u)})
        inline = run_items(tiny, [{**item, "mask": make_direction(u)}])
        assert mx.array_equal(port.logits, inline.logits)

    def test_f_is_compiled_once_per_node_not_per_token(self, tiny, monkeypatch):
        monkeypatch.setattr(generate_mod, "_stop_ids", lambda tokenizer: set())
        compile_operator.cache_clear()
        seen = []
        check = expr_engine.Engine.check

        def counted(self, expr):
            seen.append(expr)
            return check(self, expr)

        monkeypatch.setattr(expr_engine.Engine, "check", counted)
        out = generate_op.run(Context(loaded=tiny), {"records": [RECORD]},
                              {"model": "tiny", "temperature": 0.0, "max_tokens": 6,
                               "spec": [{"point": "resid_post", "layers": [1], "positions": "all",
                                         "f": "x * 0.5 + 1"}]})
        assert len(out["items"]) == 2 and seen == ["x * 0.5 + 1"]


class TestAStrength:
    def test_scales_the_edit_and_one_is_f(self):
        spec = iv.Spec({"point": "resid_post", "layers": [1], "positions": "all", "f": "x + 4"},
                       n_layers=4, seed=0)
        act = mx.array(np.arange(12, dtype=np.float32).reshape(1, 3, 4))
        info = HookInfo(name="blocks.1.resid_post", layer=1, point="resid_post")
        full = np.array(spec.build(1, ["a", "b", "c"])(act, info))
        half = np.array(iv.scale_specs([spec], 0.5)[0].build(1, ["a", "b", "c"])(act, info))
        assert np.array_equal(full, np.array(act) + 4) and np.array_equal(half, np.array(act) + 2)
