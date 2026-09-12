"""Trajectories (task 000368): capture along either axis, replay from a
trace, projection, comparison and aggregation — against a stub model
whose residuals are a known function of (layer, token), so every
number is checkable by hand."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import blocks, trajectory
from mechbench_compute.block_params import check_params

N_LAYERS = 4
D = 8


class StubArch:
    n_layers = N_LAYERS
    d_model = D
    n_heads = 2


class StubTokenizer:
    all_special_ids = (0,)

    def decode(self, ids):
        return "".join(f"t{int(i)}" for i in ids)


class StubModel:
    """resid_post[layer][pos] = one-hot(token % D) * (layer + 1); the
    token at a position is 1 + len(word) % 7 when tokenized from text."""

    arch = StubArch()
    tokenizer = StubTokenizer()
    model_id = "stub/model"

    def __init__(self):
        self.seen_ids: list[list[int]] = []

    def tokenize(self, prompt: str, chat_template: bool = True):
        return mx.array([[0] + [1 + (len(w) % 7) for w in prompt.split()]])

    def run(self, ids, interventions=None):
        arr = np.array(ids)[0]
        self.seen_ids.append([int(t) for t in arr])
        cache = {}
        for layer in range(N_LAYERS):
            resid = np.zeros((1, len(arr), D), dtype=np.float32)
            for pos, tok in enumerate(arr):
                resid[0, pos, int(tok) % D] = float(layer + 1)
            cache[f"blocks.{layer}.resid_post"] = mx.array(resid)

        class R:
            pass
        r = R()
        r.cache = cache
        return r

    def project_to_logits(self, residual):
        # identity unembedding onto a D-token vocabulary
        return residual


def onehot(tok: int, scale: float) -> list[float]:
    v = [0.0] * D
    v[tok % D] = scale
    return v


class TestCaptureLayersAxis:
    def test_one_position_at_every_layer(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "user": "hi there"}],
                                 {"axis": "layers", "position": "final"})
        assert out["kind"] == "trajectory" and out["axis"] == "layers"
        assert [r["step"] for r in out["rows"]] == [0, 1, 2, 3]
        assert [r["layer"] for r in out["rows"]] == [0, 1, 2, 3]
        # final token of "hi there" is 1 + 5 % 7 = 6; layer L scales by L+1
        assert out["rows"][2]["vector"] == onehot(6, 3.0)
        assert out["rows"][2]["position"] == 2
        assert out["rows"][2]["norm"] == 3.0

    def test_vocab_top_reads_the_unembedding(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "user": "hi"}],
                                 {"axis": "layers", "layers": [1], "vocab_top": 2})
        top = out["rows"][0]["vocab_top"]
        assert top[0]["token"] == "t3"  # "hi" -> 1 + 2 % 7 = 3, one-hot at 3
        assert 0 < top[0]["p"] <= 1


class TestCapturePositionsAxis:
    def test_one_layer_along_the_sequence_from_text(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "text": "a bb ccc"}],
                                 {"axis": "positions", "layer": 2,
                                  "positions": "all"})
        assert out["axis"] == "positions" and out["layers"] == [2]
        assert out["replay"] == "text"
        assert [r["position"] for r in out["rows"]] == [0, 1, 2, 3]
        assert [r["token"] for r in out["rows"]] == ["t0", "t2", "t3", "t4"]
        assert out["rows"][1]["vector"] == onehot(2, 3.0)

    def test_replays_the_stored_trace_not_the_text(self):
        m = StubModel()
        rec = {"id": "s", "text": "whatever text",
               "trace": {"token_ids": [0, 9, 9, 5, 6, 7],
                         "generation_spans": [{"token_start": 3, "token_end": 6}]}}
        out = trajectory.capture(m, [rec], {"axis": "positions", "layer": 0,
                                            "positions": "generated"})
        assert out["replay"] == "trace"
        assert m.seen_ids == [[0, 9, 9, 5, 6, 7]]  # the trace, verbatim
        # only the generated span: positions 3, 4, 5 -> steps 0, 1, 2
        assert [r["position"] for r in out["rows"]] == [3, 4, 5]
        assert [r["step"] for r in out["rows"]] == [0, 1, 2]
        assert out["rows"][0]["vector"] == onehot(5, 1.0)

    def test_replay_trace_refuses_a_record_without_one(self):
        with pytest.raises(ValueError, match="no trace"):
            trajectory.capture(StubModel(), [{"id": "a", "text": "x"}],
                               {"axis": "positions", "layer": 0,
                                "replay": "trace"})

    def test_range_and_max_steps(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "text": "a b c d e"}],
                                 {"axis": "positions", "layer": 0,
                                  "positions": {"range": [1, 10]},
                                  "max_steps": 3})
        assert [r["position"] for r in out["rows"]] == [1, 2, 3]

    def test_label_from_an_annotated_field(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "text": "x", "hit": 1}],
                                 {"axis": "positions", "layer": 0,
                                  "positions": "all", "label_field": "hit"})
        assert out["rows"][0]["label"] == 1

    def test_needs_a_layer(self):
        with pytest.raises(ValueError, match="needs `layer`"):
            trajectory.capture(StubModel(), [{"id": "a", "text": "x"}],
                               {"axis": "positions"})


def _traj(rows, axis="positions", d=D):
    return {"kind": "trajectory", "axis": axis, "d_model": d, "point": "post",
            "layers": [0], "rows": rows}


def _row(id_, step, vec, label=None):
    return {"id": id_, "label": label, "step": step, "layer": 0,
            "position": step, "vector": vec,
            "norm": float(np.linalg.norm(vec))}


class TestProject:
    def test_scalar_coordinate_along_a_direction(self):
        t = _traj([_row("a", 0, onehot(1, 2.0)), _row("a", 1, onehot(2, 2.0))])
        d = {"kind": "direction", "vector": onehot(1, 1.0), "layer": 0,
             "point": "post", "method": "test"}
        out = trajectory.project({"trajectory": t, "direction": d}, {})
        assert out["kind"] == "trajectory_projection"
        assert [r["coord"] for r in out["rows"]] == [2.0, 0.0]
        assert "vector" not in out["rows"][0]

    def test_dimension_mismatch_is_refused(self):
        t = _traj([_row("a", 0, onehot(1, 1.0))])
        d = {"kind": "direction", "vector": [1.0, 0.0], "layer": 0}
        with pytest.raises(ValueError, match="dims"):
            trajectory.project({"trajectory": t, "direction": d}, {})


class TestCompare:
    def test_per_step_cosine_and_divergence(self):
        a = _traj([_row("x", 0, onehot(1, 1.0)), _row("x", 1, onehot(1, 1.0)),
                   _row("x", 2, onehot(1, 1.0))])
        b = _traj([_row("x", 0, onehot(1, 2.0)), _row("x", 1, onehot(1, 1.0)),
                   _row("x", 2, onehot(3, 1.0))])
        out = trajectory.compare({"a": a, "b": b}, {"threshold": 0.9})
        assert [r["cosine"] for r in out["rows"]] == [1.0, 1.0, 0.0]
        assert out["rows"][0]["norm_ratio"] == 2.0
        assert out["rows"][2]["angle_deg"] == 90.0
        assert out["divergence_step"] == 2 and out["min_cosine_step"] == 2

    def test_axes_must_match(self):
        with pytest.raises(ValueError, match="share an axis"):
            trajectory.compare({"a": _traj([], "layers"),
                                "b": _traj([], "positions")}, {})


class TestAggregate:
    def _labelled(self):
        return _traj([
            _row("s1", 0, onehot(1, 1.0), "lh"), _row("s1", 1, onehot(1, 3.0), "lh"),
            _row("s2", 0, onehot(1, 3.0), "lh"), _row("s2", 1, onehot(1, 5.0), "lh"),
            _row("s3", 0, onehot(2, 1.0), "other"),
            _row("s3", 1, onehot(2, 1.0), "other"),
        ])

    def test_per_step_mean_vectors_by_label(self):
        out = trajectory.aggregate({"trajectory": self._labelled()},
                                   {"by": "label"})
        rows = {(r["group"], r["step"]): r for r in out["rows"]}
        assert rows[("lh", 0)]["vector"] == onehot(1, 2.0)
        assert rows[("lh", 1)]["vector"] == onehot(1, 4.0)
        assert rows[("lh", 0)]["n"] == 2 and rows[("lh", 0)]["spread"] == 1.0

    def test_window_as_vectors_feeds_from_vectors(self):
        from mechbench_compute import directions as dirs
        out = trajectory.aggregate({"trajectory": self._labelled()},
                                   {"by": "label", "as": "vectors",
                                    "steps": {"range": [0, 2]}})
        assert out["kind"] == "residual_vectors" and out["layers"] == [0]
        by = {r["label"]: r for r in out["rows"]}
        assert by["lh"]["vector"] == onehot(1, 3.0)  # mean of 1,3,3,5
        assert by["lh"]["n_pooled"] == 4
        # the direction algebra reads it unchanged
        d = dirs.from_vectors(out, layer=0, positive="lh", negative="other")
        assert d["provenance"]["method"] == "diff_of_means"
        v = np.asarray(d["vector"])
        assert v[1] > 0 and v[2] < 0

    def test_window_over_projected_coords_by_id(self):
        t = _traj([{"id": "s1", "step": 0, "layer": 0, "position": 0, "coord": 1.0},
                   {"id": "s1", "step": 1, "layer": 0, "position": 1, "coord": 3.0},
                   {"id": "s2", "step": 0, "layer": 0, "position": 0, "coord": 5.0}])
        t["kind"] = "trajectory_projection"
        out = trajectory.aggregate({"trajectory": t}, {"by": "id", "as": "window"})
        by = {r["group"]: r for r in out["rows"]}
        assert by["s1"]["mean"] == 2.0 and by["s1"]["n"] == 2
        assert by["s2"]["mean"] == 5.0

    def test_vectors_mode_refuses_a_projection(self):
        t = _traj([{"id": "a", "step": 0, "layer": 0, "position": 0, "coord": 1.0}])
        with pytest.raises(ValueError, match="vector rows"):
            trajectory.aggregate({"trajectory": t}, {"as": "vectors"})


class TestWiring:
    def test_pure_blocks_are_registered(self):
        for ref in ("~canonical/ops/trajectory/project/1",
                    "~canonical/ops/trajectory/compare/1",
                    "~canonical/ops/trajectory/aggregate/1"):
            assert ref in blocks.PURE_BLOCKS

    def test_params_are_guarded(self):
        with pytest.raises(ValueError, match="does not accept"):
            check_params("~canonical/ops/trajectory/capture/1", {"nope": 1})
        check_params("~canonical/ops/trajectory/capture/1",
                     {"axis": "positions", "layer": 12, "replay": "auto"})

    def test_select_reads_an_annotated_field(self):
        recs = [{"id": "a", "coords": {"p": "flash"}, "hit": 1},
                {"id": "b", "coords": {"p": "flash"}, "hit": 0}]
        assert [r["id"] for r in blocks.select(recs, {"where": {"hit": 1}})] == ["a"]
        assert len(blocks.select(recs, {"where": {"p": "flash"}})) == 2

    def test_union_of_vector_records_stays_a_vector_record(self):
        # base and adapted captures come from two model nodes; their
        # union must still be what direction/from-vectors reads.
        def vec(label_rows):
            return {"kind": "residual_vectors", "point": "post", "source": "resid",
                    "position": "final", "layers": [12], "d_model": D,
                    "template": "chat", "rows": label_rows}
        base = vec([{"id": "flash", "layer": 12, "vector": onehot(1, 1.0)}])
        adapted = vec([{"id": "flash", "layer": 12, "vector": onehot(2, 1.0)}])
        out = blocks.union({"base": base, "adapted": adapted}, {})
        assert out["kind"] == "residual_vectors" and out["layers"] == [12]
        assert [r["label"] for r in out["rows"]] == ["adapted", "base"]  # port order
        from mechbench_compute import directions as dirs
        d = dirs.from_vectors(out, layer=12, positive="base", negative="adapted")
        v = np.asarray(d["vector"])
        assert v[1] > 0 and v[2] < 0

    def test_union_of_plain_records_is_unchanged(self):
        out = blocks.union({"a": [{"id": "1"}], "b": [{"id": "2"}]}, {})
        assert out["kind"] == "record_set"
        assert [r["coords"]["batch"] for r in out["records"]] == ["a", "b"]

    def test_text_stats_keep_retains_the_item(self):
        items = {"items": [{"id": "s1", "text": "The old lighthouse keeper sat.",
                            "trace": {"token_ids": [1, 2, 3]}}]}
        params = {"mode": "annotate", "keep": True,
                  "measures": [{"kind": "pattern", "name": "opening",
                                "where": "prefix", "ignore_case": True,
                                "patterns": [r"the old lighthouse"]}]}
        rows = blocks.text_stats({"documents": items}, params)
        assert rows[0]["opening"] == 1
        assert rows[0]["trace"] == {"token_ids": [1, 2, 3]}  # kept for capture
        assert rows[0]["text"].startswith("The old")


class TestFirstKPool:
    def test_windowed_pool_reads_the_opening(self):
        from mechbench_compute.interp import _pool_spec, _pooled
        spec = _pool_spec({"pool": "first_k", "pool_k": 2, "pool_skip": 1})
        mat = np.array([[0.0], [1.0], [3.0], [10.0]])
        v, n = _pooled(mat, spec)
        assert n == 2 and v[0] == 2.0  # positions 1 and 2

    def test_first_k_needs_k(self):
        from mechbench_compute.interp import _pool_spec
        with pytest.raises(ValueError, match="pool_k"):
            _pool_spec({"pool": "first_k"})
