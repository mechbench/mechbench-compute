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

    def encode(self, text, add_special_tokens=True):
        return [0] + [1 + (len(w) % 7) for w in text.split()]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages[-1]["content"]

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
        assert out["item_kind"] == "trajectory/point" and out["axis"] == "layers"
        assert [r["step"] for r in out["items"]] == [0, 1, 2, 3]
        assert [r["space"]["layer"] for r in out["items"]] == [0, 1, 2, 3]
        assert out["items"][2]["space"] == {"model": "stub/model", "layer": 2,
                                            "point": "resid_post", "head": None, "d": D}
        # final token of "hi there" is 1 + 5 % 7 = 6; layer L scales by L+1
        assert out["items"][2]["vector"] == onehot(6, 3.0)
        assert out["items"][2]["position"] == 2
        assert out["items"][2]["norm"] == 3.0
        assert out["items"][2]["token"] == {"id": 6, "text": "t6"}

    def test_vocab_top_reads_the_unembedding(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "user": "hi"}],
                                 {"axis": "layers", "layers": [1], "vocab_top": 2})
        vocab = out["items"][0]["vocab"]
        top = vocab["top"]
        assert top[0]["token"]["text"] == "t3"  # "hi" -> 1 + 2 % 7 = 3, one-hot at 3
        assert 0 < top[0]["p"] <= 1 and top[0]["logp"] <= 0
        assert len(top) == 2 and vocab["entropy_bits"] >= 0


class TestCapturePositionsAxis:
    def test_one_layer_along_the_sequence_from_text(self):
        m = StubModel()
        out = trajectory.capture(m, [{"id": "a", "text": "a bb ccc"}],
                                 {"axis": "positions", "layer": 2,
                                  "positions": "all"})
        assert out["axis"] == "positions" and out["layers"] == [2]
        assert out["replay"] == "text"
        assert [r["position"] for r in out["items"]] == [0, 1, 2, 3]
        assert [r["token"]["text"] for r in out["items"]] == ["t0", "t2", "t3", "t4"]
        assert out["items"][1]["vector"] == onehot(2, 3.0)

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
        assert [r["position"] for r in out["items"]] == [3, 4, 5]
        assert [r["step"] for r in out["items"]] == [0, 1, 2]
        assert out["items"][0]["vector"] == onehot(5, 1.0)

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
        assert [r["position"] for r in out["items"]] == [1, 2, 3]

    def test_a_measurement_groups_once_it_is_a_coordinate(self):
        # `text/stats` writes a hit as a field; `records/rename` moves it
        # into coords, and from there every item carries it.
        from mechbench_compute.blocks import rename

        m = StubModel()
        recs = rename([{"id": "a", "text": "x", "hit": 1}], {"fields": {"hit": "coords.hit"}})
        assert recs == [{"id": "a", "text": "x", "coords": {"hit": 1}}]
        out = trajectory.capture(m, recs, {"axis": "positions", "layer": 0,
                                           "positions": "all"})
        assert out["items"][0]["coords"] == {"hit": 1}

    def test_needs_a_layer(self):
        with pytest.raises(ValueError, match="needs `layer`"):
            trajectory.capture(StubModel(), [{"id": "a", "text": "x"}],
                               {"axis": "positions"})

    def test_reduce_mean_over_a_step_window(self):
        # positions 1..3 of "a bb ccc dddd": tokens 2,3,4 at layer 0 (scale 1)
        m = StubModel()
        pool = {"reduce": "mean", "over": {"range": [1, 4]}}
        out = trajectory.capture(m, [{"id": "a", "text": "a bb ccc dddd"}],
                                 {"axis": "positions", "layer": 0, "positions": "all",
                                  "pool": pool})
        assert out["pool"] == pool and len(out["items"]) == 1
        row = out["items"][0]
        assert row["n_pooled"] == 3 and row["pool"] == pool
        v = row["vector"]  # rounded to 5 places on the wire
        third = pytest.approx(1 / 3, abs=1e-4)
        assert v[2] == third and v[3] == third and v[4] == third
        assert v[1] == 0.0

    def test_project_at_capture_time_emits_coords_only(self):
        m = StubModel()
        d = {"kind": "direction", "vector": onehot(3, 1.0), "layer": 0,
             "point": "post", "derivation": {"method": "test"}}
        out = trajectory.capture(m, [{"id": "a", "text": "a bb ccc"}],
                                 {"axis": "positions", "layer": 1, "positions": "all"},
                                 project=d)
        assert out["item_kind"] == "activations/coordinate" and out["projected"]
        assert out["items"][0]["direction"]["method"] == "test"
        assert out["items"][0]["direction"]["space"]["layer"] == 0
        # layer 1 scales by 2; token 3 sits at position 2 ("bb" -> 1 + 2 % 7 = 3)
        assert [r["coord"] for r in out["items"]] == [0.0, 0.0, 2.0, 0.0]
        assert [r["step"] for r in out["items"]] == [0, 1, 2, 3]
        assert all("vector" not in r for r in out["items"])

    def test_the_cap_counts_what_is_emitted_not_what_is_read(self):
        # 014's shape: many records × many steps. As vectors this is over
        # the cap; with `project` it is coordinates, and with `reduce` it
        # is one vector per record — both must be allowed through.
        from mechbench_compute import interp
        records = [{"id": f"s{i}", "text": "a bb ccc dddd"} for i in range(60)]
        big = {"axis": "positions", "layer": 0, "positions": "all", "max_steps": 4}
        monkey = interp.MAX_VECTOR_FLOATS
        try:
            # vectors 60×4×8 = 1920 over; reduce 60×8 = 480 under; project 0.
            interp.MAX_VECTOR_FLOATS = 1000
            with pytest.raises(ValueError, match="exceeds the"):
                trajectory.capture(StubModel(), records, big)
            d = {"kind": "direction", "vector": onehot(3, 1.0), "layer": 0,
                 "point": "post"}
            out = trajectory.capture(StubModel(), records, big, project=d)
            assert out["projected"] and len(out["items"]) == 240
            out2 = trajectory.capture(StubModel(), records,
                                      {**big, "pool": {"reduce": "mean", "over": "all"}})
            assert len(out2["items"]) == 60  # one pooled vector each
        finally:
            interp.MAX_VECTOR_FLOATS = monkey

    def test_project_dimension_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="dims"):
            trajectory.capture(StubModel(), [{"id": "a", "text": "x"}],
                               {"axis": "positions", "layer": 0},
                               project={"kind": "direction", "vector": [1.0, 0.0],
                                        "layer": 0, "point": "post"})


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
        assert out["item_kind"] == "activations/coordinate" and out["projected"]
        assert [r["coord"] for r in out["items"]] == [2.0, 0.0]
        assert [r["step"] for r in out["items"]] == [0, 1]
        assert "vector" not in out["items"][0]
        assert out["items"][0]["space"]["layer"] == 0

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
        assert [r["cosine"] for r in out["items"]] == [1.0, 1.0, 0.0]
        assert out["items"][0]["norm_ratio"] == 2.0
        assert out["items"][2]["angle_deg"] == 90.0
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
        rows = {(r["group"], r["step"]): r for r in out["items"]}
        assert rows[("lh", 0)]["vector"] == onehot(1, 2.0)
        assert rows[("lh", 1)]["vector"] == onehot(1, 4.0)
        assert rows[("lh", 0)]["n"] == 2 and rows[("lh", 0)]["spread"] == 1.0

    def test_window_as_vectors_feeds_from_vectors(self):
        from mechbench_compute import directions as dirs
        out = trajectory.aggregate({"trajectory": self._labelled()},
                                   {"by": "label", "as": "vectors",
                                    "steps": {"range": [0, 2]}})
        assert out["item_kind"] == "activations/vector" and out["layers"] == [0]
        by = {r["coords"]["label"]: r for r in out["items"]}
        assert by["lh"]["vector"] == onehot(1, 3.0)  # mean of 1,3,3,5
        assert by["lh"]["n_pooled"] == 4
        assert by["lh"]["space"]["layer"] == 0
        # the direction algebra reads it unchanged
        d = dirs.from_vectors(out, layer=0, positive="lh", negative="other")
        assert d["derivation"]["method"] == "diff_of_means"
        v = np.asarray(d["vector"])
        assert v[1] > 0 and v[2] < 0

    def test_window_over_projected_coords_by_id(self):
        t = _traj([{"id": "s1", "step": 0, "layer": 0, "position": 0, "coord": 1.0},
                   {"id": "s1", "step": 1, "layer": 0, "position": 1, "coord": 3.0},
                   {"id": "s2", "step": 0, "layer": 0, "position": 0, "coord": 5.0}])
        t["kind"] = "trajectory_projection"
        out = trajectory.aggregate({"trajectory": t}, {"by": "id", "as": "window"})
        by = {r["group"]: r for r in out["items"]}
        assert by["s1"]["mean"] == 2.0 and by["s1"]["n"] == 2
        assert by["s2"]["mean"] == 5.0

    def test_vectors_mode_refuses_a_projection(self):
        t = _traj([{"id": "a", "step": 0, "layer": 0, "position": 0, "coord": 1.0}])
        with pytest.raises(ValueError, match="vector rows"):
            trajectory.aggregate({"trajectory": t}, {"as": "vectors"})


class TestWiring:
    def test_pure_blocks_are_registered(self):
        for ref in ("trajectory/project",
                    "trajectory/compare",
                    "trajectory/aggregate"):
            assert ref in blocks.PURE_BLOCKS

    def test_params_are_guarded(self):
        with pytest.raises(ValueError, match="does not accept"):
            check_params("trajectory/capture", {"nope": 1})
        check_params("trajectory/capture",
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
        assert out["item_kind"] == "activations/vector" and "layers" not in out
        # Every item carries its own space; the port is the batch coordinate.
        assert [r["coords"]["batch"] for r in out["items"]] == ["adapted", "base"]  # port order
        assert all(r["space"]["layer"] == 12 and "label" not in r for r in out["items"])
        from mechbench_compute import directions as dirs
        d = dirs.from_vectors(out, layer=12, axis="batch", positive="base", negative="adapted")
        v = np.asarray(d["vector"])
        assert v[1] > 0 and v[2] < 0

    def test_union_of_plain_records_is_unchanged(self):
        out = blocks.union({"a": [{"id": "1"}], "b": [{"id": "2"}]}, {})
        assert out["item_kind"] == "records/record"
        assert [r["coords"]["batch"] for r in out["items"]] == ["a", "b"]

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


class TestPoolClause:
    def test_a_window_reads_the_opening(self):
        from mechbench_compute import positions as POS
        spec = POS.pool_spec({"pool": {"reduce": "mean", "over": {"range": [1, 3]}}})
        mat = np.array([[0.0], [1.0], [3.0], [10.0]])
        v, n = POS.pooled(mat, POS.resolve(spec["over"], 4), spec["reduce"])
        assert n == 2 and v[0] == 2.0  # positions 1 and 2

    def test_the_retired_string_form_is_refused_with_the_new_one(self):
        from mechbench_compute import positions as POS
        with pytest.raises(ValueError, match="range"):
            POS.pool_spec({"pool": "first_k"})

    def test_every_selector_resolves(self):
        from mechbench_compute import positions as POS
        toks = ["a", "bb", "ccc", "dd", "a"]
        n = len(toks)
        assert POS.resolve("last", n) == [4] and POS.resolve("final", n) == [4]
        assert POS.resolve("all", n) == [0, 1, 2, 3, 4]
        assert POS.resolve([1, -1], n) == [1, 4]
        assert POS.resolve({"tokens": ["a"]}, n, tokens=toks) == [0, 4]
        assert POS.resolve({"range": [1, 3]}, n) == [1, 2]
        assert POS.resolve({"range": [-2, None]}, n) == [3, 4]
        assert POS.resolve({"after": 3}, n) == [3, 4]
        assert POS.resolve("subject", n, tokens=toks, record={"subject": "bb ccc"}) == [2]
        assert POS.resolve("generated", n, gen_start=3) == [3, 4]
        assert POS.resolve("generated", n, prompt_len=2) == [2, 3, 4]
        with pytest.raises(ValueError, match="generated"):
            POS.resolve("generated", n)
        with pytest.raises(ValueError, match="one is needed"):
            POS.one("all", n)
        assert POS.one({"range": [-1, None]}, n) == 4
