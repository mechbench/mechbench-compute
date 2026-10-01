from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import directions as d
from mechbench_compute import shapes as S
from mechbench_compute.ops.direction.average import average
from mechbench_compute.ops.direction.classify import fit_probe
from mechbench_compute.ops.direction.decompose import fit_component
from mechbench_compute.ops.direction.fit import fit_mean_difference
from mechbench_compute.ops.direction.orthogonalize import orthogonalize
from mechbench_compute.ops.direction.project import project_rows
from mechbench_compute.ops.direction.regress import fit_regression


def _space(layer, d_=3, point="resid_post"):
    return S.space(model="fake/m@r", layer=layer, point=point, d=d_)


def _vectors(layer=3, point="post", n=4, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    rows = []
    for i in range(n):
        rows.append({"id": f"p{i}", "label": "pos", "layer": layer,
                     "vector": [float(x) for x in base + rng.normal(scale=0.1, size=dim) + 2.0]})
        rows.append({"id": f"n{i}", "label": "neg", "layer": layer,
                     "vector": [float(x) for x in base + rng.normal(scale=0.1, size=dim) - 2.0]})
    return {"kind": "residual_vectors", "point": point, "model": "fake/m@r", "rows": rows}


def _cos(a, b):
    from mechbench_compute import metrics as M

    return float(M.matrix("activations/vector", [a, b], "cosine")[0][0, 1])


class TestMake:
    def test_unit_and_provenance(self):
        x = d.make([3.0, 4.0], _space(2, 2), method="test", sources=["a"])
        assert x["kind"] == "direction/vector" and x["space"]["d"] == 2
        assert x["space"] == {"model": "fake/m@r", "layer": 2, "point": "resid_post", "head": None, "d": 2}
        assert abs(np.linalg.norm(x["vector"]) - 1.0) < 1e-6 and x["norm"] == 5.0
        assert x["derivation"]["method"] == "test" and x["derivation"]["sources"] == ["a"]

    def test_zero_refused(self):
        with pytest.raises(ValueError):
            d.make([0.0, 0.0], _space(0, 2), method="t")


def _measured(layer=3, n=120, dim=8, seed=7, noise=0.05):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=dim)
    axis /= np.linalg.norm(axis)
    base = rng.normal(size=dim)
    rows = []
    for i in range(n):
        value = float(rng.uniform(0.0, 10.0))
        v = base + value * axis + rng.normal(scale=noise, size=dim)
        rows.append({"id": f"t{i}", "layer": layer, "coords": {"surprisal": value},
                     "vector": [float(x) for x in v]})
    return {"kind": "residual_vectors", "point": "post", "model": "fake/m@r",
            "rows": rows}, axis


class TestRegression:
    def test_recovers_the_axis_a_number_is_written_along(self):
        v, axis = _measured()
        x = fit_regression(v, layer=3, target="surprisal")
        assert x["kind"] == "direction/vector"
        assert x["derivation"]["method"] == "ridge"
        assert _cos(x, d.make(axis, _space(3, 8), method="planted")) > 0.98
        assert x["derivation"]["r2_test"] > 0.9
        assert x["derivation"]["n_train"] + x["derivation"]["n_test"] == 120

    def test_noise_scores_near_zero_but_still_fits(self):
        rng = np.random.default_rng(3)
        rows = [{"id": f"r{i}", "layer": 3, "coords": {"surprisal": float(rng.normal())},
                 "vector": [float(x) for x in rng.normal(size=8)]} for i in range(60)]
        v = {"kind": "residual_vectors", "point": "post", "model": "fake/m@r", "rows": rows}
        x = fit_regression(v, layer=3, target="surprisal")
        assert x["derivation"]["r2_test"] < 0.5

    def test_repeats_exactly_and_refuses_too_few(self):
        v, _ = _measured()
        a = fit_regression(v, layer=3, target="surprisal", seed=11)
        b = fit_regression(v, layer=3, target="surprisal", seed=11)
        assert a["vector"] == b["vector"] and a["derivation"] == b["derivation"]
        with pytest.raises(ValueError, match="at least 8"):
            fit_regression(v, layer=3, target="not_a_coordinate")


class TestProducers:
    def test_diff_of_means_points_from_neg_to_pos(self):
        v = _vectors()
        x = fit_mean_difference(v, layer=3, positive="pos", negative="neg")
        assert x["space"] == {"model": "fake/m@r", "layer": 3, "point": "resid_post", "head": None, "d": 8}
        assert x["derivation"]["axis"] == "label"
        assert (x["derivation"]["positive"], x["derivation"]["negative"]) == ("pos", "neg")
        assert x["derivation"]["n_positive"] == 4
        assert all(c > 0 for c in x["vector"])

    def test_pca_first_component_is_the_label_axis(self):
        v = _vectors()
        x = fit_component(v, layer=3)
        m = fit_mean_difference(v, layer=3, positive="pos", negative="neg")
        assert abs(_cos(x, m)) > 0.95
        assert 0.5 < x["derivation"]["explained"] <= 1.0

    def test_missing_layer_refused(self):
        with pytest.raises(ValueError):
            fit_mean_difference(_vectors(), layer=9, positive="pos", negative="neg")

    def test_an_axis_fit_across_two_models_lives_in_neither(self):
        from mechbench_compute import ops
        from mechbench_compute.lexicon import kinds as K

        def capture(model, seed):
            sp = S.space(model=model, layer=12, point="resid_post", d=4)
            rng = np.random.default_rng(seed)
            return K.collection("activations/vector",
                                [S.vector(list(rng.normal(size=4)), sp, id="p")], model=model)

        base = capture("fake/base", 0)
        axes = {}
        for i, name in enumerate(["die", "letters"]):
            pair = ops.run_standalone(
                "records/union", {"base": base, "adapted": capture(f"fake/{name}", i + 1)}, {})
            axes[name] = fit_mean_difference(pair, layer=12, axis="batch", positive="base", negative="adapted")
            assert axes[name]["space"]["model"] is None
            assert axes[name]["derivation"]["models"] == [f"fake/{name}", "fake/base"]
        union = ops.run_standalone("records/union", axes, {})
        sim = ops.run_standalone("geometry/compare", {"items": union}, {"axis": "batch"})
        assert sim["items"][0]["ids"] == ["die", "letters"]
        assert -1.0 <= sim["items"][0]["matrix"][0][1] <= 1.0


class TestArithmetic:
    def test_orthogonalize_removes_the_component(self):
        a = d.make([1.0, 0.0, 0.0], _space(1), method="t")
        b = d.make([1.0, 1.0, 0.0], _space(1), method="t")
        o = orthogonalize(b, [a])
        assert abs(_cos(o, a)) < 1e-6
        assert o["derivation"]["method"] == "orthogonalize"

    def test_add_and_average(self):
        a = d.make([1.0, 0.0], _space(1, 2), method="t")
        b = d.make([0.0, 1.0], _space(1, 2), method="t")
        s = d.add([a, b], [1.0, 1.0])
        assert np.allclose(s["vector"], [np.sqrt(0.5)] * 2, atol=1e-6)
        assert average([a, b])["derivation"]["method"] == "average"

    def test_spaces_must_match(self):
        a = d.make([1.0, 0.0], _space(1, 2), method="t")
        b = d.make([1.0, 0.0], _space(2, 2), method="t")
        with pytest.raises(ValueError, match="different spaces"):
            d.add([a, b])
        c = d.make([1.0, 0.0], S.space(model="other/m", layer=1, point="resid_post", d=2), method="t")
        with pytest.raises(ValueError, match="model"):
            d.add([a, c])

    def test_project_rows(self):
        v = _vectors()
        m = fit_mean_difference(v, layer=3, positive="pos", negative="neg")
        pr = project_rows(v, m)
        assert pr["item_kind"] == "activations/coordinate"
        pos = [r["coord"] for r in pr["items"] if r["coords"]["label"] == "pos"]
        neg = [r["coord"] for r in pr["items"] if r["coords"]["label"] == "neg"]
        assert min(pos) > max(neg)
        assert pr["items"][0]["direction"]["method"] == "diff_of_means"
        assert pr["items"][0]["space"]["layer"] == 3


def _wide(layer=3, n=30, dim=6, seed=11):
    from mechbench_compute.lexicon import kinds as K

    rng = np.random.default_rng(seed)
    scales = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])[:dim]
    sp = S.space(model="fake/m@r", layer=layer, point="resid_post", d=dim)
    return K.collection("activations/vector",
                        [S.vector(list(rng.normal(size=dim) * scales), sp, id=f"v{i}")
                         for i in range(n)], model="fake/m@r")


class TestDecompose:
    def test_one_component_is_unchanged(self):
        from mechbench_compute.ops.direction import decompose

        x = decompose.run(None, {"vectors": _vectors()}, {"layer": 3})
        assert list(x) == ["kind", "coords", "space", "vector", "norm", "unit", "derivation"]
        assert {k: v for k, v in x.items() if k != "vector"} == {
            "kind": "direction/vector", "coords": {},
            "space": {"model": "fake/m@r", "layer": 3, "point": "resid_post", "head": None, "d": 8},
            "norm": 1.0, "unit": True,
            "derivation": {"method": "pca", "sources": [], "model": "fake/m@r",
                           "component": 0, "explained": 0.9986, "n_items": 8}}
        assert [round(c, 4) for c in x["vector"]] == [
            0.3501, 0.347, 0.3527, 0.3537, 0.3588, 0.3627, 0.3504, 0.3528]
        y = decompose.run(None, {"vectors": _vectors()}, {"layer": 3, "component": 1})
        assert y["derivation"]["component"] == 1 and y["derivation"]["explained"] == 0.0007

    def test_k_makes_a_collection_ordered_by_variance(self):
        from mechbench_compute.lexicon import kinds as K
        from mechbench_compute.ops.direction import decompose

        out = decompose.run(None, {"vectors": _wide()}, {"layer": 3, "k": 4})
        assert out["kind"] == K.COLLECTION and out["item_kind"] == "direction/vector"
        items = K.items_of(out)
        assert [it["id"] for it in items] == ["pc0", "pc1", "pc2", "pc3"]
        assert [it["coords"]["component"] for it in items] == [0, 1, 2, 3]
        assert [it["derivation"]["component"] for it in items] == [0, 1, 2, 3]
        shares = [it["derivation"]["explained"] for it in items]
        assert shares == sorted(shares, reverse=True)
        assert out["explained"] == round(sum(shares), 4)
        assert 0.0 < out["explained"] <= 1.0
        assert (out["components"], out["layer"], out["point"], out["n_items"]) == (4, 3, "resid_post", 30)
        assert out["order_by"] == ["coords.component"]
        for i, it in enumerate(items):
            one = decompose.run(None, {"vectors": _wide()}, {"layer": 3, "component": i})
            assert it["vector"] == one["vector"]
            assert it["derivation"] == one["derivation"]

    def test_k_and_component_together_are_refused(self):
        from mechbench_compute.ops.direction import decompose

        with pytest.raises(ValueError, match="not both"):
            decompose.run(None, {"vectors": _wide()}, {"layer": 3, "k": 2, "component": 0})

    def test_k_out_of_range_is_refused(self):
        from mechbench_compute.ops.direction import decompose

        with pytest.raises(ValueError, match="k is between 1 and 6"):
            decompose.run(None, {"vectors": _wide()}, {"layer": 3, "k": 7})

    def test_a_map_reads_each_component_as_one_direction(self):
        from mechbench_compute.lexicon import kinds as K
        from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

        vectors = _wide()
        body = {"nodes": [{"id": "proj", "block": "direction/project",
                           "inputs": {"vectors": vectors}}],
                "edges": [{"from": {"input": "record"}, "to": {"node": "proj", "port": "direction"}}]}
        graph = {"dataflow": 2, "nodes": [
            {"id": "pcs", "block": "direction/decompose", "params": {"layer": 3, "k": 3},
             "inputs": {"vectors": vectors}},
            {"id": "each", "block": "records/map", "params": {"body": body}},
        ], "edges": [{"from": {"node": "pcs"}, "to": {"node": "each", "port": "records"}}]}
        out = ProtocolExecutor().run(ProtocolSpec(
            kind="pipeline", prompt="", model_id=None,
            extra={"graph": graph})).payload["outputs"]["each"]
        mapped = sorted({it["coords"]["mapped"] for it in K.items_of(out)})
        assert mapped == ["pc0", "pc1", "pc2"]
        assert len(K.items_of(out)) == 3 * 30

    def test_examples_reads_a_component_handed_over_as_a_collection_of_one(self):
        from mechbench_compute.lexicon import kinds as K
        from mechbench_compute.ops.activations.examples import find_top_examples
        from mechbench_compute.ops.direction.decompose import fit_components
        from tests.test_interp_blocks import D_MODEL, StubModel

        rng = np.random.default_rng(3)
        sp = S.space(model="stub", layer=1, point="resid_post", d=D_MODEL)
        vectors = K.collection("activations/vector",
                               [S.vector(list(rng.normal(size=D_MODEL)), sp, id=f"v{i}")
                                for i in range(10)], model="stub")
        pcs = fit_components(vectors, layer=1, k=2)
        records = [{"id": "r0", "user": "aa bbb aa"}]
        for item in K.items_of(pcs):
            one = K.collection("direction/vector", [item])
            out = find_top_examples(StubModel(), records, {"k": 2}, direction=one)
            assert (out["layer"], out["point"]) == (1, "resid_post")
            assert out["items"] == find_top_examples(StubModel(), records, {"k": 2},
                                                     direction=item)["items"]


class TestBlocks:
    def test_registered_and_callable(self):
        from mechbench_compute import ops

        v = _vectors()
        x = ops.run_standalone(
            "direction/fit", {"vectors": v}, {"layer": 3, "positive": "pos", "negative": "neg"})
        assert x["kind"] == "direction/vector"
        pair = ops.run_standalone("records/union", {"a": x, "b": dict(x)}, {})
        sim = ops.run_standalone("geometry/compare", {"items": pair}, {})
        assert abs(sim["items"][0]["matrix"][0][1] - 1.0) < 1e-6
        avg = ops.run_standalone("direction/average", {"d1": x, "d2": x}, {})
        assert avg["derivation"]["method"] == "average"

    def test_levels_declared(self):
        from mechbench_compute import resume as rm

        assert rm.resume_level("direction/add") == "reproducible"
        assert rm.resume_level("intervene/apply") == "reproducible"
        assert rm.item_resumable("intervene/apply")


class TestClassify:
    D = 8

    def _corpus(self, planted_at=6, flat_at=2, n=40, seed=0):
        rng = np.random.default_rng(seed)
        axis = rng.normal(size=self.D)
        axis /= np.linalg.norm(axis)
        items = []
        for layer, sep in ((flat_at, 0.0), (planted_at, 3.0)):
            for i in range(n):
                cls = "dusk" if i % 2 else "dawn"
                v = rng.normal(size=self.D) + (sep if cls == "dawn" else -sep) * axis
                items.append({**S.vector(v.astype(np.float32),
                                        S.space(model="m", layer=layer, point="resid_post", d=self.D),
                                        id=f"p{i}", coords={"sense": cls}),
                              "kind": "activations/vector"})
        return {"kind": "collection", "item_kind": "activations/vector", "items": items}, axis

    def test_the_curve_says_where_the_label_becomes_decodable(self):
        corpus, axis = self._corpus()
        out = fit_probe(corpus, axis="sense", seed=1)
        assert out["item_kind"] == "direction/vector" and out["axis"] == "sense"
        by_layer = {it["coords"]["layer"]: it for it in out["items"]}
        assert sorted(by_layer) == [2, 6]
        flat, planted = by_layer[2], by_layer[6]
        assert planted["accuracy_test"] == 1.0 and planted["over_baseline"] > 0.4
        assert flat["accuracy_test"] <= flat["baseline"] + 0.2
        cos = abs(float(np.asarray(planted["vector"]) @ axis))
        assert cos > 0.9, cos
        assert planted["auc"] == 1.0 and planted["derivation"]["method"] == "logistic"

    def test_one_direction_for_two_labels_pointing_at_the_positive(self):
        corpus, _ = self._corpus()
        [it] = [i for i in fit_probe(corpus, axis="sense")["items"]
                if i["coords"]["layer"] == 6]
        made = it["derivation"]
        assert {made["positive"], made["negative"]} == {"dawn", "dusk"} and made["axis"] == "sense"
        assert it["coords"]["label"] == made["positive"]

    def test_three_labels_give_one_probe_each_against_the_rest(self):
        rng = np.random.default_rng(4)
        items = []
        for i in range(60):
            cls = ["a", "b", "c"][i % 3]
            centre = np.zeros(self.D)
            centre[i % 3] = 4.0
            v = rng.normal(size=self.D) * 0.4 + centre
            items.append({**S.vector(v.astype(np.float32),
                                     S.space(model="m", layer=3, point="resid_post", d=self.D),
                                     id=f"p{i}", coords={"topic": cls}), "kind": "activations/vector"})
        out = fit_probe({"kind": "collection", "item_kind": "activations/vector",
                                        "items": items}, axis="topic", seed=2)
        assert [it["coords"]["label"] for it in out["items"]] == ["a", "b", "c"]
        assert all(it["derivation"]["negative"] == "rest" for it in out["items"])
        assert all(it["accuracy_test"] > 0.8 for it in out["items"])

    def test_it_refuses_what_it_cannot_answer(self):
        corpus, _ = self._corpus(n=40)
        with pytest.raises(ValueError, match="no items carry"):
            fit_probe(corpus, axis="nothing")
        one_label = {"kind": "collection", "item_kind": "activations/vector",
                     "items": [{**S.vector(np.ones(self.D, np.float32),
                                           S.space(model="m", layer=1, point="resid_post", d=self.D),
                                           id=f"p{i}", coords={"sense": "same"}),
                                "kind": "activations/vector"} for i in range(10)]}
        with pytest.raises(ValueError, match="needs two labels"):
            fit_probe(one_label, axis="sense")
        few = {"kind": "collection", "item_kind": "activations/vector",
               "items": corpus["items"][:4]}
        with pytest.raises(ValueError, match="at least 8 labelled"):
            fit_probe(few, axis="sense")

    def test_a_selected_probe_is_a_direction_an_intervention_takes(self):
        corpus, _ = self._corpus()
        out = fit_probe(corpus, axis="sense")
        one = {"kind": "collection", "item_kind": "direction/vector",
               "items": [it for it in out["items"] if it["coords"]["layer"] == 6]}
        assert d.coerce_array(one).shape == (self.D,)
        with pytest.raises(ValueError, match="collection of 2"):
            d.coerce_array(out)
