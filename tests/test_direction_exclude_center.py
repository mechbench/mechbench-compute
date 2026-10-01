from __future__ import annotations

import numpy as np
import pytest

from mechbench_compute import ops
from mechbench_compute import shapes as S
from mechbench_compute.expr.engine import load_engine
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.activations.capture import capture_residual_vectors
from mechbench_compute.ops.direction.average import average
from mechbench_compute.ops.direction.fit import fit_mean_difference
from tests.tiny_models import build_tiny_model

LAYER = 1
FIT = {"layer": LAYER, "axis": "side", "positive": "die", "negative": "colour"}


@pytest.fixture(scope="module")
def captured():
    model = build_tiny_model("gemma3")
    prompts = [("roll a die", "die"), ("roll two dice", "die"), ("throw a d20", "die"),
               ("name a colour", "colour"), ("pick a colour", "colour"), ("say a hue", "colour")]
    records = [{"id": f"r{i}", "user": u, "coords": {"side": s}} for i, (u, s) in enumerate(prompts)]
    return capture_residual_vectors(model, records, {"layers": [LAYER], "position": "last"})


def raw_difference(vectors):
    rows = vectors["items"]
    pos = np.array([r["vector"] for r in rows if r["coords"]["side"] == "die"], dtype=np.float32)
    neg = np.array([r["vector"] for r in rows if r["coords"]["side"] == "colour"], dtype=np.float32)
    return pos.mean(0) - neg.mean(0)


def planted(offsets, seed=3, n=40, dim=16):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=dim)
    axis -= axis.mean()
    axis /= np.linalg.norm(axis)
    sp = S.space(model="fake/m@r", layer=LAYER, point="resid_post", d=dim)
    items = []
    for side, shift, offset in (("die", 1.0, offsets[0]), ("colour", 0.0, offsets[1])):
        for i in range(n):
            v = shift * axis + offset + rng.normal(scale=0.01, size=dim) + rng.normal(scale=2.0)
            items.append(S.vector(list(v), sp, id=f"{side}{i}", coords={"side": side}))
    return K.collection("activations/vector", items, model="fake/m@r"), axis


class TestExclude:
    def test_exclude_zeroes_the_named_dimensions_and_the_norm_drops(self, captured):
        raw = raw_difference(captured)
        big = int(np.argmax(np.abs(raw)))
        whole = fit_mean_difference(captured, **FIT)
        left = fit_mean_difference(captured, **FIT, exclude=[big])
        assert left["vector"][big] == 0.0 and whole["vector"][big] != 0.0
        by_hand = raw.astype(np.float64).copy()
        by_hand[big] = 0.0
        assert left["norm"] == pytest.approx(float(np.linalg.norm(by_hand)), rel=1e-5)
        assert left["norm"] < whole["norm"]
        share = raw[big] ** 2 / float(np.sum(raw.astype(np.float64) ** 2))
        assert left["derivation"]["exclude"] == [big]
        assert left["derivation"]["excluded_share"] == pytest.approx(share, abs=1e-6)
        assert left["norm"] ** 2 == pytest.approx(whole["norm"] ** 2 * (1 - share), rel=1e-4)
        assert np.dot(left["vector"], by_hand / np.linalg.norm(by_hand)) == pytest.approx(1.0, abs=1e-4)

    def test_a_cosine_downstream_agrees_with_dropping_by_hand(self, captured):
        raw = raw_difference(captured)
        big = int(np.argmax(np.abs(raw)))
        a = fit_mean_difference(captured, **FIT, exclude=[big])
        b = fit_mean_difference(captured, **FIT, exclude=[big, 0])
        pair = ops.run_standalone("records/union", {"a": a, "b": b}, {})
        sim = ops.run_standalone("geometry/compare", {"items": pair}, {})
        va, vb = np.array(a["vector"]), np.array(b["vector"])
        assert sim["items"][0]["matrix"][0][1] == pytest.approx(float(va @ vb), abs=1e-5)

    def test_out_of_range_dimensions_are_refused(self, captured):
        with pytest.raises(ValueError, match="outside"):
            fit_mean_difference(captured, **FIT, exclude=[32])

    def test_average_leaves_the_dimensions_out_of_every_input(self, captured):
        raw = raw_difference(captured)
        big = int(np.argmax(np.abs(raw)))
        a = fit_mean_difference(captured, **FIT)
        out = average([a, dict(a)], exclude=[big])
        assert out["vector"][big] == 0.0 and out["derivation"]["exclude"] == [big]
        dropped = fit_mean_difference(captured, **FIT, exclude=[big])
        assert np.dot(out["vector"], dropped["vector"]) == pytest.approx(1.0, abs=1e-5)


class TestCenter:
    def test_center_removes_the_shared_offset(self):
        vectors, axis = planted(offsets=(3.0, -1.0))
        ones = np.ones(axis.size) / np.sqrt(axis.size)
        raw = fit_mean_difference(vectors, **FIT)
        centred = fit_mean_difference(vectors, **FIT, center=True)
        assert abs(float(np.dot(raw["vector"], ones))) > 0.9
        assert abs(float(np.dot(centred["vector"], ones))) < 1e-4
        assert float(np.dot(centred["vector"], axis)) > 0.99
        assert centred["derivation"]["center"] is True
        assert "center" not in raw["derivation"]

    def test_center_works_over_the_dimensions_left_in(self):
        vectors, _ = planted(offsets=(3.0, -1.0))
        out = fit_mean_difference(vectors, **FIT, center=True, exclude=[2])
        v = np.array(out["vector"])
        assert v[2] == 0.0 and abs(float(np.delete(v, 2).sum())) < 1e-4


class TestDefaults:
    def test_the_defaults_are_bit_identical_to_leaving_the_params_out(self, captured):
        bare = ops.run_standalone("direction/fit", {"vectors": captured}, dict(FIT))
        for extra in ({"center": False, "exclude": None}, {"center": False, "exclude": []}):
            given = ops.run_standalone("direction/fit", {"vectors": captured}, {**FIT, **extra})
            assert given == bare
        assert not {"center", "exclude", "excluded_share"} & set(bare["derivation"])
        a = fit_mean_difference(captured, **FIT)
        avg = ops.run_standalone("direction/average", {"d1": a, "d2": bare}, {})
        assert ops.run_standalone("direction/average", {"d1": a, "d2": bare}, {"exclude": None}) == avg


class TestSpeak:
    def test_the_vector_says_what_was_done(self, captured):
        speak = K.BY_KIND["direction/vector"].speak
        plain = fit_mean_difference(captured, **FIT)
        shaped = fit_mean_difference(captured, **FIT, center=True, exclude=[3])
        said = load_engine().render(speak, [plain, shaped], {}, {}).values
        assert said[0].startswith("diff_of_means direction at resid_post layer 1, from colour to die; norm ")
        assert "centred" not in said[0] and "left out" not in said[0]
        assert ", on centred vectors, dimensions [3] left out (they carried " in said[1]
