from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mechbench_compute.conformance import check_core, read_inputs_from
from mechbench_compute.conformance.finding import read_subject
from mechbench_compute.conformance.run_examples import check_outputs, dump_result, run_once
from mechbench_compute.ops.geometry.align import OP, align_sets, rank_values, score_cka, score_rsa
from mechbench_compute.registry import REGISTRY

INPUTS = Path(__file__).resolve().parent / "fixtures" / "geometry_align"
IDS = ["r0", "r1", "r2", "r3", "r4"]


def make_similarity(matrices: list[np.ndarray], metric_kind: str = "similarity") -> dict:
    items = [{"group": f"layer={i}", "layer": i, "ids": IDS, "labels": [None] * len(IDS),
              "matrix": m.tolist()} for i, m in enumerate(matrices)]
    return {"kind": "collection", "item_kind": "geometry/similarity", "key": ["group"], "items": items,
            "metric": "cosine", "metric_kind": metric_kind, "symmetric": True}


def make_cosines(seed: int, layers: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(layers):
        x = rng.normal(size=(len(IDS), 8))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        out.append(x @ x.T)
    return out


def read_scores(rows: list[dict]) -> dict[tuple[int, int], float]:
    return {(r["a_layer"], r["b_layer"]): r["score"] for r in rows}


@pytest.mark.parametrize("method", ["cka", "rsa"])
@pytest.mark.parametrize("center", [True, False])
def test_identical_sets_score_one_on_the_diagonal(method, center):
    s = make_similarity(make_cosines(0, 3))
    scores = read_scores(align_sets(s, s, method, center))
    assert set(scores) == {(i, j) for i in range(3) for j in range(3)}
    for i in range(3):
        assert scores[(i, i)] == 1.0
    for (i, j), v in scores.items():
        if i != j:
            assert v < 1.0


def test_cka_is_blind_to_scale_and_rsa_to_any_monotone_map():
    k = make_cosines(1, 1)[0]
    assert score_cka(k, 3.0 * k, center=True) == pytest.approx(1.0)
    assert score_rsa(k, np.exp(k)) == pytest.approx(1.0)
    assert score_rsa(k, -k) == pytest.approx(-1.0)


def test_cka_on_a_hand_computed_pair():
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    l = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert score_cka(k, l, center=False) == pytest.approx(2.0 / np.sqrt(2.0 * 4.0))
    assert score_cka(k, l, center=True) == 0.0


def test_rank_values_averages_ties():
    assert rank_values(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_a_distance_set_reads_as_its_kernel():
    d = 1.0 - make_cosines(2, 1)[0]
    np.fill_diagonal(d, 0.0)
    sims = make_similarity([1.0 - d])
    dists = make_similarity([d], metric_kind="distance")
    assert read_scores(align_sets(sims, dists, "rsa", True))[(0, 0)] == 1.0
    assert read_scores(align_sets(dists, dists, "cka", True))[(0, 0)] == 1.0


def test_different_records_are_refused():
    a = make_similarity(make_cosines(0, 1))
    b = make_similarity(make_cosines(0, 1))
    b["items"][0]["ids"] = list(reversed(IDS))
    with pytest.raises(ValueError, match="different"):
        align_sets(a, b, "cka", True)


def test_a_per_head_group_is_refused():
    a = make_similarity(make_cosines(0, 1))
    a["items"][0]["head"] = 1
    with pytest.raises(ValueError, match="more than one matrix"):
        align_sets(a, a, "cka", True)


def test_the_core_declarations_pass_and_the_example_runs_twice_identically():
    assert [f for f in check_core() if read_subject(f.at) in ("geometry/align", "geometry/alignment")] == []
    inputs = read_inputs_from(INPUTS)(OP.name, OP.example_inputs)
    resolved = REGISTRY.resolve(OP.name)
    first = run_once(resolved, inputs, OP.example, None)
    second = run_once(resolved, inputs, OP.example, None)
    kind, problems = check_outputs(first, OP, OP.name)
    assert problems == [] and kind == "geometry/alignment"
    assert dump_result(first) == dump_result(second)
    assert read_scores(first["items"])[(0, 0)] == pytest.approx(0.961409)
