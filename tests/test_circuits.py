from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import lexicon, lora
from mechbench_compute import resume as rm
from mechbench_compute.blocks.estimate_bootstrap_ratio import estimate_bootstrap_ratio
from mechbench_compute.distill import render
from mechbench_compute.expr.engine import load_engine
from mechbench_compute.interp.read_last_logp import read_last_logp
from mechbench_compute.interp.read_metric import read_metric
from mechbench_compute.interp.resolve_target import resolve_target
from mechbench_compute.intervene.compile import compile as compile_intervention
from mechbench_compute.intervene.spec_intervention import SpecIntervention
from mechbench_compute.interventions import Ablate
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops import Context
from mechbench_compute.ops.geometry.compare import compare_geometry
from mechbench_compute.ops.intervene import ablate_circuit as ablate_circuit_op
from mechbench_compute.ops.intervene import prune as prune_op
from mechbench_compute.ops.intervene.ablate_circuit import ablate_circuits
from mechbench_compute.ops.intervene.ablate_heads import ablate_heads
from mechbench_compute.ops.intervene.prune import prune_circuits
from mechbench_compute.protocol.model import ModelLoading
from tests.tiny_models import build_tiny_model

RECORDS = [{"id": "a", "user": "the cat sat on a mat"}, {"id": "b", "user": "a dog ran and the cat sat"}]

TRACKED = {"answer": "mat"}

HEADS = {
    "kind": "intervene/heads", "id": "mean", "coords": {}, "axes": ["layer", "head"],
    "measures": {"mean_delta": [[-0.5, 0.1, -0.2], [-0.05, -0.15, 0.3]]},
    "layers": [0, 1], "n_heads": 3, "n_conditions": 2, "n_off_top1": 1,
    "conditions": [{"id": "p", "target": {"id": 4, "text": " Paris"}},
                   {"id": "q", "target": {"id": 5, "text": " Rome"}}],
}

HEAD_UNIVERSE = {"points": ["attn.per_head_out"], "layers": [0, 1, 2, 3], "n_heads": 4, "positions": "all"}


def build_trace(lengths=(3, 3), share=(True, True)):
    pairs = []
    for i, (n, has_share) in enumerate(zip(lengths, share, strict=True)):
        recovery = [[0.1 * (i + 1) * (p + 1) for p in range(n)], [0.0] * (n - 1) + [1.0 + i]]
        measures = {"recovery": recovery}
        if has_share:
            measures["share"] = [[v / 2 for v in row] for row in recovery]
        pairs.append({"id": f"pair{i}", "coords": {}, "axes": ["layer", "position"], "measures": measures,
                      "tokens": ["<start>", "the", "cat", "sat"][-n:], "target": {"id": 8, "text": "cat"},
                      "metric": "logprob"})
    return lexicon.collection("intervene/trace", pairs, point="attn_out", metric="logprob",
                              method="attribution", layers=[2, 5])


def hand_circuit(cid, heads, universe=HEAD_UNIVERSE, ids=("a", "b")):
    comps = [{"address": f"L{layer}.attn.per_head_out.H{head}@all", "point": "attn.per_head_out",
              "layer": layer, "head": head, "position": "all", "effect": -0.1} for layer, head in heads]
    return {"id": cid, "components": comps, "metric": "logprob", "measure": "mean_delta", "ablation": "hand",
            "universe": universe, "source": {"kind": "hand"}, "task": {"ids": list(ids)},
            "derivation": {"method": "hand", "sign": "negative", "kept": len(comps), "total": 16}}


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_model("gemma3")


class TestPruneAHeadsGrid:
    def test_a_threshold_list_gives_one_circuit_per_value(self):
        out = prune_circuits(HEADS, {"name": "c", "threshold": [0.1, 0.3]})
        assert out["item_kind"] == "intervene/circuit" and out["thresholds"] == [0.1, 0.3]
        low, high = out["items"]
        assert low["id"] == "c@0.1" and high["id"] == "c@0.3"
        assert [c["address"] for c in low["components"]] == [
            "L0.attn.per_head_out.H0@all", "L0.attn.per_head_out.H2@all", "L1.attn.per_head_out.H1@all"]
        assert low["derivation"] == {"method": "prune", "threshold": 0.1, "sign": "negative", "kept": 3,
                                     "total": 6, "kept_share": round(0.85 / 0.9, 4)}
        assert [c["address"] for c in high["components"]] == ["L0.attn.per_head_out.H0@all"]
        assert low["components"][0] == {"address": "L0.attn.per_head_out.H0@all", "point": "attn.per_head_out",
                                        "layer": 0, "head": 0, "position": "all", "effect": -0.5}

    def test_the_circuit_carries_its_metric_universe_and_task(self):
        c = prune_circuits(HEADS, {"name": "c", "threshold": 0.1})["items"][0]
        assert c["id"] == "c" and c["metric"] == "logprob" and c["measure"] == "mean_delta"
        assert c["ablation"] == "zero" and "links" not in c
        assert c["universe"] == {"points": ["attn.per_head_out"], "layers": [0, 1], "n_heads": 3,
                                 "positions": "all"}
        assert c["task"] == {"ids": ["p", "q"], "n_off_top1": 1,
                             "targets": [{"id": "p", "target": {"id": 4, "text": " Paris"}},
                                         {"id": "q", "target": {"id": 5, "text": " Rome"}}]}
        assert c["source"] == {"path": None, "kind": "intervene/heads"}

    def test_top_per_layer_and_sign(self):
        top = prune_circuits(HEADS, {"name": "c", "top": 2})["items"][0]
        assert [c["address"] for c in top["components"]] == [
            "L0.attn.per_head_out.H0@all", "L0.attn.per_head_out.H2@all"]
        assert top["derivation"]["top"] == 2 and "threshold" not in top["derivation"]
        capped = prune_circuits(HEADS, {"name": "c", "threshold": 0.1, "per_layer": 1})["items"][0]
        assert [c["address"] for c in capped["components"]] == [
            "L0.attn.per_head_out.H0@all", "L1.attn.per_head_out.H1@all"]
        assert capped["derivation"]["per_layer"] == 1
        up = prune_circuits(HEADS, {"name": "c", "threshold": 0.3, "sign": "positive"})["items"][0]
        assert [c["address"] for c in up["components"]] == ["L1.attn.per_head_out.H2@all"]
        assert up["derivation"]["kept_share"] == 0.75
        either = prune_circuits(HEADS, {"name": "c", "threshold": 0.2, "sign": "magnitude"})["items"][0]
        assert [c["effect"] for c in either["components"]] == [-0.5, 0.3]

    def test_a_threshold_that_keeps_nothing_is_a_circuit_with_no_components(self):
        c = prune_circuits(HEADS, {"name": "c", "threshold": 0.9})["items"][0]
        assert c["components"] == [] and c["derivation"]["kept"] == 0 and c["derivation"]["kept_share"] == 0.0

    def test_threshold_and_top_together_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            prune_circuits(HEADS, {"name": "c", "threshold": 0.1, "top": 2})

    def test_the_source_path_comes_from_the_port(self):
        ctx = Context(input_paths={"grid": "you/lab/heads"})
        out = prune_op.run(ctx, {"grid": HEADS}, {"name": "c"})
        assert out["items"][0]["source"]["path"] == "you/lab/heads"


class TestPruneATrace:
    def test_pairs_combine_at_positions_from_the_end(self):
        c = prune_circuits(build_trace(), {"name": "t", "threshold": 0.15})["items"][0]
        assert c["measure"] == "share" and c["ablation"] == "patch" and c["derivation"]["sign"] == "positive"
        assert c["universe"] == {"points": ["attn_out"], "layers": [2, 5], "positions": [-3, -2, -1]}
        assert c["source"] == {"path": None, "kind": "intervene/trace", "method": "attribution",
                               "point": "attn_out"}
        assert c["task"]["ids"] == ["pair0", "pair1"]
        by_address = {x["address"]: x for x in c["components"]}
        assert set(by_address) == {"L5.attn_out@-1", "L2.attn_out@-1"}
        assert by_address["L5.attn_out@-1"]["effect"] == pytest.approx(0.75)
        assert by_address["L2.attn_out@-1"] == {"address": "L2.attn_out@-1", "point": "attn_out", "layer": 2,
                                                "position": -1, "token": "sat", "effect": pytest.approx(0.225)}

    def test_a_pair_without_a_gap_reads_every_pair_by_recovery(self):
        c = prune_circuits(build_trace(share=(True, False)), {"name": "t"})["items"][0]
        assert c["measure"] == "recovery"

    def test_pairs_of_different_lengths_are_refused_by_id(self):
        with pytest.raises(ValueError, match=r"\['pair1'\] do not align from the end"):
            prune_circuits(build_trace(lengths=(3, 4)), {"name": "t"})

    def test_a_pair_with_an_error_is_refused_by_id(self):
        trace = build_trace()
        trace["items"][1] = {"id": "pair1", "axes": ["layer", "position"], "measures": {},
                             "error": "prompts tokenize to different lengths"}
        with pytest.raises(ValueError, match=r"\['pair1'\] carry no grid"):
            prune_circuits(trace, {"name": "t"})


class TestJaccard:
    def build(self):
        a = {"id": "a", "components": [{"address": "x", "effect": -0.4}, {"address": "y", "effect": 0.2}]}
        b = {"id": "b", "components": [{"address": "y", "effect": -0.1}, {"address": "z", "effect": 0.3}]}
        return lexicon.collection("intervene/circuit", [a, b])

    def test_plain_is_shared_addresses_over_the_union(self):
        out = compare_geometry({"items": self.build()}, {})
        assert out["metric"] == "jaccard" and out["metric_kind"] == "similarity"
        assert out["items"][0]["matrix"] == [[1.0, 1 / 3], [1 / 3, 1.0]]

    def test_weighted_is_min_over_max_of_the_effects(self):
        out = compare_geometry({"items": self.build()}, {"metric": "jaccard", "options": {"weighted": True}})
        assert out["items"][0]["matrix"][0][1] == pytest.approx(0.1 / (0.4 + 0.2 + 0.3))


class TestBootstrapRatio:
    def test_the_interval_covers_the_point_and_repeats_under_its_seed(self):
        rng = np.random.default_rng(1)
        den = rng.normal(2.0, 0.3, size=12)
        num = den * 0.6 + rng.normal(0, 0.1, size=12)
        lo, hi = estimate_bootstrap_ratio(num, den, 0.95, 2000, 7)
        assert lo <= num.mean() / den.mean() <= hi
        assert (lo, hi) == estimate_bootstrap_ratio(num, den, 0.95, 2000, 7)
        assert (lo, hi) != estimate_bootstrap_ratio(num, den, 0.95, 2000, 8)


def read_by_hand(model, record, heads, metric="logprob"):
    ids = render(model, record).array
    logits = model.run(ids, interventions=[Ablate.head(layer, head) for layer, head in heads]).logits
    lp = read_last_logp(logits)
    if metric == "entropy":
        p = np.exp(lp.astype(np.float64))
        return float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return float(lp[12])


ALL_HEADS = [(layer, head) for layer in range(4) for head in range(4)]


class TestAblateCircuit:
    def test_a_one_head_circuit_without_its_head_is_minus_the_grid_cell(self, tiny):
        grid = ablate_heads(tiny, RECORDS, {"tracked": TRACKED})
        circuit = prune_circuits(grid, {"name": "one", "top": 1})["items"][0]
        (comp,) = circuit["components"]
        out = ablate_circuits(tiny, [circuit], RECORDS, {"tracked": TRACKED})
        cells = lexicon.items_of(out["cells"])
        drop = np.mean([c["m_full"] - c["m_without"] for c in cells])
        cell = grid["measures"]["mean_delta"][comp["layer"]][comp["head"]]
        assert drop == pytest.approx(-cell, abs=6e-5)
        assert [c["m_full"] for c in out["out"]["conditions"]] == [
            c["baseline_logp"] for c in grid["conditions"]]
        for c, record in zip(cells, RECORDS, strict=True):
            assert c["m_without"] == pytest.approx(read_by_hand(tiny, record, [(comp["layer"], comp["head"])]),
                                                   abs=1e-6)

    def test_faithfulness_and_completeness_against_a_computation_by_hand(self, tiny):
        heads = [(0, 1), (2, 3)]
        out = ablate_circuits(tiny, [hand_circuit("h", heads)], RECORDS, {"tracked": TRACKED})
        item = lexicon.items_of(out["out"])[0]
        full = np.array([read_by_hand(tiny, r, []) for r in RECORDS])
        empty = np.array([read_by_hand(tiny, r, ALL_HEADS) for r in RECORDS])
        alone = np.array([read_by_hand(tiny, r, [h for h in ALL_HEADS if h not in heads]) for r in RECORDS])
        without = np.array([read_by_hand(tiny, r, heads) for r in RECORDS])
        gap = full.mean() - empty.mean()
        assert item["faithfulness"] == pytest.approx((alone.mean() - empty.mean()) / gap, abs=1e-4)
        assert item["completeness"] == pytest.approx((full.mean() - without.mean()) / gap, abs=1e-4)
        assert item["m_empty"] == pytest.approx(empty.mean(), abs=1e-4)
        assert item["size"] == 2 and item["n"] == 2 and out["out"]["held_out"] is False
        assert item["faithfulness_lo"] <= item["faithfulness"] <= item["faithfulness_hi"]

    def test_the_whole_universe_is_faithful_and_complete(self, tiny):
        out = ablate_circuits(tiny, [hand_circuit("all", ALL_HEADS)], RECORDS, {"tracked": TRACKED})
        item = lexicon.items_of(out["out"])[0]
        assert item["faithfulness"] == 1.0 and item["completeness"] == 1.0

    def test_entropy_is_read_in_bits(self, tiny):
        heads = [(1, 0)]
        out = ablate_circuits(tiny, [hand_circuit("h", heads)], RECORDS[:1], {"metric": "entropy"})
        cell = lexicon.items_of(out["cells"])[0]
        assert cell["m_full"] == pytest.approx(read_by_hand(tiny, RECORDS[0], [], "entropy"), abs=1e-6)
        assert cell["m_without"] == pytest.approx(read_by_hand(tiny, RECORDS[0], heads, "entropy"), abs=1e-6)
        assert out["out"]["metric"] == "entropy" and out["out"]["held_out"] is True

    def test_mean_ablation_at_a_branch_replaces_it_with_its_mean_over_the_records(self, tiny):
        universe = {"points": ["attn_out"], "layers": [1, 2], "positions": [-1]}
        circuit = {**hand_circuit("m", [], universe=universe),
                   "components": [{"address": "L1.attn_out@-1", "point": "attn_out", "layer": 1,
                                   "position": -1, "effect": 0.2}]}
        out = ablate_circuits(tiny, [circuit], RECORDS, {"tracked": TRACKED, "ablation": "mean"})
        names = ["blocks.1.attn_out", "blocks.2.attn_out"]
        acts = [tiny.run(render(tiny, r).array, capture=names).cache for r in RECORDS]
        means = {n: mx.mean(mx.stack([a[n][0, -1].astype(mx.float32) for a in acts]), axis=0) for n in names}

        def put_mean(name):
            def hook(act, info):
                mask = np.zeros(act.shape[1], dtype=bool)
                mask[-1] = True
                last = mx.array(mask).reshape(1, -1, 1)
                return mx.where(last, mx.broadcast_to(means[name].astype(act.dtype), act.shape), act)
            return hook

        cells = lexicon.items_of(out["cells"])
        for c, record in zip(cells, RECORDS, strict=True):
            ids = render(tiny, record).array
            lp = read_last_logp(tiny.run(ids, hooks={n: put_mean(n) for n in names}).logits)
            assert c["m_empty"] == pytest.approx(float(lp[12]), abs=1e-5)
            lp = read_last_logp(tiny.run(ids, hooks={names[0]: put_mean(names[0])}).logits)
            assert c["m_without"] == pytest.approx(float(lp[12]), abs=1e-5)


class TestAblateCircuitRefuses:
    def test_a_residual_component(self, tiny):
        universe = {"points": ["attn_out"], "layers": [1], "positions": "all"}
        circuit = {**hand_circuit("r", [], universe=universe),
                   "components": [{"address": "L1.resid_post@-1", "point": "resid_post", "layer": 1,
                                   "position": -1, "effect": 0.2}]}
        with pytest.raises(ValueError, match=r"L1\.resid_post@-1 of 'r': a residual location is not a component"):
            ablate_circuits(tiny, [circuit], RECORDS, {})

    def test_a_residual_universe(self, tiny):
        universe = {"points": ["resid_post"], "layers": [1], "positions": "all"}
        with pytest.raises(ValueError, match="'resid_post': a residual location is not a component"):
            ablate_circuits(tiny, [hand_circuit("r", [], universe=universe)], RECORDS, {})

    def test_mean_at_a_point_with_a_head_axis(self, tiny):
        with pytest.raises(ValueError, match="ablation 'mean' at 'attn.per_head_out': a per-head mean"):
            ablate_circuits(tiny, [hand_circuit("h", [(0, 1)])], RECORDS, {"ablation": "mean"})

    def test_a_point_the_architecture_does_not_declare(self, tiny):
        universe = {"points": ["mlp.act"], "layers": [1], "positions": "all"}
        with pytest.raises(ValueError, match="'mlp.act' is not one the 'gemma3' forward declares"):
            ablate_circuits(tiny, [hand_circuit("x", [], universe=universe)], RECORDS, {})

    def test_circuits_from_two_universes(self, tiny):
        other = {**HEAD_UNIVERSE, "layers": [0, 1]}
        with pytest.raises(ValueError, match="2 universes"):
            ablate_circuits(tiny, [hand_circuit("a", [(0, 1)]), hand_circuit("b", [(0, 1)], universe=other)],
                            RECORDS, {})

    def test_base_without_an_adapter(self, tiny):
        with pytest.raises(ValueError, match="reference 'base' is the model without its adapter, and no adapter"):
            ablate_circuits(tiny, [hand_circuit("h", [(0, 1)])], RECORDS, {"reference": "base"})
        with pytest.raises(ValueError, match="no adapter is bound"):
            ablate_circuit_op.run(Context(loaded=tiny), {"circuit": hand_circuit("h", [(0, 1)]), "records": RECORDS},
                                  {"reference": "base"})

    def test_an_unknown_reference(self, tiny):
        with pytest.raises(ValueError, match="unknown reference 'floor'"):
            ablate_circuits(tiny, [hand_circuit("h", [(0, 1)])], RECORDS, {"reference": "floor"})

    def test_a_component_outside_the_universe(self, tiny):
        universe = {**HEAD_UNIVERSE, "layers": [0, 1]}
        with pytest.raises(ValueError, match=r"L3\.attn\.per_head_out\.H0@all of 'a' is outside"):
            ablate_circuits(tiny, [hand_circuit("a", [(3, 0)], universe=universe)], RECORDS, {})


class TestAblateHeadsMetric:
    def test_logprob_is_the_default_and_reads_as_by_hand(self, tiny):
        params = {"layers": [1, 3], "tracked": TRACKED}
        out = ablate_heads(tiny, RECORDS, params)
        assert rm.content_hash(out) == rm.content_hash(ablate_heads(tiny, RECORDS, {**params, "metric": "logprob"}))
        base = [read_by_hand(tiny, r, []) for r in RECORDS]
        assert [c["baseline_logp"] for c in out["conditions"]] == [round(b, 4) for b in base]
        cut = np.mean([read_by_hand(tiny, r, [(3, 2)]) - b for r, b in zip(RECORDS, base, strict=True)])
        assert out["measures"]["mean_delta"][1][2] == pytest.approx(cut, abs=1e-4)

    def test_entropy_against_a_computation_by_hand(self, tiny):
        out = ablate_heads(tiny, RECORDS, {"layers": [2], "metric": "entropy"})
        assert out["metric"] == "entropy"
        base = [read_by_hand(tiny, r, [], "entropy") for r in RECORDS]
        assert [c["baseline"] for c in out["conditions"]] == [round(b, 4) for b in base]
        cut = np.mean([read_by_hand(tiny, r, [(2, 1)], "entropy") - b for r, b in zip(RECORDS, base, strict=True)])
        assert out["measures"]["mean_delta"][0][1] == pytest.approx(cut, abs=1e-4)

    def test_an_unknown_metric_is_refused(self, tiny):
        with pytest.raises(ValueError, match="unknown metric 'nats'"):
            ablate_heads(tiny, RECORDS, {"metric": "nats"})


def build_adapter(model, rank=2):
    keys = iter(mx.random.split(mx.random.key(5), 100))
    weights = {}
    for i in range(len(model.lm.model.layers)):
        for proj in ("q_proj", "o_proj"):
            out, d = getattr(model.lm.model.layers[i].self_attn, proj).weight.shape
            weights[f"model.layers.{i}.self_attn.{proj}.lora_a"] = mx.random.normal((rank, d), key=next(keys))
            weights[f"model.layers.{i}.self_attn.{proj}.lora_b"] = mx.random.normal((out, rank), key=next(keys))
    return weights


def read_plain(model, record, params):
    ids = render(model, record).array
    logits = model.run(ids).logits
    answer, _ = resolve_target(model, record, params, read_last_logp(logits))
    return answer, logits


@pytest.fixture(scope="module")
def adapted():
    model = build_tiny_model("gemma3")
    model.node_adapter = lora.fuse(model.lm, build_adapter(model), scale=1.0)
    return model


class TestAblateCircuitReference:
    @pytest.mark.parametrize("metric", ["logprob", "entropy"])
    def test_base_is_a_plain_pass_of_the_model_without_its_adapter(self, adapted, tiny, metric):
        params = {"tracked": TRACKED, "metric": metric, "reference": "base"}
        heads = [(0, 1), (2, 3)]
        out = ablate_circuits(adapted, [hand_circuit("h", heads)], RECORDS, params, base=adapted.unadapted)
        cells = lexicon.items_of(out["cells"])
        for c, cond, record in zip(cells, out["out"]["conditions"], RECORDS, strict=True):
            answer, full = read_plain(adapted, record, params)
            floor = read_metric(answer, metric, read_plain(tiny, record, params)[1])
            assert c["m_empty"] == round(floor, 6) and cond["m_empty"] == round(floor, 4)
            assert c["m_full"] == round(read_metric(answer, metric, full), 6)
            assert c["m_empty"] != c["m_full"]
        item = lexicon.items_of(out["out"])[0]
        full = np.array([c["m_full"] for c in cells])
        floor = np.array([c["m_empty"] for c in cells])
        alone = np.array([c["m_circuit"] for c in cells])
        assert item["faithfulness"] == pytest.approx((alone.mean() - floor.mean()) / (full.mean() - floor.mean()),
                                                     abs=1e-4)
        assert out["out"]["reference"] == "base" and out["cells"]["reference"] == "base"

    def test_the_adapter_is_back_after_the_base_pass(self, adapted):
        before = [read_plain(adapted, r, {})[1] for r in RECORDS]
        ablate_circuits(adapted, [hand_circuit("h", [(1, 0)])], RECORDS, {"reference": "base"},
                        base=adapted.unadapted)
        after = [read_plain(adapted, r, {})[1] for r in RECORDS]
        assert all(mx.array_equal(a, b).item() for a, b in zip(before, after, strict=True))

    @pytest.mark.parametrize("metric", ["logprob", "entropy"])
    def test_empty_is_the_default_and_its_floor_is_the_universe_removed(self, tiny, metric):
        heads = [(0, 1), (2, 3)]
        params = {"tracked": TRACKED, "metric": metric}
        out = ablate_circuits(tiny, [hand_circuit("h", heads)], RECORDS, params)
        named = ablate_circuits(tiny, [hand_circuit("h", heads)], RECORDS, {**params, "reference": "empty"})
        assert out["out"]["reference"] == "empty" and out["cells"]["reference"] == "empty"
        assert rm.content_hash(out["out"]) == rm.content_hash(named["out"])
        assert rm.content_hash(out["cells"]) == rm.content_hash(named["cells"])
        removal = compile_intervention(tiny, [{"point": "attn.per_head_out", "layers": [layer], "heads": [0, 1, 2, 3],
                                               "positions": "all", "op": "zero"} for layer in range(4)])
        cells = lexicon.items_of(out["cells"])
        for c, cond, record in zip(cells, out["out"]["conditions"], RECORDS, strict=True):
            answer, _ = read_plain(tiny, record, params)
            ids = render(tiny, record).array
            tokens = [tiny.tokenizer.decode([int(t)]) for t in np.array(ids).reshape(-1)]
            iv = SpecIntervention(removal.specs, tokens, record)
            floor = read_metric(answer, metric, tiny.run(ids, interventions=[iv]).logits)
            assert c["m_empty"] == round(floor, 6) and cond["m_empty"] == round(floor, 4)
        full = np.array([read_by_hand(tiny, r, [], metric) for r in RECORDS])
        empty = np.array([read_by_hand(tiny, r, ALL_HEADS, metric) for r in RECORDS])
        alone = np.array([read_by_hand(tiny, r, [h for h in ALL_HEADS if h not in heads], metric) for r in RECORDS])
        without = np.array([read_by_hand(tiny, r, heads, metric) for r in RECORDS])
        gap = full.mean() - empty.mean()
        item = lexicon.items_of(out["out"])[0]
        assert item["faithfulness"] == pytest.approx((alone.mean() - empty.mean()) / gap, abs=1e-3)
        assert item["completeness"] == pytest.approx((full.mean() - without.mean()) / gap, abs=1e-3)

    def test_the_sentence_names_the_base_model_and_only_it(self, adapted, tiny):
        speak = K.BY_KIND["intervene/faithfulness"].speak
        said = {}
        for reference, model in (("base", adapted), ("empty", tiny)):
            out = ablate_circuits(model, [hand_circuit("h", [(1, 0)])], RECORDS, {"reference": reference},
                                  base=getattr(model, "unadapted", None) if reference == "base" else None)
            header = {k: v for k, v in out["out"].items() if k != "items"}
            said[reference] = load_engine().render(speak, out["out"]["items"], header=header).values[0]
        assert said["base"].endswith("logprob, against the base model.")
        assert said["empty"].endswith("zero ablation, logprob.")

    def test_the_executor_lends_the_node_adapter_and_takes_it_back(self, tmp_path):
        model = build_tiny_model("gemma3")
        path = tmp_path / "a.safetensors"
        mx.save_safetensors(str(path), build_adapter(model))
        payload = {"data": path.read_bytes(), "lora": {"rank": 2, "alpha": 2}}
        assert model.node_adapter is None
        with ModelLoading()._adapter_fused(model, {"adapter": payload}, {}):
            assert model.node_adapter is not None
            ctx = Context(loaded=model)
            out = ablate_circuit_op.run(ctx, {"circuit": hand_circuit("h", [(1, 0)]), "records": RECORDS},
                                        {"reference": "base"})
        assert model.node_adapter is None and out["out"]["reference"] == "base"


class TestTheKinds:
    def test_both_extend_the_record_and_key_on_id(self):
        for name in ("intervene/circuit", "intervene/faithfulness"):
            kind = K.BY_KIND[name]
            assert kind.extends == "records/record" and kind.key == ("id",) and kind.speak
        assert K.BY_KIND["intervene/circuit"].draw is None
        assert K.BY_KIND["intervene/faithfulness"].draw.encoding == {
            "x": "circuit", "y": "faithfulness", "lo": "faithfulness_lo", "hi": "faithfulness_hi"}
