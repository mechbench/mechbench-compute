from __future__ import annotations

import copy
import dataclasses

import pytest

from mechbench_compute import architectures
from mechbench_compute import model as model_mod
from mechbench_compute.api import run_try
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.live.choose_baseline import choose_baseline
from mechbench_compute.live.read_caveats import read_caveats
from mechbench_compute.live.read_notable import MOST_CHANGES
from mechbench_compute.live.say_number import say_number
from mechbench_compute.protocol import ProtocolExecutor
from tests.tiny_models import KIT_MODELS, build_tiny_model

A = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]
B = [{"id": "a", "user": "a dog ran"}, {"id": "b", "user": "the cat sat"}]
TRACKED = {"c": "cat", "d": "dog"}
LABELLED = [{"id": f"v{i}", "text": t, "coords": {"label": label}}
            for i, (t, label) in enumerate([("the cat sat", "pos"), ("a cat ran", "pos"),
                                            ("the dog sat", "neg"), ("a dog ran", "neg")])]


@pytest.fixture
def warm(monkeypatch, tmp_path):
    def load(name: str = "llama"):
        model_type = dict(KIT_MODELS)[name]
        tiny = build_tiny_model(name, architectures.BY_MODEL_TYPE[model_type])
        architecture = dataclasses.replace(architectures.BY_MODEL_TYPE[model_type],
                                           load=lambda model_id, **_: (tiny._model, tiny._processor))
        monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                            lambda model_id, **_: (model_id, "0" * 40, tmp_path))
        monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": model_type})
        monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
        executor = ProtocolExecutor()
        executor._model_loaded(f"tiny/{name}")
        return executor, f"tiny/{name}"

    return load


def read(executor, model, records, **given):
    return run_try(executor, op="logits/read", inputs={"conditions": records}, params={"tracked": TRACKED},
                   model=model, **given)


def floor_of(architecture: str, field: str, spread: float, operation: str = "logits/read"):
    return K.collection("platform/noise", [{
        "id": f"{architecture}:{operation}:{field}", "architecture": architecture, "model": "",
        "operation": operation, "field": field, "dtype": "bfloat16", "machine_class": "apple-silicon",
        "spread": spread, "relative_spread": 0.0, "n": 4, "records": 2, "machines": ["one", "two"],
        "seeds": []}])


CHANGE = {"key", "field", "index", "before", "after", "difference", "metric", "distance", "floors", "floor",
          "threshold"}


def codes(notable) -> list[str]:
    return [c["code"] for c in notable["caveats"]]


def caveat(notable, code: str) -> dict:
    return next(c for c in notable["caveats"] if c["code"] == code)


def test_a_first_reading_has_no_state_and_no_changes(warm):
    executor, model = warm()
    notable = read(executor, model, A)["notable"]
    assert notable == {"state": None, "baseline": None, "compared": 0, "changes": [], "caveats": notable["caveats"]}
    assert codes(notable) == ["FEW_ITEMS"]


def test_a_try_past_the_floor_and_the_threshold_moves(warm):
    executor, model = warm()
    base = read(executor, model, A)
    got = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
               noise=floor_of("llama", "tracked.c.p", 1e-4))
    notable = got["notable"]
    assert set(notable) == {"state", "baseline", "compared", "changes", "caveats"}
    assert notable["state"] == "moved"
    assert notable["baseline"] == {"label": "$base", "origin": "declared", "seq": None, "param": None}
    top = notable["changes"][0]
    assert set(top) == CHANGE
    assert top["metric"] == "total-variation" and top["distance"] > 0.1 and top["threshold"] == 0.1
    assert top["floor"] == 1e-4 and top["floors"] > 1.0
    assert top["difference"] == pytest.approx(top["after"] - top["before"])
    assert top["field"] in ("tracked.c.p", "tracked.d.p") and top["key"]["id"] in ("a", "b")
    assert notable["compared"] == 4 == len(notable["changes"])
    assert "NO_FLOOR" not in codes(notable)
    many = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
                noise=floor_of("llama", "tracked.c.p", 1e-4), k=1e9)
    assert many["notable"]["state"] == "noise"
    assert many["notable"]["changes"][0]["floors"] == pytest.approx(top["floors"])


def lens_against(executor, model, shift: float, spread: float | None):
    lens = run_try(executor, op="logits/read-layers", inputs={"records": A}, params={"top_k": 3}, model=model)
    earlier = copy.deepcopy(lens["result"])
    item = next(it for it in earlier["items"] if it["id"] == "a" and it["layer"] == 2)
    now = item["entropy_bits"]
    item["entropy_bits"] = now + shift
    noise = None if spread is None else floor_of("llama", "entropy_bits", spread, "logits/read-layers")
    got = run_try(executor, op="logits/read-layers", inputs={"records": A}, params={"top_k": 3}, model=model,
                  baseline={"name": "lens", "result": earlier}, noise=noise)
    return got["notable"], now


@pytest.mark.parametrize(("shift", "spread", "state", "floors"), [
    (0.1, 0.0, "small", None),
    (1.0, 0.0, "moved", None),
    (0.1, 0.01, "small", 10.0),
    (1.0, 0.01, "moved", 100.0),
    (3.0, 10.0, "noise", 0.3),
    (0.4, 10.0, "noise", 0.04),
])
def test_moved_is_past_k_floors_and_past_the_threshold(warm, shift, spread, state, floors):
    executor, model = warm()
    notable, now = lens_against(executor, model, shift, spread)
    assert notable["state"] == state
    assert notable["changes"][0] == {
        "key": {"id": "a", "layer": 2}, "field": "entropy_bits", "index": None,
        "before": now + shift, "after": now, "difference": pytest.approx(-shift), "metric": "difference",
        "distance": pytest.approx(shift), "floors": None if floors is None else pytest.approx(floors),
        "floor": spread, "threshold": 0.5}
    assert all(c["difference"] == 0 for c in notable["changes"][1:])
    assert len(notable["changes"]) == notable["compared"] == 8
    assert "NO_FLOOR" not in codes(notable)


def test_the_changes_are_the_largest_first_and_at_most_ten(warm, monkeypatch):
    assert MOST_CHANGES == 10
    monkeypatch.setattr("mechbench_compute.live.read_notable.MOST_CHANGES", 3)
    executor, model = warm()
    notable, now = lens_against(executor, model, 1.0, None)
    assert notable["compared"] == 8 and len(notable["changes"]) == 3
    assert notable["changes"][0]["key"] == {"id": "a", "layer": 2}


def test_a_try_within_the_floor_is_noise_and_names_its_baseline(warm):
    executor, model = warm()
    base = read(executor, model, A)
    same = read(executor, model, A, baseline={"name": "base", "result": base["result"]},
                noise=floor_of("llama", "tracked.c.p", 1e-4))["notable"]
    assert same["state"] == "noise"
    assert {(c["difference"], c["distance"], c["floors"]) for c in same["changes"]} == {(0.0, 0.0, 0.0)}
    assert same["baseline"]["label"] == "$base"
    unfloored = read(executor, model, B, baseline={"name": "base", "result": base["result"]})["notable"]
    assert unfloored["state"] == "moved"
    wide = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
                noise=floor_of("llama", "tracked.*.p", 1.0))["notable"]
    assert wide["state"] == "noise"
    assert wide["changes"][0]["floor"] == 1.0 and wide["changes"][0]["floors"] < 1.0


def test_a_tries_result_and_hash_do_not_depend_on_its_baseline_or_floor(warm):
    executor, model = warm()
    alone = read(executor, model, A, seq=7)
    other = read(executor, model, B, seq=7)
    told = [
        {"baseline": {"name": "base", "result": other["result"], "machine": "elsewhere"}, "machine": "here"},
        {"baseline": {"name": "base", "result": other["result"]}, "noise": floor_of("llama", "tracked.*.p", 0.0)},
        {"baseline": {"name": "base", "result": other["result"]}, "noise": floor_of("llama", "tracked.*.p", 9.0),
         "k": 3.0},
        {"tries": [{"seq": 1, "op": "logits/read", "inputs": {"conditions": A},
                    "params": {"tracked": {"c": "cat"}}, "result": other["result"]}]},
    ]
    for given in told:
        got = read(executor, model, A, seq=7, **given)
        assert got["notable"]["baseline"] is not None
        assert got["hash"] == alone["hash"] and got["result"] == alone["result"], given


def test_with_no_floor_the_threshold_judges_and_a_caveat_says_so(warm):
    executor, model = warm()
    base = read(executor, model, A)
    notable = read(executor, model, B, baseline={"name": "base", "result": base["result"]})["notable"]
    assert notable["state"] == "moved"
    assert {(c["floors"], c["floor"]) for c in notable["changes"]} == {(None, None)}
    assert caveat(notable, "NO_FLOOR") == {"code": "NO_FLOOR", "noise": False, "architecture": "llama",
                                           "operation": "logits/read", "field": "tracked.*.p"}
    elsewhere = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
                     noise=floor_of("gemma4", "tracked.c.p", 1e-4))["notable"]
    assert caveat(elsewhere, "NO_FLOOR") == {"code": "NO_FLOOR", "noise": True, "architecture": "llama",
                                             "operation": "logits/read", "field": "tracked.*.p"}
    still = read(executor, model, A, baseline={"name": "base", "result": base["result"]})["notable"]
    assert still["state"] == "small"
    assert still["changes"][0]["distance"] == 0 and still["changes"][0]["threshold"] == 0.1


def test_a_saturated_softcap_read_is_a_caveat_with_its_count(warm):
    executor, model = warm("gemma4-small-cap")
    got = read(executor, model, A)
    saturated = [t for it in got["result"]["items"] for t in it["tracked"].values() if t.get("saturated")]
    assert saturated
    assert caveat(got["notable"], "SATURATED") == {"code": "SATURATED", "count": len(saturated)}
    executor, model = warm("llama")
    assert "SATURATED" not in codes(read(executor, model, A)["notable"])


def test_a_kind_without_a_notable_says_none(warm):
    executor, model = warm()
    got = run_try(executor, op="activations/capture", inputs={"records": A}, params={"layers": [1]},
                  model=model)
    assert got["kind"] == "activations/vector" and K.BY_KIND["activations/vector"].notable is None
    assert got["notable"] is None and got["lines"]


def test_a_change_names_its_item_and_field_and_carries_the_raw_numbers(warm):
    executor, model = warm()
    read_now = read(executor, model, A)
    earlier = copy.deepcopy(read_now["result"])
    answer = next(it for it in earlier["items"] if it["id"] == "b")["tracked"]["d"]
    p = answer["p"]
    answer["p"] = p + 0.25
    got = read(executor, model, A, baseline={"name": "base", "result": earlier})["notable"]
    top = got["changes"][0]
    assert got["state"] == "small"
    assert (top["key"], top["field"], top["index"]) == ({"id": "b"}, "tracked.d.p", None)
    assert (top["before"], top["after"], top["difference"]) == (p + 0.25, p, pytest.approx(-0.25))
    assert top["metric"] == "total-variation" and top["distance"] < 0.1
    floored = read(executor, model, A, baseline={"name": "base", "result": earlier},
                   noise=floor_of("llama", "tracked.*.p", 0.01))["notable"]
    assert floored["state"] == "small"
    assert (floored["changes"][0]["floors"], floored["changes"][0]["floor"]) == (pytest.approx(25.0), 0.01)


def test_an_attribution_names_the_piece_that_moved_by_its_index(warm):
    executor, model = warm()
    for tracked in ({"x": "mat", "y": "dog"}, {"x": "mat"}):
        first = run_try(executor, op="logits/attribute", inputs={"records": A}, params={"tracked": tracked},
                        model=model)
        got = run_try(executor, op="logits/attribute", inputs={"records": B}, params={"tracked": tracked},
                      model=model, baseline={"name": "first", "result": first["result"]})
        top = got["notable"]["changes"][0]
        assert top["field"] == "measures.contribution" and set(top["key"]) == {"id"}
        assert 0 <= top["index"] < len(got["result"]["components"])
        assert "OFF_TOP1" in codes(got["notable"]) or got["result"].get("n_off_top1") == 0


def test_a_steer_try_reads_against_its_strength_zero_control(warm):
    executor, model = warm()
    vectors = run_try(executor, op="activations/capture", inputs={"records": LABELLED},
                      params={"layers": [1], "point": "resid_post"}, model=model)["result"]
    got = run_try(executor, op="intervene/steer", inputs={"records": A, "vectors": vectors},
                  params={"layer": 1, "direction": {"axis": "label", "positive": "pos", "negative": "neg"},
                          "alphas": [-2.0, 0.0, 2.0], "tracked": {"c": "cat"}, "top_k": 4}, model=model)
    notable = got["notable"]
    assert notable["baseline"] == {"label": "factor 0", "origin": "control", "seq": None, "param": None}
    assert {c["key"]["factor"] for c in notable["changes"]} <= {2.0, -2.0}
    assert notable["changes"][0]["field"] == "tracked.c.p"


def test_a_collection_counts_its_records():
    rows = [{"id": i} for i in "abc"]
    first = run_try(ProtocolExecutor(), op="records/filter", inputs={"records": rows},
                    params={"where": "id != 'a'"}, model="tiny/llama", seq=1)
    got = run_try(ProtocolExecutor(), op="records/filter", inputs={"records": rows},
                  params={"where": "id == 'a'"}, model="tiny/llama", seq=2,
                  tries=[{"seq": 1, "op": "records/filter", "inputs": {"records": rows},
                          "params": {"where": "id != 'a'"}, "result": first["result"]}])
    assert got["notable"] == {
        "state": "moved",
        "baseline": {"label": "t1", "origin": "try", "seq": 1, "param": "where"},
        "compared": 1,
        "changes": [{"key": {}, "field": "items", "index": None, "before": 2, "after": 1, "difference": -1,
                     "metric": "difference", "distance": 1, "floors": None, "floor": None, "threshold": 0}],
        "caveats": [{"code": "FEW_ITEMS", "count": 1, "unit": "record", "fewest": 8}],
    }


def test_the_baseline_is_declared_then_the_control_then_the_first_try_with_one_param_changed():
    funnel = K.BY_KIND["logits/funnel"].notable
    prompts = {"records": {"$name": "prompts"}}
    tries = [
        {"seq": 0, "op": "logits/read-layers", "inputs": prompts, "params": {"top_k": 1}, "result": None},
        {"seq": 1, "op": "logits/read-layers", "inputs": prompts, "params": {"top_k": 3}, "result": "r1",
         "name": "base"},
        {"seq": 2, "op": "logits/read-layers", "inputs": {"records": {"$name": "others"}},
         "params": {"top_k": 5}, "result": "r2"},
        {"seq": 3, "op": "logits/read-layers",
         "inputs": {"records": {"$ref": {"bench": "~scratch/live_x/prompts", "sha256": "ab"}}},
         "params": {"top_k": 5}, "result": "r3"},
    ]

    def choose(params, **given):
        return choose_baseline(funnel, [], tries=tries, op="logits/read-layers", inputs=prompts,
                               params=params, **given)

    first = choose({"top_k": 7})
    assert (first.label, first.origin, first.seq, first.param, first.result) == ("$base", "try", 1, "top_k", "r1")
    later = choose({"top_k": 3})
    assert (later.label, later.seq, later.result) == ("t3", 3, "r3")
    assert choose({"top_k": 7, "tracked": {"c": "cat"}}) is None
    declared = choose({"top_k": 7}, declared={"name": "mine", "result": "r9", "machine": "two"})
    assert (declared.label, declared.origin, declared.result, declared.machine) == ("$mine", "declared", "r9", "two")
    assert choose_baseline(funnel, [], tries=tries, op=None, inputs=prompts, params={"top_k": 7}) is None
    readout = K.BY_KIND["intervene/readout"].notable
    control = choose_baseline(readout, [{"id": "a", "factor": 0.0}, {"id": "a", "factor": 2.0}])
    assert (control.label, control.origin) == ("factor 0", "control")
    assert choose_baseline(readout, [{"id": "a", "factor": 2.0}]) is None


def test_a_baseline_from_another_machine_is_a_caveat_with_both_names(warm):
    executor, model = warm()
    base = read(executor, model, A)
    got = read(executor, model, A, baseline={"name": "base", "result": base["result"], "machine": "mac-2"},
               machine="mac-3")
    assert caveat(got["notable"], "OTHER_MACHINE") == {"code": "OTHER_MACHINE", "baseline_machine": "mac-2",
                                                       "machine": "mac-3"}
    here = read(executor, model, A, baseline={"name": "base", "result": base["result"], "machine": "mac-3"},
                machine="mac-3")
    assert "OTHER_MACHINE" not in codes(here["notable"])


def test_nothing_to_compare_when_no_item_shares_its_key(warm):
    executor, model = warm()
    base = read(executor, model, [{"id": "z", "user": "the cat sat"}])
    notable = read(executor, model, A, baseline={"name": "base", "result": base["result"]})["notable"]
    assert (notable["state"], notable["compared"], notable["changes"]) == (None, 0, [])
    assert notable["baseline"]["label"] == "$base"
    assert caveat(notable, "NOTHING_TO_COMPARE") == {"code": "NOTHING_TO_COMPARE", "field": "tracked.*.p"}


def test_the_caveats_carry_the_counts_the_result_carries():
    cut = [{"id": f"r{i}", "metadata": {"sampling": {"ended": "max_tokens" if i == 0 else "stop"}}}
           for i in range(3)]
    assert read_caveats(cut, {"n_off_top1": 2}) == [
        {"code": "CUT", "count": 1},
        {"code": "OFF_TOP1", "count": 2},
        {"code": "FEW_ITEMS", "count": 3, "unit": "record", "fewest": 8},
    ]
    probes = [{"id": f"p{i}", "accuracy_test": 0.4, "baseline": 0.5, "over_baseline": -0.1, "n_items": 6}
              for i in range(2)]
    assert read_caveats(probes, {}) == [
        {"code": "UNDER_MAJORITY", "count": 2},
        {"code": "FEW_ITEMS", "count": 6, "unit": "item", "fewest": 8},
    ]
    assert read_caveats([{"id": "x", "truncated": True}, {"id": "y", "own_top1": {"id": 3}}], {})[:2] == [
        {"code": "CUT", "count": 1},
        {"code": "OFF_TOP1", "count": 1},
    ]
    assert read_caveats([{"id": f"r{i}"} for i in range(8)], {}) == []


@pytest.mark.parametrize(("x", "said"), [(0.48333, "0.483"), (9.0, "9"), (12, "12"), (0.0004, "0"),
                                         (-0.4836, "-0.484"), (0.1666, "0.167"), (2.5, "2.5"), (-3.0, "-3")])
def test_numbers_print_as_the_readings_print_them(x, said):
    assert say_number(x) == said
