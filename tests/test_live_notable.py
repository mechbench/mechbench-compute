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
from mechbench_compute.live.read_notable import NONE_YET, WITHIN_FLOOR
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


def codes(notable) -> list[str]:
    return [c["code"] for c in notable["caveats"]]


def caveat(notable, code: str) -> str:
    return next(c["line"] for c in notable["caveats"] if c["code"] == code)


def test_a_first_reading_has_nothing_to_compare_yet(warm):
    executor, model = warm()
    notable = read(executor, model, A)["notable"]
    assert notable["line"] == NONE_YET == "first reading here; nothing to compare yet"
    assert notable["moved"] is False and notable["baseline"] is None


def test_a_try_past_the_floor_and_the_threshold_moves(warm):
    executor, model = warm()
    base = read(executor, model, A)
    got = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
               noise=floor_of("llama", "tracked.c.p", 1e-4))
    notable = got["notable"]
    assert notable["moved"] is True
    assert notable["line"].startswith("against $base, on ")
    assert notable["line"].endswith(" floors, past 0.1)") and "total variation " in notable["line"]
    assert notable["baseline"] == {"label": "$base", "origin": "declared", "seq": None, "param": None}
    assert "NO_FLOOR" not in codes(notable) and WITHIN_FLOOR not in [c["line"] for c in notable["caveats"]]
    many = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
                noise=floor_of("llama", "tracked.c.p", 1e-4), k=1e9)
    assert many["notable"]["moved"] is False
    assert many["notable"]["line"].startswith("against $base, within the floor: ")


def lens_against(executor, model, shift: float, spread: float | None):
    lens = run_try(executor, op="logits/read-layers", inputs={"records": A}, params={"top_k": 3}, model=model)
    earlier = copy.deepcopy(lens["result"])
    item = next(it for it in earlier["items"] if it["id"] == "a" and it["layer"] == 2)
    now = item["entropy_bits"]
    item["entropy_bits"] = now + shift
    noise = None if spread is None else floor_of("llama", "entropy_bits", spread, "logits/read-layers")
    got = run_try(executor, op="logits/read-layers", inputs={"records": A}, params={"top_k": 3}, model=model,
                  baseline={"name": "lens", "result": earlier}, noise=noise)
    clause = f"at layer 2 of a the entropy falls from {say_number(now + shift)} to {say_number(now)} bits"
    return got["notable"], clause


@pytest.mark.parametrize(("shift", "spread", "said", "moved"), [
    (0.1, 0.0, "below the threshold: {clause} (above a floor of 0, under 0.5)", False),
    (1.0, 0.0, "{clause} (above a floor of 0, past 0.5)", True),
    (0.1, 0.01, "below the threshold: {clause} (10 floors, under 0.5)", False),
    (1.0, 0.01, "{clause} (100 floors, past 0.5)", True),
    (3.0, 10.0, "within the floor: {clause} (0.3 floors)", False),
    (0.4, 10.0, "within the floor: {clause} (0.04 floors)", False),
])
def test_moved_is_past_k_floors_and_past_the_threshold(warm, shift, spread, said, moved):
    executor, model = warm()
    notable, clause = lens_against(executor, model, shift, spread)
    assert notable["line"] == "against $lens, " + said.format(clause=clause)
    assert notable["moved"] is moved
    assert "NO_FLOOR" not in codes(notable)
    assert ("WITHIN_FLOOR" in codes(notable)) is said.startswith("within the floor")


def test_a_try_within_the_floor_says_so_and_names_its_baseline(warm):
    executor, model = warm()
    base = read(executor, model, A)
    same = read(executor, model, A, baseline={"name": "base", "result": base["result"]},
                noise=floor_of("llama", "tracked.c.p", 1e-4))["notable"]
    assert same["moved"] is False
    assert same["line"].startswith("against $base, within the floor: on ")
    assert same["line"].endswith("(total variation 0, 0 floors)")
    assert "WITHIN_FLOOR" not in codes(same)
    unfloored = read(executor, model, B, baseline={"name": "base", "result": base["result"]})["notable"]
    assert unfloored["moved"] is True
    wide = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
                noise=floor_of("llama", "tracked.*.p", 1.0))["notable"]
    assert wide["moved"] is False
    assert wide["line"].startswith("against $base, within the floor: ")
    assert caveat(wide, "WITHIN_FLOOR") == "this difference is within the floor"


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
    assert notable["moved"] is True and notable["line"].endswith(", past 0.1)")
    assert caveat(notable, "NO_FLOOR") == "no noise floor: moved is judged against the threshold, 0.1"
    elsewhere = read(executor, model, B, baseline={"name": "base", "result": base["result"]},
                     noise=floor_of("gemma4", "tracked.c.p", 1e-4))["notable"]
    assert caveat(elsewhere, "NO_FLOOR") == (
        "the noise floor has no record for llama logits/read tracked.*.p: "
        "moved is judged against the threshold, 0.1")
    still = read(executor, model, A, baseline={"name": "base", "result": base["result"]})["notable"]
    assert still["moved"] is False
    assert still["line"].startswith("against $base, below the threshold: ")
    assert still["line"].endswith("(total variation 0, under 0.1)")


def test_a_saturated_softcap_read_is_a_caveat(warm):
    executor, model = warm("gemma4-small-cap")
    got = read(executor, model, A)
    saturated = [t for it in got["result"]["items"] for t in it["tracked"].values() if t.get("saturated")]
    assert saturated
    assert caveat(got["notable"], "SATURATED") == (
        f"{len(saturated)} tracked answers are saturated by the final softcap: the cap, not the model's "
        "margin, sets their probabilities")
    executor, model = warm("llama")
    assert "SATURATED" not in codes(read(executor, model, A)["notable"])


def test_a_kind_without_a_notable_line_says_none(warm):
    executor, model = warm()
    got = run_try(executor, op="activations/capture", inputs={"records": A}, params={"layers": [1]},
                  model=model)
    assert got["kind"] == "activations/vector" and K.BY_KIND["activations/vector"].notable is None
    assert got["notable"] is None and got["lines"]


def test_the_line_says_what_the_kind_declares(warm):
    executor, model = warm()
    lens = run_try(executor, op="logits/read-layers", inputs={"records": A}, params={"top_k": 3}, model=model)
    earlier = copy.deepcopy(lens["result"])
    item = next(it for it in earlier["items"] if it["id"] == "a" and it["layer"] == 2)
    now = item["entropy_bits"]
    item["entropy_bits"] = now + 1.0
    got = run_try(executor, op="logits/read-layers", inputs={"records": A}, params={"top_k": 3}, model=model,
                  baseline={"name": "lens", "result": earlier})
    declared = K.BY_KIND["logits/funnel"].notable
    assert declared.line == "at layer {layer} of {id} the entropy {change} bits"
    change = f"falls from {say_number(now + 1.0)} to {say_number(now)}"
    assert got["notable"]["line"] == f"against $lens, {declared.line.format(layer=2, id='a', change=change)} (past 0.5)"
    assert got["notable"]["moved"] is True

    read_now = read(executor, model, A)
    earlier = copy.deepcopy(read_now["result"])
    answer = next(it for it in earlier["items"] if it["id"] == "b")["tracked"]["d"]
    p = answer["p"]
    answer["p"] = p + 0.25
    got = read(executor, model, A, baseline={"name": "base", "result": earlier})
    decision = K.BY_KIND["logits/decision"].notable
    clause = decision.line.format(id="b", name="d", change=f"falls from {say_number(p + 0.25)} to {say_number(p)}")
    assert got["notable"]["line"] == f"against $base, below the threshold: {clause} (total variation 0, under 0.1)"
    floored = read(executor, model, A, baseline={"name": "base", "result": earlier},
                   noise=floor_of("llama", "tracked.*.p", 0.01))
    assert floored["notable"]["line"] == (
        f"against $base, below the threshold: {clause} (total variation 0, 25 floors, under 0.1)")
    assert floored["notable"]["moved"] is False


def test_an_attribution_names_the_piece_that_moved_and_what_the_pieces_build(warm):
    executor, model = warm()
    for tracked, built in (({"x": "mat", "y": "dog"}, "margin"), ({"x": "mat"}, "logit")):
        first = run_try(executor, op="logits/attribute", inputs={"records": A}, params={"tracked": tracked},
                        model=model)
        got = run_try(executor, op="logits/attribute", inputs={"records": B}, params={"tracked": tracked},
                      model=model, baseline={"name": "first", "result": first["result"]})
        line = got["notable"]["line"]
        piece = line.split(" the ", 1)[1].split(" piece ", 1)[0]
        assert piece in got["result"]["components"], line
        assert f" piece of the {built} " in line
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
    assert notable["line"].startswith("against factor 0, ")
    assert " at factor 2 on " in notable["line"] or " at factor -2 on " in notable["line"]


def test_a_collection_says_how_many_records_it_holds():
    rows = [{"id": i} for i in "abc"]
    first = run_try(ProtocolExecutor(), op="records/filter", inputs={"records": rows},
                    params={"where": "id != 'a'"}, model="tiny/llama", seq=1)
    got = run_try(ProtocolExecutor(), op="records/filter", inputs={"records": rows},
                  params={"where": "id == 'a'"}, model="tiny/llama", seq=2,
                  tries=[{"seq": 1, "op": "records/filter", "inputs": {"records": rows},
                          "params": {"where": "id != 'a'"}, "result": first["result"]}])
    notable = got["notable"]
    assert notable["line"] == "against t1, the number of records falls from 2 to 1 (past 0)"
    assert notable["moved"] is True
    assert notable["baseline"] == {"label": "t1", "origin": "try", "seq": 1, "param": "where"}
    assert codes(notable) == ["FEW_ITEMS"]


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


def test_a_baseline_from_another_machine_is_a_margin_line(warm):
    executor, model = warm()
    base = read(executor, model, A)
    got = read(executor, model, A, baseline={"name": "base", "result": base["result"], "machine": "mac-2"},
               machine="mac-3")
    assert caveat(got["notable"], "OTHER_MACHINE") == (
        "the baseline was read on mac-2 and this on mac-3: part of any difference may be the machines'")
    here = read(executor, model, A, baseline={"name": "base", "result": base["result"], "machine": "mac-3"},
                machine="mac-3")
    assert "OTHER_MACHINE" not in codes(here["notable"])


def test_nothing_to_compare_when_no_item_shares_its_key(warm):
    executor, model = warm()
    base = read(executor, model, [{"id": "z", "user": "the cat sat"}])
    notable = read(executor, model, A, baseline={"name": "base", "result": base["result"]})["notable"]
    assert notable["line"] == "against $base, nothing to compare: no item carries tracked.*.p on both sides"
    assert notable["moved"] is False


def test_the_caveats_come_from_fields_the_result_carries():
    cut = [{"id": f"r{i}", "metadata": {"sampling": {"ended": "max_tokens" if i == 0 else "stop"}}}
           for i in range(3)]
    assert read_caveats(cut, {"n_off_top1": 2}) == [
        {"code": "CUT", "line": "1 record was cut short of its end"},
        {"code": "OFF_TOP1", "line": "2 records track a target the model would not say itself (`own_top1`)"},
        {"code": "FEW_ITEMS", "line": "3 records, fewer than 8"},
    ]
    probes = [{"id": f"p{i}", "accuracy_test": 0.4, "baseline": 0.5, "over_baseline": -0.1, "n_items": 6}
              for i in range(2)]
    assert read_caveats(probes, {}) == [
        {"code": "UNDER_MAJORITY", "line": "2 probes score under the majority baseline"},
        {"code": "FEW_ITEMS", "line": "6 items, fewer than 8"},
    ]
    assert read_caveats([{"id": "x", "truncated": True}, {"id": "y", "own_top1": {"id": 3}}], {})[:2] == [
        {"code": "CUT", "line": "1 record was cut short of its end"},
        {"code": "OFF_TOP1", "line": "1 record tracks a target the model would not say itself (`own_top1`)"},
    ]
    assert read_caveats([{"id": f"r{i}"} for i in range(8)], {}) == []


@pytest.mark.parametrize(("x", "said"), [(0.48333, "0.483"), (9.0, "9"), (12, "12"), (0.0004, "0"),
                                         (-0.4836, "-0.484"), (0.1666, "0.167"), (2.5, "2.5"), (-3.0, "-3")])
def test_numbers_print_as_the_readings_print_them(x, said):
    assert say_number(x) == said
