from __future__ import annotations

import json

import numpy as np
import pytest

from mechbench_compute import platform_kinds
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.lexicon.kinds.platform.noise import KIND
from mechbench_compute.ops.records.diff import diff_collections
from mechbench_compute.ops.records.measure_noise import measure_noise
from mechbench_compute.registry import SEALED

jsonschema = pytest.importorskip("jsonschema")

BF16_EPS = 2.0 ** -8


def to_bf16(x):
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32).astype(float)


def reads(machine_seed, jitter=0.0, shift=0.0):
    rng = np.random.default_rng(machine_seed)
    base = np.linspace(-6.0, -0.5, 8)
    items = []
    for i, lp in enumerate(base):
        noisy = lp * (1 + jitter * rng.uniform(-1, 1)) + shift
        vec = to_bf16(np.sin(np.arange(4) + i) * (1 + jitter * rng.uniform(-1, 1, 4)))
        items.append({"id": f"p{i}", "coords": {"prompt": f"p{i}", "layer": 3},
                      "logprob": float(to_bf16(noisy)), "vector": [float(v) for v in vec],
                      "token": "the", "latency_ms": 10 + machine_seed,
                      "metadata": {"created_at": f"2026-09-30T0{machine_seed}:00:00Z"}})
    return K.collection("records/record", items, model="tiny/llama@0000")


HOME, BOX, THIRD = reads(1, BF16_EPS), reads(2, BF16_EPS), reads(3, BF16_EPS)
PARAMS = {"architecture": "gemma4", "operation": "logits/read",
          "machines": ["Apple M4 Max", "Apple M2 Ultra", "Apple M4 Max"]}


def test_the_kind_is_a_sealed_platform_record_keyed_by_what_the_floor_holds_for():
    assert KIND.platform and KIND.family == "platform" and "platform" in SEALED
    assert KIND.key == ("architecture", "operation", "field", "dtype", "machine_class")
    assert set(KIND.key) <= set(KIND.fields) and K.satisfies("platform/noise", "records/record")


def test_the_floor_is_the_widest_spread_per_numeric_field_and_validates():
    floor = measure_noise([HOME, BOX, THIRD], PARAMS)
    by_field = {r["field"]: r for r in floor["items"]}
    assert set(by_field) == {"logprob", "vector"}
    lp = by_field["logprob"]
    values = np.array([[r["logprob"] for r in c["items"]] for c in (HOME, BOX, THIRD)])
    assert lp["spread"] == pytest.approx((values.max(0) - values.min(0)).max())
    assert 0 < lp["relative_spread"] < 4 * BF16_EPS
    assert lp["n"] == 24 and lp["records"] == 8
    assert by_field["vector"]["n"] == 3 * 8 * 4
    assert lp["machines"] == ["Apple M2 Ultra", "Apple M4 Max"] and lp["seeds"] == []
    assert lp["model"] == "tiny/llama@0000"
    assert lp["id"] == "gemma4:logits/read:logprob:bfloat16:apple-silicon"
    assert floor["key"] == list(KIND.key) and floor["matched_by"] == ["id"]
    assert [r["machine"] for r in floor["runs"]] == PARAMS["machines"]
    assert floor["not_numeric"] == {} and floor["compute"]
    schema = platform_kinds.item_schema(KIND)
    for item in floor["items"]:
        jsonschema.validate(item, schema)
    json.dumps(floor)


def test_a_field_that_differs_but_is_not_a_number_has_no_floor():
    b = json.loads(json.dumps(BOX))
    b["items"][0]["token"] = "a"
    floor = measure_noise([HOME, b], {**PARAMS, "machines": ["m1", "m2"], "seeds": [0, 1]})
    assert floor["not_numeric"] == {"token": 1}
    assert "token" not in {r["field"] for r in floor["items"]}
    assert floor["items"][0]["seeds"] == [0, 1]


def test_measuring_refuses_one_run_and_unnamed_machines():
    with pytest.raises(ValueError, match="at least two runs"):
        measure_noise([HOME], PARAMS)
    with pytest.raises(ValueError, match="name the machine"):
        measure_noise([HOME, BOX], PARAMS)


def test_bf16_noise_is_within_the_floor_and_a_finding_without_it():
    floor = measure_noise([HOME, BOX, THIRD], PARAMS)
    replica = reads(4, BF16_EPS / 2)
    plain = diff_collections(HOME, replica, {})["diff"]
    assert not plain["equivalent"] and plain["verdict"].endswith("records differ")
    judged = diff_collections(HOME, replica, {}, noise=floor)
    d = judged["diff"]
    assert d["findings"] == 0 and d["within"] and d["verdict"] == "within the floor"
    assert d["records"]["within"] >= 1 and not d["equivalent"]
    for row in judged["items"]:
        assert row["status"] == "within" and row["findings"] == 0
        for change in row["fields"].values():
            assert change["finding"] is False and change["floors"] <= 1
    exact = diff_collections(HOME, replica, {"tolerance": 0}, noise=floor)["diff"]
    assert exact["findings"] > 0 and exact["verdict"].endswith("findings above the tolerance")
    assert sum(f.get("findings", 0) for f in exact["changed_fields"].values()) > 0


def test_a_real_shift_is_counted_in_floors_above_k():
    floor = measure_noise([HOME, BOX, THIRD], PARAMS)
    shifted = reads(1, BF16_EPS, shift=0.5)
    out = diff_collections(HOME, shifted, {}, noise=floor)
    d = out["diff"]
    assert d["findings"] == 8 and d["verdict"] == "8 findings above the floor"
    change = out["items"][0]["fields"]["logprob"]
    assert change["finding"] and change["floors"] > 1
    assert d["floor"]["logprob"]["spread"] == pytest.approx(
        next(r for r in floor["items"] if r["field"] == "logprob")["spread"])
    loose = diff_collections(HOME, shifted, {"k": 1e6}, noise=floor)["diff"]
    assert loose["verdict"] == "within the floor"


def test_a_tolerance_judges_without_a_floor_per_field_or_globally():
    shifted = reads(1, 0.0, shift=0.25)
    base = reads(1, 0.0)
    assert diff_collections(base, shifted, {"tolerance": 0.3})["diff"]["verdict"] == \
        "within the tolerance"
    per = diff_collections(base, shifted, {"tolerance": {"fields": {"logprob": {"rel": 1e-4}}}})
    assert per["diff"]["verdict"] == "8 findings above the tolerance"
    assert diff_collections(base, shifted, {"tolerance": {"rel": 0.9}})["diff"]["within"]
    with pytest.raises(ValueError, match="abs, rel and fields"):
        diff_collections(base, shifted, {"tolerance": {"absolute": 1}})


def test_a_changed_text_and_a_field_without_a_floor_are_always_findings():
    floor = measure_noise([HOME, BOX, THIRD], {**PARAMS, "fields": ["logprob"]})
    b = json.loads(json.dumps(HOME))
    b["items"][0]["token"] = "a"
    b["items"][1]["vector"][0] += 1.0
    d = diff_collections(HOME, b, {}, noise=floor)
    rows = {r["id"]: r for r in d["items"]}
    assert rows["p0"]["fields"]["token"]["finding"]
    assert rows["p1"]["fields"]["vector"]["floors"] is None
    assert rows["p1"]["fields"]["vector"]["max_abs_delta"] == pytest.approx(1.0)
    assert d["diff"]["verdict"] == "2 findings above the floor"


def test_noise_for_picks_the_floor_records_and_refuses_an_empty_pick():
    floor = measure_noise([HOME, BOX, THIRD], PARAMS)
    assert diff_collections(HOME, reads(4, BF16_EPS / 2), {"noise_for": {"operation": "logits/read"}},
                            noise=floor)["diff"]["within"]
    with pytest.raises(ValueError, match="holds no records"):
        diff_collections(HOME, BOX, {"noise_for": {"operation": "text/generate"}}, noise=floor)


def test_without_a_floor_or_tolerance_the_diff_is_what_it_was():
    out = diff_collections(HOME, BOX, {})
    assert "findings" not in out["diff"] and "within" not in out["diff"]["records"]
    assert all("finding" not in c for r in out["items"] for c in r["fields"].values())
    assert diff_collections(HOME, HOME, {})["diff"]["verdict"] == "identical"


def test_the_ops_take_their_ports_as_the_executor_hands_them():
    from mechbench_compute.ops.records import diff, measure_noise as op

    floor = op.run(None, {"runs": [{"node": "home", "value": HOME}, {"node": "box", "value": BOX}]},
                   {**PARAMS, "machines": ["m1", "m2"]})
    assert floor["kind"] == "platform/noise" or floor.get("item_kind") == "platform/noise"
    out = diff.run(None, {"a": HOME, "b": reads(4, BF16_EPS / 4), "noise": floor}, {})
    assert "findings" in out["diff"]
