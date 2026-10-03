from __future__ import annotations

import pytest

from mechbench_compute.expr import engine as engine_mod
from mechbench_compute.expr.engine import FUEL, ExprError, FuelRefused, load_engine
from mechbench_compute.lexicon import kinds as K
from mechbench_compute.ops.records.derive import derive
from mechbench_compute.ops.records.filter import filter_records
from mechbench_compute.ops.records.group import group_records
from mechbench_compute.ops.records.join import join_records
from mechbench_compute.ops.records.sort import sort_records
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec

WIDTH = 2560

NORM = "sqrt(sum([v * v for v in vector]))"

OPS = {
    "records/derive": lambda r: derive(r, {"fields": {"norm": NORM}}, {}),
    "records/derive templates": lambda r: derive(r, {"templates": {"norm": "{" + NORM + "}"}}, {}),
    "records/filter": lambda r: filter_records(r, {"where": f"{NORM} > 0"}, {}),
    "records/sort": lambda r: sort_records(r, {"by": [NORM]}, {}),
    "records/group": lambda r: group_records(r, {"aggregates": {"norm": f"mean({NORM})"}}, {}),
    "records/join": lambda r: join_records(r, r, {"on": NORM}, {}),
}


def build_items(n: int) -> list[dict]:
    return [{"id": f"r{i}", "vector": [0.5] * WIDTH} for i in range(n)]


def build_vectors(n: int) -> dict:
    return K.collection("records/record", build_items(n))


@pytest.fixture
def sent(monkeypatch) -> list[int]:
    engine = load_engine()
    real = engine.call
    rows: list[int] = []

    def call(request):
        if request.get("records") is not None:
            rows.append(len(request["records"]))
        return real(request)

    monkeypatch.setattr(engine, "call", call)
    return rows


def test_an_expression_that_would_run_out_of_fuel_is_refused_on_its_first_record(sent):
    with pytest.raises(FuelRefused) as e:
        derive(build_vectors(300), {"fields": {"norm": NORM}}, {})
    refused = e.value
    assert sent == [1]
    assert (refused.code, refused.expr, refused.rows, refused.limit) == ("EXPRESSION_FUEL", NORM, 300, FUEL)
    assert refused.estimate >= 2 * FUEL
    assert str(refused).startswith(f"EXPRESSION_FUEL: `{NORM}` spends more than ")
    assert f"{refused.estimate:,}" in str(refused) and f"{FUEL:,}" in str(refused)
    assert "`top: k` on `activations/capture`" in refused.remedy and str(refused).endswith(refused.remedy)
    assert refused.issue == {"code": "EXPRESSION_FUEL", "expr": NORM, "rows": 300,
                             "estimate": refused.estimate, "limit": FUEL, "remedy": refused.remedy,
                             "message": str(refused).removeprefix("EXPRESSION_FUEL: ")}


def test_a_cheap_expression_over_the_same_records_runs(sent):
    out = derive(build_vectors(300), {"fields": {"total": "sum(vector)", "width": "len(vector)"}}, {})
    assert [(it["total"], it["width"]) for it in out["items"]] == [(0.5 * WIDTH, WIDTH)] * 300
    assert sent == [1, 300]


def test_two_records_run_without_a_first_pass(sent):
    out = derive(build_vectors(2), {"fields": {"norm": NORM}}, {})
    assert [it["norm"] for it in out["items"]] == [pytest.approx((0.25 * WIDTH) ** 0.5)] * 2
    assert sent == [2]


def test_under_twice_the_limit_the_collection_runs_until_the_engine_stops_it(sent):
    with pytest.raises(ValueError, match=r"^records/derive: the evaluation ran out of fuel in record 'r\d+'") as e:
        derive(build_vectors(200), {"fields": {"norm": NORM}}, {})
    assert not isinstance(e.value, FuelRefused)
    assert sent == [1, 200]


@pytest.mark.parametrize("op", sorted(OPS))
def test_every_expression_op_refuses_before_the_collection_runs(op, sent):
    with pytest.raises(FuelRefused, match="^EXPRESSION_FUEL: "):
        OPS[op](build_vectors(300))
    assert sent == [1]


def test_the_refusal_reaches_the_run_as_it_was_raised():
    graph = {"dataflow": 2, "edges": [], "nodes": [
        {"id": "norms", "block": "records/derive", "params": {"fields": {"norm": NORM}},
         "inputs": {"records": build_items(300)}}]}
    with pytest.raises(FuelRefused) as e:
        ProtocolExecutor().run(ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                                            extra={"graph": graph, "params": {}, "inputs": {}}))
    assert e.value.issue["code"] == "EXPRESSION_FUEL" and e.value.rows == 300


def test_each_call_runs_on_the_fuel_compute_names(monkeypatch):
    monkeypatch.setattr(engine_mod, "FUEL", 50)
    engine = load_engine()
    assert engine.evaluate("1", [{}] * 50).values == [1] * 50
    with pytest.raises(ExprError, match="ran out of fuel"):
        engine.evaluate("1", [{}] * 51)
