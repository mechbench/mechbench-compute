"""Progress accounting for pipelines (task 000316).

Two invariants: the flat scalar never overruns its denominator (the
old unconditional per-node bump made an expanded node worth n+1 of n),
and a three-argument callback receives the node structure — index and
count that never change mid-run, per-node done/total that reset at
each node — while a two-argument callback keeps working untouched.
"""

from __future__ import annotations

from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec


def _spec():
    # read1 -> read2, both "model" blocks (faked below): the shape that
    # showed the overrun in production.
    graph = {
        "nodes": [
            {"id": "read1", "block": "~canonical/ops/decision-read/1",
             "params": {}},
            {"id": "read2", "block": "~canonical/ops/decision-read/1",
             "params": {}},
        ],
        "edges": [
            {"from": {"node": "read1", "port": "out"},
             "to": {"node": "read2", "port": "records"}, "kind": "records"},
        ],
    }
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": graph})


def _fake_model_block(monkeypatch, n_items):
    def fake(self, fn, inputs, params, *args, on_item=None, on_start=None):
        if on_start:
            on_start(n_items)
        for _ in range(n_items):
            if on_item:
                on_item()
        return {"kind": "decision_read", "conditions": []}

    monkeypatch.setattr(ProtocolExecutor, "_run_model_block", fake)


class TestScalar:
    def test_done_never_overruns_total(self, monkeypatch):
        _fake_model_block(monkeypatch, n_items=3)
        ticks = []
        ProtocolExecutor().run(_spec(), on_progress=lambda d, t: ticks.append((d, t)))
        assert all(d <= t for d, t in ticks)
        assert ticks[-1] == (6, 6)  # two nodes x three items, exactly

    def test_totals_only_grow(self, monkeypatch):
        _fake_model_block(monkeypatch, n_items=3)
        ticks = []
        ProtocolExecutor().run(_spec(), on_progress=lambda d, t: ticks.append((d, t)))
        assert [t for _, t in ticks] == sorted(t for _, t in ticks)


class TestNodeStructure:
    def test_three_arg_callback_gets_the_node_view(self, monkeypatch):
        _fake_model_block(monkeypatch, n_items=3)
        views = []
        ProtocolExecutor().run(
            _spec(),
            on_progress=lambda d, t, node: views.append((d, t, node)),
        )
        # count never changes; index only steps forward
        assert {v[2]["count"] for v in views} == {2}
        indexes = [v[2]["index"] for v in views]
        assert indexes == sorted(indexes) and indexes[-1] == 2
        # inside node 2 the per-node counters run 0..3 of 3
        in_second = [v[2] for v in views if v[2]["index"] == 2]
        assert in_second[0]["done"] == 0
        assert in_second[-1] == {"index": 2, "count": 2, "id": "read2",
                                 "done": 3, "total": 3}

    def test_an_unexpanded_node_reports_no_item_counts(self, monkeypatch):
        def fake(self, fn, inputs, params, *args, on_item=None, on_start=None):
            return {"kind": "decision_read", "conditions": []}

        monkeypatch.setattr(ProtocolExecutor, "_run_model_block", fake)
        views = []
        ProtocolExecutor().run(
            _spec(), on_progress=lambda d, t, node: views.append(node))
        assert all(v["total"] == 0 for v in views)
        assert views[-1]["index"] == 2

    def test_two_arg_callback_is_untouched(self, monkeypatch):
        _fake_model_block(monkeypatch, n_items=2)
        ticks = []
        ProtocolExecutor().run(_spec(), on_progress=lambda d, t: ticks.append((d, t)))
        assert ticks[-1] == (4, 4)
