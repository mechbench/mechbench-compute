"""Fork/join pays off only if the forks run at once (task 000396).

The graph has been DAG-general since 000248 and the executor ran it in
topological order, one node at a time — so two prompts to two providers,
then a judge, took the sum of the two calls. Two nodes whose inputs are
all computed do not depend on each other; the only thing waiting buys is
latency.

What is parallel is deliberately narrow: blocks whose work happens on
someone else's machine. A local model node must serialize (one model in
memory, one fused adapter at a time) and a pure block takes
microseconds, where a thread would be risk without a gain.
"""

from __future__ import annotations

import threading
import time

import pytest

from mechbench_compute.protocol import (
    ProtocolExecutor,
    ProtocolSpec,
    _is_remote,
)

ENDPOINT = {"provider": "mock", "model": "mock-large"}
LOCAL = "acme/tiny"
RECORDS = [{"id": "r1", "user": "hi"}, {"id": "r2", "user": "yo"}]


def _chat(nid, n_records=1, **params):
    """A chat node with ONE record by default: `text/chat` runs its own
    records concurrently, so a node with several would make the request
    counter say "two in flight" without two NODES being in flight."""
    return {"id": nid, "block": "text/chat",
            "params": {"model": ENDPOINT, "budget_usd": 5.0, "concurrency": 1,
                       **params},
            "inputs": {"records": RECORDS[:n_records]}}


def _spec(nodes, edges=()):
    return ProtocolSpec(kind="pipeline", prompt="", model_id=None,
                        extra={"graph": {"nodes": nodes, "edges": list(edges)}})


class TestWhatCountsAsRemote:
    def test_a_provider_backed_chat_is_remote(self):
        assert _is_remote("text/chat", {"model": ENDPOINT})
        assert _is_remote("eval/judge", {"judge": ENDPOINT})

    def test_local_weights_are_not(self):
        assert not _is_remote("text/chat", {"model": LOCAL})
        assert not _is_remote("text/generate", {"model": ENDPOINT})
        assert not _is_remote("records/select", {})


class TestBranchesRunTogether:
    def _timed(self, spec, monkeypatch, delay=0.25):
        """Every provider call sleeps; the wall clock then says whether
        the branches waited for each other."""
        from mechbench_compute.providers import mock as mock_mod

        real = mock_mod.MockTransport._chat
        in_flight, peak = [], [0]
        lock = threading.Lock()

        def slow(self, req, **kw):
            with lock:
                in_flight.append(1)
                peak[0] = max(peak[0], len(in_flight))
            time.sleep(delay)
            with lock:
                in_flight.pop()
            return real(self, req, **kw)

        monkeypatch.setattr(mock_mod.MockTransport, "_chat", slow)
        t0 = time.time()
        out = ProtocolExecutor().run(spec)
        return out, time.time() - t0, peak[0]

    def test_two_remote_branches_take_one_branch_s_time(self, monkeypatch):
        spec = _spec([_chat("left"), _chat("right")])
        out, elapsed, peak = self._timed(spec, monkeypatch)
        assert peak == 2, "the two branches did not overlap"
        # One call each: serial is two delays, together is about one.
        assert elapsed < 0.25 * 1.8, f"took {elapsed:.2f}s, no better than serial"
        assert set(out.payload["outputs"]) == {"left", "right"}

    def test_a_dependent_node_still_waits(self, monkeypatch):
        # left → right: not siblings, and no scheduler may pretend so.
        spec = _spec(
            [_chat("left"), {"id": "right", "block": "text/chat",
                             "params": {"model": ENDPOINT, "budget_usd": 5.0}}],
            [{"from": {"node": "left", "port": "documents"},
              "to": {"node": "right", "port": "records"}, "kind": "text/document"}])
        out, _elapsed, peak = self._timed(spec, monkeypatch, delay=0.05)
        assert peak == 1, "a node ran before its upstream finished"
        assert set(out.payload["outputs"]) == {"right"}

    def test_the_results_are_the_serial_results(self, monkeypatch):
        """Concurrency changes when work happens, never what it is."""
        spec = _spec([_chat("left"), _chat("right")])
        parallel = ProtocolExecutor().run(spec).payload

        monkeypatch.setattr("mechbench_compute.protocol.MAX_PARALLEL_NODES", 1)
        serial = ProtocolExecutor().run(_spec([_chat("left"), _chat("right")])).payload

        assert parallel["nodes_executed"] == serial["nodes_executed"]
        for nid in ("left", "right"):
            assert ([i["text"] for i in parallel["outputs"][nid]["items"]] ==
                    [i["text"] for i in serial["outputs"][nid]["items"]])

    def test_a_branch_that_fails_does_not_stop_its_sibling(self, monkeypatch):
        from mechbench_compute.providers import mock as mock_mod

        real = mock_mod.MockTransport._chat
        ran = []

        def picky(self, req, **kw):
            if "boom" in str(req.messages[0].content):
                raise RuntimeError("that provider said no")
            ran.append(1)
            return real(self, req, **kw)

        monkeypatch.setattr(mock_mod.MockTransport, "_chat", picky)
        spec = _spec([
            {"id": "bad", "block": "text/chat",
             "params": {"model": ENDPOINT, "budget_usd": 5.0},
             "inputs": {"records": [{"id": "b", "user": "boom"}]}},
            _chat("good"),
        ])
        with pytest.raises(RuntimeError, match="said no"):
            ProtocolExecutor().run(spec)
        assert ran, "the sibling branch never ran"


class TestSpoolingUnderConcurrency:
    def test_each_node_s_items_are_spooled_under_its_own_id(self, monkeypatch):
        """The failure this guards: an item spooled under the wrong node
        is a resumed job reusing another node's work."""
        spooled: list[tuple[str, str]] = []
        ex = ProtocolExecutor()
        ex._on_spool_item = lambda nid, key, item: spooled.append((nid, key))
        ex.run(_spec([_chat("left", 2), _chat("right", 2)]))
        for nid, key in spooled:
            assert key.split(":")[0] in ("r1", "r2")
            assert nid in ("left", "right")
        assert {nid for nid, _k in spooled} == {"left", "right"}
        assert len(spooled) == 4          # two records on each of two nodes
