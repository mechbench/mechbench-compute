"""The bench client library — launch / watch / results_for / result, the
unwrapping fetch, and credential discovery (task 000450).

Every HTTP call funnels through `bench._request`; these scripts that one
chokepoint and assert the shape of what goes out and what comes back. The
point is that an experiment (and the three CLI verbs over this) get the
launch/watch/read plumbing from here, not re-derived over httpx, and that a
payload is never unwrapped by hand.
"""

from __future__ import annotations

import mechbench_schema as ms
import pytest

from mechbench_compute import bench


class Seq(list):
    """A scripted SEQUENCE of responses — one consumed per call, the last
    repeating. Distinct from a plain list, which is returned whole (a
    listing response is itself a list)."""


class FakeReq:
    """A stand-in for `bench._request`: scripted responses by (method,
    url-substring), every call recorded. A scripted value that is an
    Exception is raised; a `Seq` is consumed one entry per call; anything
    else is returned whole."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.script: dict[tuple[str, str], object] = {}

    def add(self, method: str, sub: str, value: object) -> FakeReq:
        self.script[(method, sub)] = value
        return self

    def __call__(self, method, url, key, body=None, headers=None,
                 return_headers=False, timeout=60):
        self.calls.append({"method": method, "url": url, "body": body,
                           "headers": headers, "timeout": timeout})
        for (m, sub), val in self.script.items():
            if m == method and sub in url:
                if isinstance(val, Seq):
                    val = val.pop(0) if len(val) > 1 else val[0]
                if isinstance(val, BaseException):
                    raise val
                return val
        raise AssertionError(f"unscripted {method} {url}")


@pytest.fixture
def fake(monkeypatch):
    """A scripted transport, with credentials already resolved so no test
    touches the real environment or `~/.mechbench`."""
    fr = FakeReq()
    monkeypatch.setattr(bench, "_request", fr)
    monkeypatch.setattr(bench, "_config", lambda u, k: ("https://api.test", "K"))
    return fr


def _envelope(payload):
    return ms.dump_canonical({"payload": payload,
                              "provenance": {"created_at": "now"}})


class TestFetchUnwraps:
    def test_fetch_returns_the_payload(self, fake):
        fake.add("GET", "/objects/", _envelope({"kind": "ladder", "n": 3}))
        assert bench.fetch("o/p/x") == {"kind": "ladder", "n": 3}

    def test_fetch_envelope_keeps_the_provenance(self, fake):
        fake.add("GET", "/objects/", _envelope({"kind": "ladder"}))
        env = bench.fetch_envelope("o/p/x")
        assert env["payload"] == {"kind": "ladder"} and "provenance" in env

    def test_a_typed_record_is_not_unwrapped(self, fake):
        # top-level provenance but no `payload` — a typed record, left alone
        rec = {"kind": "decision_expansion", "provenance": {"created_at": "t"}}
        fake.add("GET", "/objects/", ms.dump_canonical(rec))
        assert bench.fetch("o/p/x") == rec

    def test_with_meta_still_carries_the_hash(self, monkeypatch):
        monkeypatch.setattr(bench, "_config", lambda u, k: ("https://api.test", "K"))

        def req(method, url, key, return_headers=False, **kw):
            assert return_headers is True
            return _envelope({"n": 1}), {"x-content-hash": "sha256:abc"}
        monkeypatch.setattr(bench, "_request", req)
        payload, meta = bench.fetch("o/p/x", with_meta=True)
        assert payload == {"n": 1} and meta["content_hash"] == "sha256:abc"


class TestLaunch:
    def test_it_posts_bindings_and_budget_and_returns_the_bare_run(self, fake):
        fake.add("POST", "/protocols/owner~p~proto/runs",
                 {"id": "r1", "jobId": "j1"})
        out = bench.launch("owner~p~proto", {"model": "gemma"}, budget=2.5)
        assert out == {"id": "r1", "jobId": "j1"}
        call = fake.calls[-1]
        assert call["method"] == "POST" and call["url"].endswith("/runs")
        import json
        assert json.loads(call["body"]) == {"bindings": {"model": "gemma"},
                                            "budgetUsd": 2.5}
        assert call["timeout"] == 90  # binding+queue can be slow

    def test_no_budget_sends_no_cap(self, fake):
        fake.add("POST", "/runs", {"id": "r", "jobId": "j"})
        bench.launch("p", {"a": "b"})
        import json
        assert "budgetUsd" not in json.loads(fake.calls[-1]["body"])


class TestCreateProtocol:
    def test_it_posts_the_graph_and_returns_the_bare_protocol(self, fake):
        # The protocols routes still wrap (`{protocol: …}`, task 000456);
        # the library unwraps once so no author does.
        fake.add("POST", "/protocols", {"protocol": {"id": "prt_1", "version": 1,
                                                     "name": "018-axes"}})
        out = bench.create_protocol("benji", "lab", "018-axes",
                                    graph={"nodes": [], "edges": []},
                                    description="d",
                                    signature={"inputs": [], "outputs": []})
        assert out == {"id": "prt_1", "version": 1, "name": "018-axes"}
        import json
        body = json.loads(fake.calls[-1]["body"])
        assert body["ownerHandle"] == "benji" and body["projectSlug"] == "lab"
        assert body["graph"] == {"nodes": [], "edges": []}
        assert body["signature"] == {"inputs": [], "outputs": []}
        assert fake.calls[-1]["url"].endswith("/protocols")

    def test_a_bare_reply_passes_through(self, fake):
        fake.add("POST", "/protocols", {"id": "prt_2", "version": 1})
        assert bench.create_protocol("o", "p", "n", graph={})["id"] == "prt_2"


class TestWatch:
    def test_it_yields_only_on_change_until_terminal(self, fake):
        fake.add("GET", "/jobs/j", Seq([
            {"status": "running", "progressNum": 1, "progressDen": 2},
            {"status": "running", "progressNum": 1, "progressDen": 2},  # same
            {"status": "done", "progressNum": 2, "progressDen": 2},
        ]))
        seen = list(bench.watch(["j"], interval=0))
        # three polls, two distinct states -> two yields
        assert [j["status"] for _, j in seen] == ["running", "done"]
        assert all(jid == "j" for jid, _ in seen)

    def test_a_transient_error_is_yielded_and_the_poll_continues(self, fake):
        fake.add("GET", "/jobs/j", Seq([
            bench.BenchError("GET .../jobs/j -> 502: bad gateway"),
            {"status": "done"},
        ]))
        seen = list(bench.watch(["j"], interval=0))
        assert seen[0][1]["status"] is None and "502" in seen[0][1]["error"]
        assert seen[-1][1]["status"] == "done"

    def test_two_jobs_both_run_to_terminal(self, fake):
        fake.add("GET", "/jobs/a", {"status": "done"})
        fake.add("GET", "/jobs/b", {"status": "failed", "errorMessage": "boom"})
        finals = dict(bench.watch(["a", "b"], interval=0))
        assert finals["a"]["status"] == "done"
        assert finals["b"]["status"] == "failed"


class TestResultsFor:
    def test_string_and_object_bindings_both_filter(self, fake):
        fake.add("GET", "/protocols/024/runs",
                 [{"id": "r", "jobId": "j", "resultPath": "o/p/results/j"}])
        out = bench.results_for("024", corpus="benji/c/animals",
                                ref={"provider": "x", "model": "y"})
        assert out and out[0]["jobId"] == "j"
        url = fake.calls[-1]["url"]
        assert "binding.corpus=benji" in url  # url-encoded value
        # a structured binding travels as canonical JSON the server parses
        assert "binding.ref=" in url and "provider" in url

    def test_no_bindings_lists_the_runs(self, fake):
        fake.add("GET", "/protocols/p/runs", [{"id": "r"}])
        assert bench.results_for("p") == [{"id": "r"}]
        assert "?" not in fake.calls[-1]["url"]


class TestResult:
    def test_it_reads_a_node_through_the_job_and_unwraps(self, fake):
        fake.add("GET", "/jobs/j", {"resultPath": "o/p/results/j",
                                    "status": "done"})
        fake.add("GET", "/objects/o/p/results/j/grade",
                 _envelope({"kind": "metric_table", "rows": [{"n": 3}]}))
        out = bench.result("j", "grade")
        assert out == {"kind": "metric_table", "rows": [{"n": 3}]}

    def test_a_row_with_a_result_path_skips_the_job_fetch(self, fake):
        fake.add("GET", "/objects/o/p/results/j/n", _envelope({"v": 1}))
        out = bench.result({"resultPath": "o/p/results/j"}, "n")
        assert out == {"v": 1}
        assert not any("/jobs/" in c["url"] for c in fake.calls)

    def test_no_result_yet_is_an_error(self, fake):
        fake.add("GET", "/jobs/j", {"status": "running", "resultPath": None})
        with pytest.raises(bench.BenchError, match="no result yet"):
            bench.result("j", "grade")


class TestCredentialDiscovery:
    """`_config` finds credentials the way the CLI does: argument, then
    `configure`, then the environment, then `~/.mechbench/config.toml`."""

    @pytest.fixture(autouse=True)
    def clean(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MECHBENCH_API_KEY", raising=False)
        monkeypatch.delenv("MECHBENCH_API_URL", raising=False)
        bench._DEFAULTS.clear()
        cfg = tmp_path / "config.toml"
        monkeypatch.setattr(bench, "_config_file", lambda: cfg)
        return cfg

    def test_stored_login_is_used_when_nothing_else_is_set(self, clean):
        clean.write_text('[runner]\napi_url = "https://api.stored"\n'
                         'api_key = "mbk_stored"\n')
        assert bench._config(None, None) == ("https://api.stored", "mbk_stored")

    def test_env_key_owns_the_pair_and_ignores_the_file(self, clean, monkeypatch):
        clean.write_text('[runner]\napi_url = "https://api.stored"\n'
                         'api_key = "mbk_stored"\n')
        monkeypatch.setenv("MECHBENCH_API_KEY", "mbk_env")
        monkeypatch.setenv("MECHBENCH_API_URL", "https://api.env")
        assert bench._config(None, None) == ("https://api.env", "mbk_env")

    def test_an_explicit_argument_wins(self, clean):
        clean.write_text('[runner]\napi_url = "https://api.stored"\n'
                         'api_key = "mbk_stored"\n')
        assert bench._config("https://api.arg", "mbk_arg") == (
            "https://api.arg", "mbk_arg")

    def test_a_missing_login_is_a_clear_error(self, clean):
        with pytest.raises(bench.BenchError, match="mechbench login"):
            bench._config(None, None)
