"""`bench._request` retries what carries no verdict, and nothing else.

A node result is PUT once. If the socket times out and the exception
propagates out of the node, the job is marked `failed`, its spool is
cleared, and hours of generation are unrecoverable — while the API is
healthy a second later.

Two halves, and the second matters as much as the first: a timeout or a
502 is silent about whether the request was acceptable, so retrying is
free (object PUTs are content-addressed, so a repeat writes identical
bytes or no-ops). A 400 is a verdict. Retrying it would turn a clear
error into a slow one, so the test asserts the call count is exactly one.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from mechbench_compute import bench


class FakeUrlopen:
    """Scripted `urlopen`: one scripted outcome consumed per call, the
    last repeating. An Exception is raised; anything else is returned as a
    context manager standing in for the response."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, req, timeout=None, context=None):
        self.calls += 1
        out = (self.outcomes.pop(0) if len(self.outcomes) > 1
               else self.outcomes[0])
        if isinstance(out, BaseException):
            raise out
        return out


class FakeResp:
    def __init__(self, body: bytes = b'{"ok":true}',
                 ctype: str = "application/json") -> None:
        self._body = body
        self.headers = {"content-type": ctype}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int, detail: str = "nope") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.test/objects/x", code, detail, {},
        io.BytesIO(detail.encode()))


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Backoff is real in production and must not be real here."""
    slept: list[float] = []
    import time
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def _patch(monkeypatch, fake):
    monkeypatch.setattr(bench.urllib.request, "urlopen", fake)


def test_timeout_then_success_does_not_raise(monkeypatch):
    """The observed failure: a write timeout, then a healthy API."""
    fake = FakeUrlopen(
        urllib.error.URLError("The write operation timed out"),
        urllib.error.URLError("The write operation timed out"),
        FakeResp(),
    )
    _patch(monkeypatch, fake)
    out = bench._request("PUT", "https://api.test/objects/x", "k", b"body")
    assert out == {"ok": True}
    assert fake.calls == 3


def test_4xx_is_never_retried(monkeypatch):
    """A rejected payload does not become acceptable on the second try."""
    fake = FakeUrlopen(http_error(400, "bad path"))
    _patch(monkeypatch, fake)
    with pytest.raises(bench.BenchError) as caught:
        bench._request("PUT", "https://api.test/objects/x", "k", b"body")
    assert fake.calls == 1
    assert not isinstance(caught.value, bench.BenchTransportError)
    assert "400" in str(caught.value)


def test_404_is_never_retried(monkeypatch):
    """Reads hit this path: a missing object is an answer, not a blip."""
    fake = FakeUrlopen(http_error(404, "no such object"))
    _patch(monkeypatch, fake)
    with pytest.raises(bench.BenchError):
        bench._request("GET", "https://api.test/objects/x", "k")
    assert fake.calls == 1


@pytest.mark.parametrize("code", [502, 503, 504, 429])
def test_5xx_and_429_retry_then_give_up_as_transport(monkeypatch, code):
    """Exhausted retries raise the TRANSPORT error, which is what tells
    the runner to interrupt the job rather than fail it."""
    fake = FakeUrlopen(http_error(code, "upstream"))
    _patch(monkeypatch, fake)
    with pytest.raises(bench.BenchTransportError):
        bench._request("PUT", "https://api.test/objects/x", "k", b"body",
                       attempts=3)
    assert fake.calls == 3


def test_exhausted_timeout_is_a_transport_error(monkeypatch):
    fake = FakeUrlopen(urllib.error.URLError("connection refused"))
    _patch(monkeypatch, fake)
    with pytest.raises(bench.BenchTransportError) as caught:
        bench._request("PUT", "https://api.test/objects/x", "k", b"body",
                       attempts=2)
    assert fake.calls == 2
    assert "unreachable" in str(caught.value)


def test_bare_timeout_error_retries(monkeypatch):
    """`socket.timeout` is `TimeoutError`, and on some paths it arrives
    un-wrapped rather than inside a URLError."""
    fake = FakeUrlopen(TimeoutError("timed out"), FakeResp())
    _patch(monkeypatch, fake)
    assert bench._request("GET", "https://api.test/objects/x", "k") == {"ok": True}
    assert fake.calls == 2


def test_backoff_grows_and_is_jittered(monkeypatch, _no_sleeping):
    fake = FakeUrlopen(urllib.error.URLError("down"), urllib.error.URLError("down"),
                       urllib.error.URLError("down"), FakeResp())
    _patch(monkeypatch, fake)
    bench._request("PUT", "https://api.test/objects/x", "k", b"body")
    assert len(_no_sleeping) == 3
    # Full jitter: each delay is within its own bound, and the bounds
    # double. Asserting the bound rather than the value keeps this from
    # being a flaky test about a random number.
    for i, delay in enumerate(_no_sleeping, start=1):
        assert 0.0 <= delay <= bench._RETRY_BASE_DELAY * (2 ** (i - 1))


def test_transport_error_is_a_bench_error():
    """Callers that already catch BenchError keep working unchanged."""
    assert issubclass(bench.BenchTransportError, bench.BenchError)
