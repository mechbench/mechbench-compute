from __future__ import annotations

import io
import urllib.error

import pytest

from mechbench_compute import bench


class FakeUrlopen:
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
    slept: list[float] = []
    import time
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def _patch(monkeypatch, fake):
    monkeypatch.setattr(bench.urllib.request, "urlopen", fake)


def test_timeout_then_success_does_not_raise(monkeypatch):
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
    fake = FakeUrlopen(http_error(400, "bad path"))
    _patch(monkeypatch, fake)
    with pytest.raises(bench.BenchError) as caught:
        bench._request("PUT", "https://api.test/objects/x", "k", b"body")
    assert fake.calls == 1
    assert not isinstance(caught.value, bench.BenchTransportError)
    assert "400" in str(caught.value)


def test_404_is_never_retried(monkeypatch):
    fake = FakeUrlopen(http_error(404, "no such object"))
    _patch(monkeypatch, fake)
    with pytest.raises(bench.BenchError):
        bench._request("GET", "https://api.test/objects/x", "k")
    assert fake.calls == 1


@pytest.mark.parametrize("code", [502, 503, 504, 429])
def test_5xx_and_429_retry_then_give_up_as_transport(monkeypatch, code):
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
    for i, delay in enumerate(_no_sleeping, start=1):
        assert 0.0 <= delay <= bench._RETRY_BASE_DELAY * (2 ** (i - 1))


def test_transport_error_is_a_bench_error():
    assert issubclass(bench.BenchTransportError, bench.BenchError)
