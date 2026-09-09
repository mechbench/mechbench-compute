"""JSON over HTTPS, on the standard library (task 000337).

Deliberately small and dependency-free: `urllib.request` with certifi's
CA bundle, a timeout, and one place that turns a status code into the
error the retry loop understands. The whole surface a provider adapter
needs is `post_json` / `get_json`, so swapping in a pooled client later
is a one-file change.

Secrets go in headers and NOWHERE else: never in a URL, never in a log
line, never in an exception message. `_redact` runs over every error
body before it becomes an exception, because provider errors quote the
request back at you often enough to matter.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mechbench_compute.providers.errors import (
    AuthError,
    ProviderError,
    RateLimited,
    TransientError,
)

DEFAULT_TIMEOUT = 600.0

_ctx: ssl.SSLContext | None = None


def _ssl_context() -> ssl.SSLContext:
    global _ctx
    if _ctx is None:
        import certifi

        _ctx = ssl.create_default_context(cafile=certifi.where())
    return _ctx


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: Any


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "«redacted»")
    return text


def request_json(method: str, url: str, *, headers: Mapping[str, str],
                 payload: Any = None, timeout: float = DEFAULT_TIMEOUT,
                 secrets: tuple[str, ...] = ()) -> HttpResponse:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"content-type": "application/json",
                                          **dict(headers)})
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=_ssl_context()) as resp:
            raw = resp.read()
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return HttpResponse(status=resp.status, headers=hdrs,
                                body=json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        body = _redact(e.read().decode("utf-8", "replace")[:2000], secrets)
        hdrs = {k.lower(): v for k, v in (e.headers or {}).items()}
        raise _status_error(e.code, body, hdrs) from None
    except urllib.error.URLError as e:
        raise TransientError(f"{method} {_host(url)}: {e.reason}") from None
    except TimeoutError:
        raise TransientError(f"{method} {_host(url)}: timed out after {timeout:g}s") from None


def post_json(url: str, **kw: Any) -> HttpResponse:
    return request_json("POST", url, **kw)


def get_json(url: str, **kw: Any) -> HttpResponse:
    return request_json("GET", url, **kw)


def _host(url: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _status_error(status: int, body: str, headers: Mapping[str, str]) -> ProviderError:
    if status in (401, 403):
        return AuthError(f"HTTP {status}: {body}")
    if status == 429:
        from mechbench_compute.providers.limiter import RateLimits

        return RateLimited(f"HTTP 429: {body}",
                           retry_after=RateLimits.from_headers(headers).retry_after)
    if status in (408, 409, 425, 500, 502, 503, 504, 529):
        return TransientError(f"HTTP {status}: {body}", status=status)
    return ProviderError(f"HTTP {status}: {body}")
