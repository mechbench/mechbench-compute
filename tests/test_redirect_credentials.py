from __future__ import annotations

import http.server
import json
import threading
from typing import ClassVar

import pytest

from mechbench_compute import bench
from mechbench_compute.providers import http as ph


class _Seen(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, str]]] = []
    target = ""

    def do_GET(self):
        if self.path.startswith("/hop"):
            self.send_response(302)
            self.send_header("Location", self.target + "/landed")
            self.end_headers()
            return
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def hosts():
    servers = []
    for _ in range(2):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Seen)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    _Seen.seen = []
    _Seen.target = f"http://localhost:{servers[1].server_address[1]}"
    yield f"http://127.0.0.1:{servers[0].server_address[1]}"
    for server in servers:
        server.shutdown()


class TestARedirectCarriesNoCredentials:
    def test_the_bench_key_stays_with_the_bench(self, hosts):
        bench._request("GET", f"{hosts}/hop", "mb_secret", attempts=1)
        assert _Seen.seen and "authorization" not in _Seen.seen[0]

    def test_a_provider_key_stays_with_the_provider(self, hosts):
        got = ph.get_json(f"{hosts}/hop", headers={"x-api-key": "sk-secret",
                                                  "authorization": "Bearer sk-secret"})
        assert got.body == {"ok": True}
        assert not {"x-api-key", "authorization"} & set(_Seen.seen[0])

    def test_without_a_redirect_the_key_is_sent(self, hosts):
        ph.get_json(f"{_Seen.target}/direct", headers={"x-api-key": "sk-secret"})
        assert _Seen.seen[0]["x-api-key"] == "sk-secret"
