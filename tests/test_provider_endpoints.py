from __future__ import annotations

import pytest

from mechbench_compute import lexicon
from mechbench_compute.block_params import check_params
from mechbench_compute.protocol.check_graph import check_graph
from mechbench_compute.providers import endpoint, make_transport
from mechbench_compute.providers.errors import EndpointRefused

PUBLIC = "104.18.2.3"


@pytest.fixture
def resolves_to(monkeypatch):
    monkeypatch.delenv(endpoint.ALLOW_PRIVATE, raising=False)

    def set_to(*addresses):
        monkeypatch.setattr(endpoint, "resolve_addresses",
                            lambda _host, _port: list(addresses))

    return set_to


class TestTheProtocolCannotNameAnEndpoint:
    def test_base_url_is_not_a_param_of_text_chat(self):
        op = lexicon.BY_NAME["text/chat"]
        assert "base_url" not in op.param_names
        assert "base_url" not in {p["name"] for p in op.to_dict()["params"]}

    def test_no_op_declares_an_endpoint_param(self):
        names = {p.name for op in lexicon.BY_NAME.values() for p in op.params}
        assert not {n for n in names if n in ("base_url", "endpoint", "url", "host")}

    def test_a_stored_protocol_carrying_base_url_is_refused_by_name(self):
        with pytest.raises(ValueError) as e:
            check_params("text/chat", {"base_url": "https://collector.example"})
        assert str(e.value) == "PARAM_REMOVED: base_url — set the endpoint on the credential"

    def test_the_graph_is_refused_before_anything_runs(self):
        nodes = {"ask": {"block": "text/chat",
                         "params": {"model": {"provider": "anthropic", "model": "claude-opus-5"},
                                    "base_url": "http://collector.example"},
                         "inputs": {"records": [{"id": "r0", "user": "hi"}]}}}
        with pytest.raises(ValueError, match="PARAM_REMOVED: base_url"):
            check_graph(nodes, [], ["ask"])


class TestTheCredentialsEndpoint:
    @pytest.mark.parametrize("provider", ["anthropic", "gemini", "openai", "openai-compatible"])
    def test_http_is_refused(self, resolves_to, provider):
        resolves_to(PUBLIC)
        with pytest.raises(EndpointRefused, match="must be https://"):
            make_transport(provider, {"token": "k", "base_url": "http://api.example.com/v1"})

    @pytest.mark.parametrize("address", [
        "127.0.0.1", "::1", "169.254.169.254", "10.0.0.5", "172.16.3.4",
        "192.168.1.10", "100.64.0.1", "fd00::1", "fe80::1", "::ffff:10.0.0.1", "0.0.0.0",
    ])
    def test_a_private_address_is_refused(self, resolves_to, address):
        resolves_to(address)
        with pytest.raises(EndpointRefused) as e:
            make_transport("anthropic", {"token": "k", "base_url": "https://proxy.example.com"})
        assert "not a public address" in str(e.value)
        assert endpoint.ALLOW_PRIVATE in str(e.value)

    def test_one_private_address_among_public_ones_is_refused(self, resolves_to):
        resolves_to(PUBLIC, "10.1.2.3")
        with pytest.raises(EndpointRefused, match="10.1.2.3"):
            make_transport("openai", {"token": "k", "base_url": "https://proxy.example.com"})

    def test_other_schemes_are_refused_even_when_opted_in(self, resolves_to, monkeypatch):
        monkeypatch.setenv(endpoint.ALLOW_PRIVATE, "1")
        with pytest.raises(EndpointRefused, match="not an https:// URL"):
            make_transport("openai-compatible", {"base_url": "file:///etc/passwd"})

    def test_a_public_https_endpoint_is_accepted(self, resolves_to):
        resolves_to(PUBLIC)
        t = make_transport("openai-compatible",
                           {"token": "k", "base_url": "https://openrouter.ai/api/v1"})
        assert t._base == "https://openrouter.ai/api/v1"

    def test_the_runner_owner_opts_in_to_a_local_server(self, resolves_to, monkeypatch):
        monkeypatch.setenv(endpoint.ALLOW_PRIVATE, "1")
        t = make_transport("openai-compatible", {"base_url": "http://127.0.0.1:8080/v1"})
        assert t._base == "http://127.0.0.1:8080/v1"

    def test_the_default_endpoint_needs_no_lookup(self, monkeypatch):
        def no_lookup(_host, _port):
            raise AssertionError("looked up")

        monkeypatch.setattr(endpoint, "resolve_addresses", no_lookup)
        t = make_transport("anthropic", {"token": "k"})
        assert t._base.startswith("https://api.anthropic.com")
