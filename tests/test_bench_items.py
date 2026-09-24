from __future__ import annotations

import urllib.parse

import pytest

from mechbench_compute import bench


@pytest.fixture
def sent(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake(method, url, key, **_k):
        calls.append((method, url))
        return {"items": []}

    monkeypatch.setattr(bench, "_request", fake)
    monkeypatch.setattr(bench, "_config", lambda u, k: ("http://api.test", "mbk"))
    return calls


def query(url: str) -> list[tuple[str, str]]:
    return urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)


def test_fields_where_and_cuts_become_the_query(sent):
    bench.items("benji/lab/r/gen", fields=["id", "text"],
                where=["coords.prompt=flash", "metadata.tokens>100"],
                lines=1, limit=200)
    method, url = sent[-1]
    assert method == "GET" and url.startswith("http://api.test/objects/~items?")
    assert query(url) == [
        ("path", "benji/lab/r/gen"), ("fields", "id,text"), ("limit", "200"),
        ("lines", "1"), ("where", "coords.prompt=flash"),
        ("where", "metadata.tokens>100"),
    ]


def test_a_mapping_is_equalities_and_count_and_header_are_flags(sent):
    bench.items("p", where={"coords.prompt": "flash", "coords.sample": 3},
                count=True)
    assert query(sent[-1][1]) == [
        ("path", "p"), ("where", "coords.prompt=flash"),
        ("where", "coords.sample=3"), ("count", "1"),
    ]
    bench.items("p", header=True)
    assert query(sent[-1][1]) == [("path", "p"), ("header", "1")]


def test_fetch_items_is_still_a_whole_page(sent):
    bench.fetch_items("p", offset=2, limit=3)
    assert query(sent[-1][1]) == [("path", "p"), ("offset", "2"), ("limit", "3")]
