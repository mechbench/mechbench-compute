"""Fetch-on-first-use guests (task 000359). No network, no real
guest: the contract is about hashes, and hashes can be tested with
any bytes."""
from __future__ import annotations

import hashlib
import pathlib

import pytest

from mechbench_compute import guests


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MECHBENCH_GUEST_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(guests, "REGISTRY", {})
    return tmp_path


def _artifact(tmp_path, data=b"\0asm\1\0\0\0fake") -> pathlib.Path:
    p = tmp_path / "guest.wasm"
    p.write_bytes(data)
    return p


class TestInstallLocal:
    def test_it_hashes_rather_than_trusting(self, tmp_path):
        p = _artifact(tmp_path)
        g = guests.install_local("fake", p, source="here")
        assert g.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
        assert g.size == p.stat().st_size and g.source == "here"

    def test_ensure_then_finds_it_without_fetching(self, tmp_path):
        guests.install_local("fake", _artifact(tmp_path))
        path = guests.ensure("fake", fetch=False)
        assert path.is_file() and path.name.startswith("fake-")


class TestTheHashIsTheContract:
    def test_a_corrupt_cached_file_is_discarded_not_run(self, tmp_path):
        g = guests.install_local("fake", _artifact(tmp_path))
        cached = guests.ensure("fake", fetch=False)
        cached.write_bytes(b"tampered")
        with pytest.raises(guests.GuestUnavailable, match="not cached"):
            guests.ensure("fake", fetch=False)
        assert not cached.exists(), "the bad file was left for the next caller"

    def test_a_fetch_that_hashes_wrong_is_refused_and_not_kept(self, tmp_path, monkeypatch):
        served = tmp_path / "served.wasm"
        served.write_bytes(b"what the host actually serves")
        guests.register(guests.Guest(name="g", url=served.as_uri(),
                                     sha256="0" * 64, size=len(served.read_bytes())))
        with pytest.raises(guests.GuestUnavailable, match="hashed to"):
            guests.ensure("g")
        assert not list(guests.cache_dir().glob("g-*")), "a mismatched fetch was kept"

    def test_a_good_fetch_lands_under_its_hash(self, tmp_path):
        served = tmp_path / "served.wasm"
        data = b"\0asm real enough"
        served.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        guests.register(guests.Guest(name="g", url=served.as_uri(),
                                     sha256=digest, size=len(data)))
        path = guests.ensure("g")
        assert path.read_bytes() == data and digest[:12] in path.name

    def test_size_is_checked_too(self, tmp_path):
        served = tmp_path / "served.wasm"
        data = b"\0asm sized"
        served.write_bytes(data)
        guests.register(guests.Guest(name="g", url=served.as_uri(),
                                     sha256=hashlib.sha256(data).hexdigest(),
                                     size=len(data) + 1))
        with pytest.raises(guests.GuestUnavailable, match="registry error"):
            guests.ensure("g")


class TestThePin:
    def test_a_build_that_differs_from_the_pin_is_refused(self, tmp_path):
        guests.register(guests.Guest("g", "", "0" * 64, 1, source="somewhere@abc"))
        with pytest.raises(guests.GuestUnavailable, match="pinned at 0000"):
            guests.install_local("g", _artifact(tmp_path))
        assert guests.REGISTRY["g"].sha256 == "0" * 64, "the pin was overwritten"

    def test_replace_re_pins_deliberately(self, tmp_path):
        guests.register(guests.Guest("g", "", "0" * 64, 1, source="somewhere@abc"))
        g = guests.install_local("g", _artifact(tmp_path), replace=True)
        assert g.sha256 != "0" * 64 and g.source == "somewhere@abc"

    def test_a_pinned_unhosted_guest_says_how_to_get_it(self):
        guests.register(guests.Guest("g", "", "0" * 64, 1, source="repo@sha"))
        with pytest.raises(guests.GuestUnavailable, match="not hosted yet.*repo@sha.*install_local"):
            guests.ensure("g")

    def test_busybox_is_pinned_in_the_real_registry(self, monkeypatch):
        monkeypatch.undo()  # the autouse fixture emptied it; look at the real one
        from mechbench_compute import guests as real
        assert "busybox" in real.REGISTRY
        assert not real.REGISTRY["busybox"].hosted, "hosting is blocked on the LICENSE question"


class TestRefusals:
    def test_an_unknown_guest_names_the_known_ones(self):
        guests.register(guests.Guest("busybox", "x", "y", 1))
        with pytest.raises(guests.GuestUnavailable, match="known: busybox"):
            guests.ensure("cpython")

    def test_no_fetch_means_no_fetch(self, tmp_path):
        guests.register(guests.Guest("g", (tmp_path / "nope").as_uri(), "0" * 64, 1))
        with pytest.raises(guests.GuestUnavailable, match="fetching is disabled"):
            guests.ensure("g", fetch=False)
