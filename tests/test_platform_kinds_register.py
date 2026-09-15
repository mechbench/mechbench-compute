"""register_all follows the declarations, including when they change.

Task 000508: the registry refused a changed manifest at a registered
version (`MANIFEST_PINNED`), and `register_all` printed "pinned" and
moved on — so after phase 2b of the typology the published contract
described the fields eight kinds no longer had. A refusal that names
the registered version is now an instruction: register the next one.
"""

from __future__ import annotations

import pytest

from mechbench_compute import bench, platform_kinds


class _Registry:
    """A registry holding one version per path, as the API does."""

    def __init__(self, held: dict[str, tuple[int, str]] | None = None):
        # path -> (version, content marker)
        self.held = dict(held or {})
        self.calls: list[tuple[str, str]] = []

    def register(self, manifest, **_kw):
        obj = manifest.model_dump(mode="json")
        path, version = obj["path"], obj["version"]
        self.calls.append((path, version))
        current = self.held.get(path)
        if current is None:
            self.held[path] = (int(version), "declared")
            return {"path": path, "version": version}
        held_version, marker = current
        if marker == "declared":
            return {"path": path, "version": str(held_version), "idempotent": True}
        if int(version) <= held_version:
            raise bench.BenchError(
                f"PUT /kinds/{path} -> 409",
                status=409,
                body={"code": "MANIFEST_PINNED",
                      "error": "version is registered with different content",
                      "registeredVersion": str(held_version)},
            )
        self.held[path] = (int(version), "declared")
        return {"path": path, "version": version,
                "supersedes": {"version": str(held_version)}}


@pytest.fixture
def one_manifest(monkeypatch):
    """Just the first declared manifest, so the test reads as one kind."""
    first = platform_kinds.manifests()[0]
    monkeypatch.setattr(platform_kinds, "manifests", lambda: [first])
    return first


class TestRegisterAll:
    def test_a_new_kind_registers_at_version_one(self, monkeypatch, one_manifest):
        reg = _Registry()
        monkeypatch.setattr(bench, "register_kind", reg.register)
        platform_kinds.register_all()
        assert reg.calls == [(one_manifest.path, "1")]
        assert reg.held[one_manifest.path][0] == 1

    def test_an_unchanged_kind_is_left_alone(self, monkeypatch, one_manifest):
        reg = _Registry({one_manifest.path: (1, "declared")})
        monkeypatch.setattr(bench, "register_kind", reg.register)
        platform_kinds.register_all()
        assert reg.calls == [(one_manifest.path, "1")]  # idempotent; no second PUT

    def test_a_changed_kind_is_registered_as_the_next_version(
        self, monkeypatch, one_manifest
    ):
        # The registry holds version 1 with content that is not the
        # declaration's — the eight kinds of phase 2b.
        reg = _Registry({one_manifest.path: (1, "stale")})
        monkeypatch.setattr(bench, "register_kind", reg.register)
        platform_kinds.register_all()
        assert reg.calls == [(one_manifest.path, "1"), (one_manifest.path, "2")]
        assert reg.held[one_manifest.path] == (2, "declared")

    def test_it_follows_a_registry_that_is_already_ahead(
        self, monkeypatch, one_manifest
    ):
        # The number counts changes on THAT registry: a registry at 4
        # goes to 5, whatever any other registry holds.
        reg = _Registry({one_manifest.path: (4, "stale")})
        monkeypatch.setattr(bench, "register_kind", reg.register)
        platform_kinds.register_all()
        assert reg.calls[-1] == (one_manifest.path, "5")

    def test_any_other_refusal_still_raises(self, monkeypatch, one_manifest):
        def forbidden(_manifest, **_kw):
            raise bench.BenchError("PUT -> 403", status=403,
                                   body={"code": "FORBIDDEN"})

        monkeypatch.setattr(bench, "register_kind", forbidden)
        with pytest.raises(bench.BenchError, match="403"):
            platform_kinds.register_all()


class TestBenchErrorCarriesTheRefusal:
    def test_the_code_is_readable_without_matching_prose(self):
        e = bench.BenchError("PUT -> 409", status=409,
                             body={"code": "MANIFEST_PINNED", "registeredVersion": "3"})
        assert e.status == 409
        assert e.code() == "MANIFEST_PINNED"
        assert platform_kinds._registered_version(e) == 3

    def test_a_refusal_with_no_body_says_nothing_about_versions(self):
        e = bench.BenchError("connection reset")
        assert e.code() is None
        assert platform_kinds._registered_version(e) is None
