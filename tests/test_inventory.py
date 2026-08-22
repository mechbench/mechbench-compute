"""The model inventory (task 000285).

The point of these is the arithmetic nobody expects: revisions of one
repository share their blobs, so a revision's apparent size is not what
deleting it returns. Measured on a real cache the difference was 68.6 GB
apparent against 6.6 MB actually reclaimable — which is the difference
between a useful tool and one that talks people into pointless deletions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mechbench_compute import inventory
from mechbench_compute.inventory import RepoInventory, RevisionInfo


def rev(commit, size, reclaimable, refs=()):
    return RevisionInfo(
        commit=commit,
        size_bytes=size,
        reclaimable_bytes=reclaimable,
        last_modified=datetime(2026, 8, 1, tzinfo=timezone.utc),
        refs=tuple(refs),
        path=Path("/tmp") / commit,
    )


class TestRevision:
    def test_a_revision_no_ref_points_at_is_superseded(self):
        assert rev("a" * 40, 100, 100).superseded is True

    def test_the_one_main_points_at_is_not(self):
        assert rev("b" * 40, 100, 100, refs=("main",)).superseded is False

    def test_short_is_the_form_people_actually_quote(self):
        assert rev("448c70a4ea86b1ad", 0, 0).short == "448c70a4"


class TestRepo:
    def test_main_commit(self):
        repo = RepoInventory(
            "mlx-community/gemma-4",
            disk_bytes=26_000_000_000,
            revisions=(rev("a" * 40, 16_000, 0), rev("e" * 40, 15_900, 10_200, ("main",))),
        )
        assert repo.main_commit == "e" * 40

    def test_main_commit_is_none_when_nothing_points_at_one(self):
        repo = RepoInventory("x", 10, (rev("a" * 40, 10, 10),))
        assert repo.main_commit is None

    def test_reclaimable_counts_only_superseded_revisions(self):
        # The kept revision's 10.2 GB is not "reclaimable" — you would
        # have to delete the model you are using to get it.
        repo = RepoInventory(
            "mlx-community/gemma-4",
            disk_bytes=26_300,
            revisions=(
                rev("a" * 40, 16_000, 16),
                rev("b" * 40, 16_000, 0),
                rev("e" * 40, 15_900, 10_200, ("main",)),
            ),
        )
        assert repo.reclaimable_bytes == 16
        assert len(repo.superseded) == 2

    def test_apparent_size_wildly_exceeds_the_disk_it_occupies(self):
        """The property that makes the whole distinction necessary."""
        repo = RepoInventory(
            "mlx-community/gemma-4-E4B-it-bf16",
            disk_bytes=26_300,
            revisions=tuple(rev(c * 40, 16_000, 16) for c in "abcd"),
        )
        apparent = sum(r.size_bytes for r in repo.revisions)
        assert apparent == 64_000
        assert repo.disk_bytes == 26_300
        # Deleting three of the four returns 48 bytes, not 48_000.
        assert repo.reclaimable_bytes < repo.disk_bytes / 100


class TestDelete:
    def test_nothing_asked_for_is_nothing_done(self):
        assert inventory.delete_revisions([]) == 0

    def test_an_unknown_commit_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="not in the local cache"):
            inventory.delete_revisions(["0" * 40])

    def test_it_refuses_an_abbreviated_commit(self):
        # Deleting weights on a prefix match is not a risk worth taking.
        with pytest.raises(ValueError, match="in full"):
            inventory.delete_revisions(["448c70a4"])


class TestFormatBytes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (1_500, "1.5 KB"),
            (26_300_000_000, "26.3 GB"),
            (2_500_000_000_000, "2.5 TB"),
        ],
    )
    def test_renders_for_a_table(self, value, expected):
        assert inventory.format_bytes(value) == expected


class TestScan:
    def test_reads_the_cache_without_a_compute_backend(self, monkeypatch):
        """inventory is one of the modules that must load anywhere: it is
        what reports on a machine that cannot run anything."""
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda n, *a, **k: None if n.split(".")[0] == "mlx" else real(n, *a, **k),
        )
        from mechbench_compute import backends

        assert backends.active() is None
        # Still importable, still callable.
        assert isinstance(inventory.format_bytes(1), str)
