"""The fingerprint contract (task 000433).

A run that errors is a nuisance. A run that returns different numbers
with no signal is a corrupted finding that gets written up and built
on — the only failure a reproducibility platform cannot recover from.
`node_fingerprint` is the thing standing between us and that, so what
it does and does not cover is asserted here rather than assumed.
"""
import pathlib

import pytest

import mechbench_compute
from mechbench_compute import resume


BASE = {
    "block": "~canonical/ops/residuals/vectors/1",
    "params": {"layers": [14, 23], "position": "final"},
    "input_hashes": ["sha256:aaa"],
    "core_version": "0.37.0",
    "model": "mlx-community/gemma-4-e2b-it-bf16",
}


def fp(**over):
    return resume.node_fingerprint(**{**BASE, **over})


class TestWhatMovesTheFingerprint:
    def test_identical_inputs_agree(self):
        assert fp() == fp()

    @pytest.mark.parametrize("field,value", [
        ("block", "~canonical/ops/vectors/mst/1"),
        ("params", {"layers": [14, 23], "position": 0}),
        ("input_hashes", ["sha256:bbb"]),
        ("core_version", "0.36.0"),
        ("model", "mlx-community/other-model"),
    ])
    def test_every_component_is_load_bearing(self, field, value):
        assert fp(**{field: value}) != fp()

    def test_param_order_does_not_matter(self):
        # Canonical serialization, not repr: two identical param sets
        # built in different orders are the same computation.
        a = fp(params={"layers": [14], "position": "final"})
        b = fp(params={"position": "final", "layers": [14]})
        assert a == b

    def test_adding_a_param_moves_it(self):
        # The 000431 case: `pool` is new, and a record made with it is
        # not the record made without it.
        assert fp(params={**BASE["params"], "pool": "mean"}) != fp()


class TestWhatDoesNotMoveIt:
    """The gaps, asserted so they are known rather than discovered.

    These are not bugs to fix here — hashing resolved defaults would
    churn every fingerprint whenever a signature is touched. They are
    the reason `core_version` has to be honest, which is what
    TestTheVersionIsHonest is about."""

    def test_an_unstated_default_is_invisible(self):
        # A protocol that omits `bridge_sigma` hashes the same before
        # and after DEFAULT_BRIDGE_SIGMA changes, because params are
        # hashed as DECLARED. Only the version separates them.
        declared = {"name": "corpus-variety"}
        assert fp(params=declared) == fp(params=declared)
        assert fp(params={**declared, "bridge_sigma": 2.0}) != fp(params=declared)

    def test_block_semantics_are_not_hashed_directly(self):
        # There is no per-block semantics version in the body: change
        # what a block DOES without changing its name or its declared
        # params, and only core_version can notice.
        assert "semantics" not in resume.node_fingerprint.__doc__.lower()


class TestTheVersionIsHonest:
    """`core_version` is `mechbench_compute.__version__`, which reads
    installed dist METADATA — a promise the source need not keep.

    Observed on this project: the runner's repo venv reported 0.20.0
    while executing 0.36.0 code, sixteen versions of drift, because an
    editable install's metadata only updates when someone reinstalls.
    Since params are hashed as declared, that string is the ONLY guard
    against reusing work computed by different code."""

    def test_a_source_import_reports_its_source(self):
        # This suite runs against the source tree, so the version must
        # carry a digest of what is actually on disk.
        assert "+src." in mechbench_compute.__version__, (
            "a source/editable import must not report a bare version: "
            f"{mechbench_compute.__version__}"
        )

    def test_an_installed_copy_reports_a_bare_version(self, monkeypatch, tmp_path):
        installed = tmp_path / "venv" / "lib" / "site-packages" / "mechbench_compute"
        installed.mkdir(parents=True)
        (installed / "__init__.py").write_text("x = 1\n")
        monkeypatch.setattr(mechbench_compute, "__file__",
                            str(installed / "__init__.py"))
        assert mechbench_compute._editable_source_digest() is None

    def test_the_digest_is_twelve_hex_and_stable(self):
        d = mechbench_compute._editable_source_digest()
        assert d is not None and len(d) == 12
        assert all(c in "0123456789abcdef" for c in d)
        assert d == mechbench_compute._editable_source_digest()


class TestTheSourceDigest:
    def _tree(self, root: pathlib.Path, files: dict[str, str]):
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return root

    def test_same_content_same_digest(self, tmp_path):
        a = self._tree(tmp_path / "a", {"m.py": "x = 1\n", "p/n.py": "y = 2\n"})
        b = self._tree(tmp_path / "b", {"m.py": "x = 1\n", "p/n.py": "y = 2\n"})
        assert (mechbench_compute._digest_tree(a)
                == mechbench_compute._digest_tree(b))

    def test_changed_content_changes_the_digest(self, tmp_path):
        a = self._tree(tmp_path / "a", {"m.py": "x = 1\n"})
        before = mechbench_compute._digest_tree(a)
        (a / "m.py").write_text("x = 2\n")
        assert mechbench_compute._digest_tree(a) != before

    def test_moving_code_between_files_changes_the_digest(self, tmp_path):
        a = self._tree(tmp_path / "a", {"m.py": "x = 1\n"})
        b = self._tree(tmp_path / "b", {"n.py": "x = 1\n"})
        assert (mechbench_compute._digest_tree(a)
                != mechbench_compute._digest_tree(b))

    def test_touching_without_editing_does_not_change_it(self, tmp_path):
        # mtime-based digests churn on every git checkout; a spurious
        # miss costs real compute, so content is what is hashed.
        a = self._tree(tmp_path / "a", {"m.py": "x = 1\n"})
        before = mechbench_compute._digest_tree(a)
        (a / "m.py").touch()
        assert mechbench_compute._digest_tree(a) == before

    def test_non_python_files_are_ignored(self, tmp_path):
        a = self._tree(tmp_path / "a", {"m.py": "x = 1\n"})
        before = mechbench_compute._digest_tree(a)
        (a / "notes.md").write_text("hello\n")
        assert mechbench_compute._digest_tree(a) == before


class TestResumeLevels:
    def test_an_unlisted_block_restarts(self):
        # Safe by omission: residuals/vectors and vectors/mst are not
        # in BLOCK_RESUME, so no partial of theirs is ever reused.
        assert resume.resume_level("~canonical/ops/vectors/mst/1") == "restart"
        assert resume.resume_level(
            "~canonical/ops/residuals/vectors/1") == "restart"

    def test_restart_satisfies_anything_because_it_recomputes(self):
        # `satisfies` reads backwards until you see that the level is a
        # DEMAND, not a guarantee: a block that restarts asks nothing of
        # the consumer and has no partial to mistrust.
        assert resume.satisfies("restart", "reproducible")

    def test_exchangeable_cannot_promise_bit_identity(self):
        assert not resume.satisfies("exchangeable", "reproducible")

    def test_state_restorable_ranks_with_reproducible(self):
        # Documented: full state capture IS bit-identical.
        assert resume.satisfies("state-restorable", "reproducible")

    def test_an_unknown_level_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match="unknown resume level"):
            resume.satisfies("probably-fine", "reproducible")
