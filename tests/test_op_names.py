"""How an op is named and resolved (docs/LEXICON.md §1; task 000495).

A protocol may spell an op bare (`records/select`), stored
(`~canonical/ops/records/select`), with the retired `/1`, or by a name
from before the 2026-09 renames. All resolve to the bare name; the
retired spellings warn once, naming the replacement and the release
that will refuse them. Every table the executor consults is keyed by
the bare name and nothing else.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import pytest

from mechbench_compute import lexicon
from mechbench_compute.lexicon import (
    ALIASES,
    ALIASES_REMOVED_IN,
    BY_NAME,
    ROOT,
    RetiredOpName,
    canonical_path,
    is_canonical,
    resolve,
)


@pytest.fixture(autouse=True)
def _forget_warnings():
    # `resolve` warns once per process per spelling; each test wants a
    # fresh slate so it can assert on the warning it expects.
    lexicon._warned.clear()
    yield
    lexicon._warned.clear()


class TestResolve:
    @pytest.mark.parametrize("name", sorted(BY_NAME))
    def test_a_bare_name_resolves_to_itself_silently(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve(name) == name

    @pytest.mark.parametrize("name", sorted(BY_NAME))
    def test_the_stored_path_resolves_silently(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve(f"{ROOT}{name}") == name
            assert canonical_path(name) == f"{ROOT}{name}"
            assert canonical_path(f"{ROOT}{name}") == f"{ROOT}{name}"

    @pytest.mark.parametrize("old,new", sorted(ALIASES.items()))
    def test_a_retired_name_resolves_and_warns_once(self, old, new):
        assert new in BY_NAME, f"alias target {new!r} is not an op"
        assert old not in BY_NAME, f"alias key {old!r} is still an op name"
        for spelling in (old, f"{ROOT}{old}/1", f"{old}/1"):
            lexicon._warned.clear()
            with pytest.warns(RetiredOpName) as w:
                assert resolve(spelling) == new
            msg = str(w[0].message)
            assert new in msg and ALIASES_REMOVED_IN in msg
            # Once per process per spelling: the second call is silent.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert resolve(spelling) == new

    def test_the_retired_version_segment_warns_on_a_new_name(self):
        with pytest.warns(RetiredOpName):
            assert resolve("records/select/1") == "records/select"
        lexicon._warned.clear()
        with pytest.warns(RetiredOpName):
            assert resolve(f"{ROOT}records/select/1") == "records/select"

    def test_warn_false_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve("decision-read", warn=False) == "logits/read"

    def test_an_unknown_or_user_op_is_refused_by_name(self):
        for s in ("benji/eval-creativity/ops/marcus-zoo-stats-v1", "records/selekt", "", "records"):
            with pytest.raises(KeyError):
                resolve(s)
            assert not is_canonical(s)
        assert is_canonical("records/select") and is_canonical("grid")


class TestEveryTableIsKeyedByBareNames:
    """The dispatcher, the pure registry, the resume table, the reduce
    algebra and the param table all key by the bare name. A stored path
    or a retired name in any of them would be a second spelling the
    resolver does not see."""

    def _bad(self, keys) -> list[str]:
        return sorted(k for k in keys if k not in BY_NAME)

    def test_pure_registry(self):
        from mechbench_compute.blocks import PURE_BLOCKS
        assert self._bad(PURE_BLOCKS) == []

    def test_resume_table(self):
        from mechbench_compute.resume import BLOCK_RESUME, DYNAMIC_LEVEL
        assert self._bad(BLOCK_RESUME) == []
        assert self._bad(DYNAMIC_LEVEL) == []

    def test_reduce_algebra(self):
        from mechbench_compute.reduce import MONOIDS, PURE_REDUCE_BLOCKS, REDUCE_ALGEBRA
        assert self._bad(REDUCE_ALGEBRA) == []
        assert self._bad(MONOIDS) == []
        assert self._bad(PURE_REDUCE_BLOCKS) == []

    def test_param_table(self):
        from mechbench_compute.block_params import ACCEPTED
        assert set(ACCEPTED) == set(BY_NAME)

    def test_the_dispatch_chain(self):
        import mechbench_compute.protocol as protocol
        src = pathlib.Path(protocol.__file__).read_text()
        compared = set(re.findall(r'block == "([^"]+)"', src))
        assert self._bad(compared) == []
        assert not re.search(r'block == "~canonical', src)

    def test_no_source_spells_the_stored_root_except_the_lexicon(self):
        """`~canonical/ops/` appears where the root is DEFINED and where
        aliases are listed, and in prose — never as a table key."""
        import mechbench_compute
        pkg = pathlib.Path(mechbench_compute.__file__).parent
        offenders = []
        for p in pkg.rglob("*.py"):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if '"~canonical/ops/' in line and p.parent.name != "lexicon":
                    offenders.append(f"{p.relative_to(pkg)}:{i}")
        assert offenders == [], offenders


class TestTheResumeAndReduceLookupsAcceptAnySpelling:
    def test_resume_level_by_old_name(self):
        from mechbench_compute.resume import item_resumable, resume_level
        assert resume_level("~canonical/ops/generate/1") == resume_level("text/generate")
        assert item_resumable("decision-read") == item_resumable("logits/read") is True

    def test_reduce_algebra_by_old_name(self):
        from mechbench_compute.reduce import algebra
        assert algebra("group-stats") == algebra("records/summarize") == "monoid"

    def test_check_params_by_old_name(self):
        from mechbench_compute.block_params import check_params
        check_params("~canonical/ops/decision-read/1", {"tracked": {"a": "a"}})
        with pytest.raises(ValueError, match="logits/read"):
            check_params("~canonical/ops/decision-read/1", {"nope": 1})
