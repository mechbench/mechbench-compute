from __future__ import annotations

import pathlib
import re
import warnings

import pytest

from mechbench_compute.lexicon import (
    ALIASES_REMOVED_IN,
    BY_NAME,
    RETIRED,
    ROOT,
    canonical_path,
    explain_unknown,
    is_canonical,
    resolve,
)


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

    @pytest.mark.parametrize("old,new", sorted(RETIRED.items()))
    def test_a_retired_name_is_refused_and_names_its_replacement(self, old, new):
        assert new in BY_NAME, f"{new!r} is not an op"
        assert old not in BY_NAME, f"{old!r} is still an op name"
        for spelling in (old, f"{ROOT}{old}", f"{ROOT}{old}/1", f"{old}/1"):
            with pytest.raises(KeyError):
                resolve(spelling)
            assert not is_canonical(spelling)
            msg = explain_unknown(spelling)
            assert new in msg and ALIASES_REMOVED_IN in msg

    def test_the_version_segment_is_refused_on_a_current_name_too(self):
        for spelling in ("records/select/1", f"{ROOT}records/select/1"):
            with pytest.raises(KeyError):
                resolve(spelling)
            assert "records/select" in explain_unknown(spelling)

    def test_warn_false_is_still_accepted_and_still_resolves(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve("logits/read", warn=False) == "logits/read"

    def test_an_unknown_or_user_op_is_refused_by_name(self):
        for s in ("benji/eval-creativity/ops/marcus-zoo-stats-v1",
                  "records/selekt", "", "records"):
            with pytest.raises(KeyError):
                resolve(s)
            assert not is_canonical(s)
            assert s in explain_unknown(s) or explain_unknown(s)
        assert is_canonical("records/select")
        assert "docs.mechbench.ai" in explain_unknown("records/selekt")


class TestEveryTableIsKeyedByBareNames:
    def _bad(self, keys) -> list[str]:
        return sorted(k for k in keys if k not in BY_NAME)

    def test_pure_registry(self):
        from mechbench_compute import ops
        assert self._bad(ops.find_standalone()) == []

    def test_resume_table(self):
        from mechbench_compute.resume import BLOCK_RESUME, DYNAMIC_LEVEL
        assert self._bad(BLOCK_RESUME) == []
        assert self._bad(DYNAMIC_LEVEL) == []

    def test_reduce_algebra(self):
        from mechbench_compute.reduce import MONOIDS, REDUCE_ALGEBRA
        assert self._bad(REDUCE_ALGEBRA) == []
        assert self._bad(MONOIDS) == []

    def test_param_table(self):
        from mechbench_compute.block_params import ACCEPTED
        assert set(ACCEPTED) == set(BY_NAME)

    def test_the_dispatch_chain(self):
        from mechbench_compute import protocol
        src = pathlib.Path(protocol.__file__).read_text()
        compared = set(re.findall(r'block == "([^"]+)"', src))
        assert self._bad(compared) == []
        assert not re.search(r'block == "~canonical', src)

    def test_no_source_spells_the_stored_root_except_the_lexicon(self):
        import mechbench_compute
        pkg = pathlib.Path(mechbench_compute.__file__).parent
        offenders = []
        for p in pkg.rglob("*.py"):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if '"~canonical/ops/' in line and p.parent.name != "lexicon":
                    offenders.append(f"{p.relative_to(pkg)}:{i}")
        assert offenders == [], offenders


class TestTheLookupsAroundIt:
    def test_resume_level_falls_through_on_a_retired_name(self):
        from mechbench_compute.resume import item_resumable, resume_level
        assert resume_level("~canonical/ops/text/generate") == \
            resume_level("text/generate")
        assert item_resumable("logits/read") is True
        assert resume_level("decision-read") == "restart"
        assert item_resumable("decision-read") is False

    def test_reduce_algebra_falls_through(self):
        from mechbench_compute.reduce import algebra
        assert algebra("records/summarize") == "monoid"
        assert algebra("group-stats") != "monoid"

    def test_check_params_refuses_a_retired_block_by_name(self):
        from mechbench_compute.block_params import check_params
        check_params("~canonical/ops/logits/read", {"tracked": {"a": "a"}})
        with pytest.raises(ValueError, match="logits/read"):
            check_params("logits/read", {"nope": 1})
