from __future__ import annotations

import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute"

BUDGET = 600

OVER_BUDGET: dict[str, int] = {
    "lexicon/kinds.py": 1026,
    "bench.py": 822,
}


def sources() -> dict[str, int]:
    return {str(p.relative_to(PKG)): len(p.read_text().splitlines())
            for p in sorted(PKG.rglob("*.py"))
            if "__pycache__" not in p.parts}


class TestTheBudget:
    def test_no_unlisted_file_is_over_the_budget(self):
        over = {rel: n for rel, n in sources().items()
                if n > BUDGET and rel not in OVER_BUDGET}
        assert not over, (
            f"over the {BUDGET}-line budget and not on the list: "
            + ", ".join(f"{rel} ({n} lines)" for rel, n in sorted(over.items()))
            + ". Split it into the topics it holds — one file per topic, "
            "named for the topic (docs/OPS_LAYOUT.md). Adding it to "
            "OVER_BUDGET is not the fix; that list only ever shrinks.")

    @pytest.mark.parametrize("rel", sorted(OVER_BUDGET))
    def test_a_listed_file_still_exists(self, rel):
        assert rel in sources(), (
            f"{rel} is on the over-budget list and is not in the package. "
            f"If it was split or removed, take it off the list.")

    @pytest.mark.parametrize("rel", sorted(OVER_BUDGET))
    def test_a_listed_file_only_shrinks(self, rel):
        now, was = sources().get(rel, 0), OVER_BUDGET[rel]
        assert now <= was, (
            f"{rel} was {was} lines when the gate went in and is {now} now. "
            f"A file over the budget may only get shorter. Take {now - was} "
            f"line(s) back out, or split it and drop it off the list.")

    @pytest.mark.parametrize("rel", sorted(OVER_BUDGET))
    def test_a_listed_file_under_the_budget_comes_off_the_list(self, rel):
        now = sources().get(rel, 0)
        assert now > BUDGET, (
            f"{rel} is {now} lines, under the {BUDGET}-line budget: remove it "
            f"from OVER_BUDGET. The list is the debt, and this one is paid.")
