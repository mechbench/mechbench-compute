"""No file is over the size budget (docs/OPS_LAYOUT.md).

The layout's whole claim is that an agent can read one file and know one
thing. A 3,000-line module breaks that claim silently: nothing fails, the
file just stops being readable, and the next person to need one operation
out of it reads all of it. Nothing else in the suite notices, so this
does.

It is a RATCHET rather than a line in the sand. The files below are
over the budget, each listed at the length it had when it was listed,
and that list is the only thing the gate is lenient about:

  - a file NOT listed must be at or under the budget — a new one over it
    fails, and so does an existing one that grows past it;
  - a LISTED file may shrink, never grow past the count recorded here;
  - a listed file that drops under the budget comes OFF the list, so the
    leniency is spent rather than inherited.

So the list only ever gets shorter, and the numbers in it only ever get
smaller. When it is empty the gate is the plain rule the layout doc
states.

Adding a line to a listed file is not a crime — it is a prompt to take
the same number of lines out of it, or to split it.
"""

from __future__ import annotations

import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute"

#: A file an agent can read in one sitting.
BUDGET = 600

#: The files over the budget, at the length each had when it was listed.
#: Each is a debt, not a dispensation — see the module docstring.
OVER_BUDGET: dict[str, int] = {
    "lexicon/kinds.py": 1101,
    "bench.py": 982,
    "protocol/pipeline.py": 916,
    "distill.py": 881,
    "plot.py": 817,
    "finetune.py": 677,
    "lexicon/_base.py": 638,
    "sandbox.py": 638,
    "geometry.py": 613,
}


def sources() -> dict[str, int]:
    """Every module of the package, by how many lines it is."""
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
