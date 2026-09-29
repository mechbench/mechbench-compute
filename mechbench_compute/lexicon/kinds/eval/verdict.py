from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import COORDS, F, ID

KIND = Kind(
    "eval/verdict",
    "A verdict on one record: a judge's score, label or preference with every vote; or an expectation's pass with the number it was judged on.",
    fields={"id": ID, "coords": COORDS,
            "score": F("number", "Mean score, for a numeric scale."), "spread": F("number", "Standard deviation of the scores."),
            "min": F("number", "Lowest score."), "max": F("number", "Highest score."),
            "label": F("string", "The majority label, for a categorical scale."), "winner": F("string", "`A` or `B`, for a pairwise scale."),
            "counts": F("object", "Label or winner → votes."), "agreement": F("number", "Share of votes for the majority."),
            "rationale": F("string", "The first parsed vote's rationale."),
            "n_votes": F("integer", "Votes cast."), "n_parsed": F("integer", "Votes that could be read."),
            "votes": F("array", "Every vote as `{vote, parsed, order, …}`.", items={"type": "object"}),
            "unparsed": F("boolean", "True when no vote could be read."),
            "expect": F("string", "For an expectation: its type."),
            "entropy_bits": F("number", "For an expectation: the read's entropy."),
            "kl_bits": F("number", "For an expectation: KL from the expected distribution."),
            "mass": F("number", "For an expectation: the probability mass on the named outcomes."),
            "p_expected": F("number", "For an `answer` expectation: the expected token's probability."),
            "pass": F("boolean", "Whether the expectation was met; null when it could not be judged."),
            "note": F("string", "Why it could not be judged, when it could not.")},
    required=("id",),
    key=("id",),
    header={"judge": "Who graded, on what scale, with what rubric.",
            "summary": "For a judge: mean/median/stdev or counts, `n_unparsed`, the position-bias diagnostic. For an expectation: `pass_rate`, `n_pass`, `n_judged`, `n_unjudgeable`.",
            "spend": "What the judging cost.", "name": "A label for the collection.", "description": "Free text beside the name."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    doc="One kind for every evaluation, so a verdict from a judge and one from an expectation sit in one table "
        "and one rate. A judge's verdict keeps every vote with the order the options were shown in, so a "
        "position bias can be seen rather than suspected; an expectation's keeps the number it was judged on "
        "(the entropy, the KL, the mass) beside the pass, and says why when it could not be judged at all.",
)
