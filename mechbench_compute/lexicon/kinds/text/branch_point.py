from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import COORDS, ID, F

KIND = Kind(
    "text/branch-point",
    "One point of a generation at which k continuations were sampled, with the outcome they reached and how far "
    "it moved from the point before.",
    doc="Written by `text/resample`, one item per branch point per record. A branch point keeps what the model "
        "wrote before it and samples the rest again, `k` times; each continuation's outcome is read at its end. "
        "`shares` is the outcome distribution at the point: with the record's `outcomes`, the mean over the "
        "branches of the model's next-token distribution renormalised over them; without, the share of "
        "branches whose most likely next token was each one. `outcome` is what the full generation reached, "
        "and `share` its share here. `shift` is the Jensen-Shannon divergence in bits between this point's "
        "`shares` and the previous point's, so it measures what keeping `kept`, the text written between "
        "the two, did to the outcome; it is null at the first point. The point with the largest shift in "
        "its record's generation carries `largest: true`: the text the answer hinged on.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "record_id": F("string", "The id of the record the generation began from, also its `coords.record`; "
                                     "`record` is the expression language's word for the row."),
            "step": F("integer", "The first step sampled again: the branches keep steps 0 to step − 1 as the "
                                 "full generation wrote them."),
            "position": F("integer", "Where the branches begin in the sequence: the prompt's length plus "
                                     "`step`, or for a turn the length of the prompt its reply is written to."),
            "turn": F("integer", "Under `where: \"turn\"`, the reply sampled again first, counted from 0."),
            "kept": F("string", "The text the full generation wrote between the previous branch point and this "
                                "one; empty at the first."),
            "k": F("integer", "How many branches were sampled here."),
            "outcome": F("string", "The outcome the full generation reached."),
            "share": F("number", "The share `shares` gives the full generation's outcome."),
            "entropy": F("number", "The entropy of `shares`, in bits."),
            "shift": F("number", "The Jensen-Shannon divergence in bits from the previous point's `shares`; "
                                 "null at the first point."),
            "shares": F("object", "The outcome distribution here: each outcome and its share, summing to 1."),
            "mass": F("number", "With `outcomes`: the mean over branches of the next-token probability on all "
                                "of them together, so a share of almost nothing reads as such."),
            "largest": F("boolean", "True at the point with the largest shift in its record's generation.")},
    required=("id", "record_id", "step", "position", "k", "outcome", "share", "entropy", "shares"),
    key=("id",),
    header={"model": "The model, as a reference.",
            "where": "`position`, `sentence` or `turn`: where the branch points were placed.",
            "k": "How many branches per point.",
            "seed": "The seed every generation's random stream derives from.",
            "temperature": "The sampling temperature.",
            "top_p": "The nucleus setting.",
            "max_tokens": "The longest generation, in tokens: per reply under `turn`, in total otherwise.",
            "cue": "The text appended before an outcome is read, or null.",
            "boundaries": "Under `sentence`: the strings that end a sentence.",
            "generations": "Per record, the full generation: `{record_id, text, outcome, shares, steps, ended}`."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    speak="{record_id} at step {step}: '{outcome}' has {round(share, 2)} of the outcome over {k} branches "
          "({round(entropy, 2)} bits){'' if shift == null else ', ' + str(round(shift, 3)) + ' bits from the "
          "point before'}{'; the largest shift in this generation, after \"' + kept + '\"' if largest else ''}.",
    draw=Draw(mark="line", encoding={"x": "step", "y": "shift", "series": "record_id"}),
)
