from __future__ import annotations

from mechbench_compute.lexicon._base import Draw, Kind
from mechbench_compute.lexicon.values import COORDS, ID, F

KIND = Kind(
    "intervene/faithfulness",
    "How much of a behaviour one circuit carries alone, and how much is lost without it, over a set of prompts.",
    doc="Written by `intervene/ablate-circuit`, one item per circuit. Four means over the records: `m_full`, "
        "the model untouched; `m_empty`, the floor the effect is measured from; `m_circuit`, everything "
        "but the circuit removed; `m_without`, the circuit removed. The floor is the header's `reference`: "
        "under `empty`, every component of the universe removed; under `base`, the model without its "
        "adapter. The field keeps the name `m_empty` under either, since it is the floor either way. `faithfulness` is "
        "(m_circuit − m_empty) / (m_full − m_empty) and `completeness` is (m_full − m_without) / "
        "(m_full − m_empty): ratios of means, not means of per-record ratios, since a record whose full and "
        "empty readings nearly agree would make its own ratio arbitrarily large. Neither is clipped to "
        "[0, 1]; a circuit whose removal helps the behaviour reads as completeness below 0. The interval is "
        "a bootstrap over records, paired, so the four readings of a record are resampled together. A "
        "ratio is null when m_full and m_empty agree.",
    extends="records/record",
    fields={"id": ID, "coords": COORDS,
            "circuit": F("string", "The circuit's id."),
            "size": F("integer", "How many components the circuit has."),
            "n": F("integer", "How many prompts were measured."),
            "faithfulness": F("number", "(m̄_C − m̄_∅) / (m̄_M − m̄_∅): the share of the effect the circuit "
                                        "carries alone."),
            "completeness": F("number", "(m̄_M − m̄_¬C) / (m̄_M − m̄_∅): the share of the effect lost when the "
                                        "circuit is removed."),
            "faithfulness_lo": F("number", "The lower end of faithfulness's interval."),
            "faithfulness_hi": F("number", "The upper end of faithfulness's interval."),
            "completeness_lo": F("number", "The lower end of completeness's interval."),
            "completeness_hi": F("number", "The upper end of completeness's interval."),
            "m_full": F("number", "m̄_M: the metric's mean with the model untouched."),
            "m_circuit": F("number", "m̄_C: the mean with everything but the circuit removed."),
            "m_without": F("number", "m̄_¬C: the mean with the circuit removed."),
            "m_empty": F("number", "m̄_∅: the floor's mean — the whole universe removed under reference "
                                   "`empty`, the model without its adapter under `base`.")},
    required=("id", "circuit", "size", "n", "faithfulness", "completeness"),
    key=("id",),
    header={"conditions": "Per record: `{id, target, variants, m_full, m_empty, template}`, with `own_top1` "
                          "where the target is not the model's own first choice; `m_empty` is the reference reading.",
            "n_off_top1": "How many records tracked a target the model would not itself have said.",
            "ablation": "`zero` or `mean`: how components were removed.",
            "metric": "The readout at the decision position: `logprob`, `prob`, `logit`, `entropy`, "
                      "`entropy_outcomes` or `mass_outcomes`.",
            "reference": "`empty` or `base`: what `m_empty` is — every component of the universe removed, or "
                         "the model without its adapter.",
            "universe": "The universe the circuits share.",
            "level": "The bootstrap interval's level.",
            "resamples": "How many bootstrap resamples.",
            "seed": "The bootstrap's seed.",
            "held_out": "True when the records are not the ones the circuits were found on (their `task.ids`)."},
    renderer={"primitive": "table", "field_map": {"rows": "items"}},
    speak="{circuit}: alone it keeps {round(faithfulness, 2)} of the effect ({round(faithfulness_lo, 2)} to "
          "{round(faithfulness_hi, 2)}); removed, the model loses {round(completeness, 2)}; {n} prompts, "
          "{header.ablation} ablation, {header.metric}"
          "{', against the base model' if header.reference == 'base' else ''}.",
    draw=Draw(mark="bar", encoding={"x": "circuit", "y": "faithfulness",
                                    "lo": "faithfulness_lo", "hi": "faithfulness_hi"}),
)
