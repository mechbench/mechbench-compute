from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechbench_compute.blocks.items import _items
from mechbench_compute.lexicon._base import In, Op, Output

OP = Op(
    name="eval/expect",
    summary=(
        "Judge each decision read against an expectation carried as data — "
        "uniform over the outcomes, a required answer, a minimum entropy, or "
        "a target distribution — and report pass/fail with the rate."
    ),
    description="""\
Results (from `logits/read`) and expectations are joined on `id`. Each
expectation record has an `expect` object:

| `type` | Passes when | Fields |
|---|---|---|
| `uniform` | the KL divergence from uniform over the named outcomes is at most `max_kl_bits` | `over` (the outcomes), `max_kl_bits` (default 0.1) |
| `weights` | the KL divergence from the normalised `weights` is at most `max_kl_bits` | `weights` (outcome → weight), `max_kl_bits` |
| `answer` | the expected token's probability is at least `min_p` | `value`, `min_p` (default 0.99) |
| `min_entropy` | the read's entropy is at least `bits` | `bits` |
| `absent` | the total mass on the named outcomes is at most `max_p` — outcomes that should not be said, such as the genres already in a list | `over` (the outcomes), `max_p` (default 0.01) |

Outcome masses come from the read's `tracked` (each outcome by its own
name, a token or a `complete` outcome), else from its `top` by exact token
text. A read with no mass on any outcome is reported as *unjudgeable*
(`pass: null` with a `note`) rather than counted as a failure — a hole in
the read must not masquerade as a verdict. For `absent` the bar is higher:
every named outcome must be in `tracked`, since an outcome the read never
scored has no mass to report, not a mass of zero.

The header's `summary` carries the pass rate: the number a write-up cites.
""",
    inputs=(
        In("results", "logits/distribution",
           "The decision reads — a collection of `logits/decision`, or of any "
           "kind that extends `logits/distribution`.", many=True),
        In("expectations", "records/record",
           "Records `{id, expect}`, one per result to judge.", many=True),
    ),
    output=(
        Output('eval/verdict', collection=True, doc='One verdict per judged result: `id`, `coords`, `expect`, `entropy_bits`, `kl_bits`, `mass`, `p_expected`, and `pass` (null with a `note` when unjudgeable). The header\'s `summary` carries `pass_rate`, `n_pass`, `n_judged` and `n_unjudgeable`.')
    ),
    params=(),
    example={},
    example_inputs={
        "results": {"$ref": {"bench": "you/lab/reads"}},
        "expectations": [
            {"id": "d6", "expect": {"type": "uniform",
                                    "over": ["1", "2", "3", "4", "5", "6"],
                                    "max_kl_bits": 0.1}},
        ],
    },
)


def run(ctx, inputs, params):
    return eval_expectation(inputs, params)


def eval_expectation(inputs: Mapping[str, Any],
                     params: Mapping[str, Any]) -> dict[str, Any]:
    """The first member of the eval block family (~canonical/ops/eval/):
    judge decision-read results against per-condition EXPECTATIONS
    carried as data, publishing a metric table with verdicts.

    Expectation kinds (per record, joined on id):
      {"type": "uniform", "over": [outcomes], "max_kl_bits": t}
          -> kl_bits from uniform over the outcome masses; pass iff
             kl_bits <= t and the outcomes carry real mass.
      {"type": "answer", "value": tok, "min_p": t}
          -> p_expected from the read's top tokens; pass iff >= t.
      {"type": "min_entropy", "bits": t}
          -> pass iff the decision entropy >= t (diversity floor).
      {"type": "weights", "weights": {outcome: w}, "max_kl_bits": t}
          -> kl_bits from the NORMALIZED weights over the outcome
             masses (the shaped-target battery: a rung is judged
             against its OWN target, not uniform); pass iff <= t.
      {"type": "absent", "over": [outcomes], "max_p": t}
          -> mass, the total on outcomes that should not be said (a
             list slot's repeats); pass iff mass <= t. Every named
             outcome must have been READ (in `tracked`): absence is
             never inferred from an outcome the read did not score.

    The aggregate row (id "ALL") carries the pass rate — the number a
    publication cites.
    """
    import math

    results = _items(inputs["results"])
    expectations = {r["id"]: r["expect"] for r in _items(inputs["expectations"])}
    from mechbench_compute import shapes as S
    from mechbench_compute.lexicon import kinds as K

    rows = []
    n_pass = 0
    n_judged = 0
    n_unjudgeable = 0
    for c in results:
        exp = expectations.get(c["id"])
        if not exp:
            continue
        expect_type = exp.get("type") or exp["kind"]
        # One shape to read: the distribution's `tracked` holds each
        # named outcome's mass, `top` the ranked tokens. A read written
        # before the shape existed is read through `distribution_of`.
        dist = S.distribution_of(c)
        tracked = dist.get("tracked") or {}
        row: dict[str, Any] = {"id": c["id"], "coords": dict(c.get("coords") or {}),
                               "expect": expect_type,
                               "entropy_bits": dist.get("entropy_bits")}

        def mass_of(name: str) -> float:
            t = tracked.get(str(name))
            if t is not None and t.get("p") is not None:
                return float(t["p"])
            # Not tracked: the ranked tokens, by exact text.
            return sum(float(t["p"]) for t in dist.get("top") or []
                       if str(t["token"].get("text")).strip() == str(name).strip()
                       and t.get("p") is not None)

        ok = False
        if expect_type == "uniform":
            over = exp["over"]
            ps = [mass_of(o) for o in over]
            tot = sum(ps)
            if tot > 0:
                kl = sum(q / tot * math.log2((q / tot) / (1.0 / len(over)))
                         for q in ps if q > 0)
                row["kl_bits"] = round(kl, 4)
                row["mass"] = round(tot, 4)
                ok = kl <= float(exp.get("max_kl_bits", 0.1))
            else:
                # Nothing to judge is not a failure — it is a hole in
                # the read, and it must not masquerade as one more
                # False among real verdicts.
                row["pass"] = None
                row["note"] = "unjudgeable: no mass on any outcome in the read"
                n_unjudgeable += 1
                rows.append(row)
                continue
        elif expect_type == "weights":
            wsum = sum(float(v) for v in exp["weights"].values())
            target = {str(k): float(v) / wsum
                      for k, v in exp["weights"].items() if float(v) > 0}
            ps = [mass_of(o) for o in target]
            tot = sum(ps)
            if tot > 0:
                kl = sum(
                    (q / tot) * math.log2((q / tot) / target[o])
                    for o, q in zip(target, ps) if q > 0)
                row["kl_bits"] = round(kl, 4)
                row["mass"] = round(tot, 4)
                ok = kl <= float(exp.get("max_kl_bits", 0.1))
            else:
                row["pass"] = None
                row["note"] = "unjudgeable: no mass on any outcome in the read"
                n_unjudgeable += 1
                rows.append(row)
                continue
        elif expect_type == "answer":
            want = str(exp["value"])
            p = mass_of(want) if (want in tracked or dist.get("top")) else None
            if p == 0.0 and want not in tracked:
                p = None
            row["p_expected"] = round(p, 4) if p is not None else None
            ok = p is not None and p >= float(exp.get("min_p", 0.99))
        elif expect_type == "min_entropy":
            ok = float(dist.get("entropy_bits") or 0.0) >= float(exp["bits"])
        elif expect_type == "absent":
            over = [str(o) for o in exp["over"]]
            unread = [o for o in over if o not in tracked]
            if unread:
                row["pass"] = None
                row["note"] = f"unjudgeable: not read — {unread[:5]}"
                n_unjudgeable += 1
                rows.append(row)
                continue
            mass = sum(mass_of(o) for o in over)
            row["mass"] = round(mass, 6)
            ok = mass <= float(exp.get("max_p", 0.01))
        else:
            raise ValueError(f"unknown expectation type: {expect_type!r}")
        row["pass"] = ok
        n_judged += 1
        n_pass += int(ok)
        rows.append(row)
    return K.collection(
        "eval/verdict", rows,
        name=params.get("name", "expectation-eval"),
        description=params.get("description", ""),
        summary={"pass_rate": round(n_pass / n_judged, 4) if n_judged else None,
                 "n_pass": n_pass, "n_judged": n_judged,
                 "n_unjudgeable": n_unjudgeable})
