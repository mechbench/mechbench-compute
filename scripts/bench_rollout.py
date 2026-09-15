#!/usr/bin/env python3
"""Where the time goes in a decision read with rollout — the per-condition
budget the release gate holds `logits/read` to.

A decision read with rollout does, per condition: render the prompt,
prefill it once into a KV cache, summarise the decision-position
distribution, then expand the most likely complete outcomes by feeding
each partial outcome against a copy of the prompt cache. The count of
forwards is the algorithm's; the cost per forward is the model stack's
and the machine's. This script measures both on the model the
experiments run, so a release cannot slow the path without someone
seeing the number.

    python scripts/bench_rollout.py                 # 12 synthetic conditions
    python scripts/bench_rollout.py --n 24 --budget-ms 150

Prints a table of medians per phase and the implied cost per expansion
forward; exits non-zero when that cost is over `--budget-ms`. Needs the
model in the local Hugging Face cache and an otherwise idle machine —
a busy GPU makes the number about the machine, not the code.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

MODEL = "mlx-community/gemma-4-e2b-it-bf16"
ROLLOUT = {"top_k": 10, "max_forwards": 96, "terminators": ['"']}

#: Chat-shaped conditions like a decision battery's: a system line, a
#: user turn that ends at a JSON prefill, so the read lands on one token.
SYSTEM = "You are a careful assistant. Answer with a single JSON string."
USERS = [
    "Pick a first name for the {who} in this story, at random. Reply as {{\"name\": \"…\"}}.",
    "Roll a fair six-sided die and report the result as {{\"roll\": \"…\"}}.",
    "Choose one colour for the {who}'s coat, any colour. Reply as {{\"colour\": \"…\"}}.",
    "Name the city the {who} moved to. Reply as {{\"city\": \"…\"}}.",
]
WHO = ["lighthouse keeper", "cartographer", "night baker", "ferry pilot", "archivist", "beekeeper"]


def conditions(n: int) -> list[dict]:
    out = []
    for i in range(n):
        user = USERS[i % len(USERS)].format(who=WHO[i % len(WHO)])
        out.append({"id": f"c{i}", "system": SYSTEM, "user": user, "prefill": '{"'})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--budget-ms", type=float, default=150.0,
                    help="the most an expansion forward may cost, in milliseconds")
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    import mechbench_compute
    from mechbench_compute import distill
    from mechbench_compute import shapes as S
    from mechbench_compute.model import Model

    def ms(t0: float) -> float:
        return (time.perf_counter() - t0) * 1000

    versions = [f"compute {mechbench_compute.__version__}", f"mlx {mx.__version__}"]
    for name in ("mlx_vlm", "mlx_lm"):
        try:
            versions.append(f"{name} {__import__(name).__version__}")
        except Exception:  # noqa: BLE001 — a stack without it is fine
            pass
    print("  ".join(versions))

    t0 = time.perf_counter()
    model = Model.load(args.model)
    print(f"model load {ms(t0):.0f} ms")
    tok = model.tokenizer
    conds = conditions(args.n)
    r = distill.render(model, conds[0])
    distill.prefill_decision(model, r.ids)  # warm the stack once

    phases: dict[str, list[float]] = {k: [] for k in ("render", "prefill", "distribution", "expand", "condition")}
    pieces: dict[str, list[float]] = {k: [] for k in ("cache_copy", "suffix_forward_5", "dist_top50")}
    forwards: list[int] = []
    for cond in conds:
        tc = time.perf_counter()
        t = time.perf_counter(); r = distill.render(model, cond); phases["render"].append(ms(t))
        t = time.perf_counter(); prefill = distill.prefill_decision(model, r.ids); phases["prefill"].append(ms(t))
        t = time.perf_counter()
        lp = np.array(prefill[1] - mx.logsumexp(prefill[1])).astype(np.float64)
        S.distribution(lp, tok, top_k=10, tracked={})
        phases["distribution"].append(ms(t))
        t = time.perf_counter()
        out = distill.expand_top_outcomes_cached(model, tok, r.ids, ROLLOUT, prefill=prefill)
        phases["expand"].append(ms(t))
        forwards.append(int(out["forwards_used"]))
        phases["condition"].append(ms(tc))
        cache = prefill[0]
        t = time.perf_counter()
        cc = distill._copy_prefix_cache(cache)
        for c in cc:
            mx.eval([v for v in vars(c).values() if isinstance(v, mx.array)])
        pieces["cache_copy"].append(ms(t))
        t = time.perf_counter()
        cc = distill._copy_prefix_cache(cache)
        o = model.lm(mx.array([r.ids[-5:]]), cache=cc)
        row = (o.logits if hasattr(o, "logits") else o)[0, -1, :].astype(mx.float32)
        mx.eval(row)
        pieces["suffix_forward_5"].append(ms(t))
        t = time.perf_counter()
        p = np.exp(np.array(row - mx.logsumexp(row)).astype(np.float64))
        np.argpartition(-p, 50)[:50]
        pieces["dist_top50"].append(ms(t))

    print(f"\n{len(conds)} conditions, ~{len(r.ids)} prompt tokens, {statistics.mean(forwards):.1f} forwards per condition")
    for k, v in phases.items():
        print(f"  {k:18s} median {statistics.median(v):8.1f} ms")
    print("expansion pieces:")
    for k, v in pieces.items():
        print(f"  {k:18s} median {statistics.median(v):8.1f} ms")
    per_forward = statistics.median(phases["expand"]) / max(1.0, statistics.mean(forwards) - 1)
    print(f"\nper expansion forward: {per_forward:.0f} ms (budget {args.budget_ms:.0f})")
    if per_forward > args.budget_ms:
        print("OVER BUDGET — a slower rollout path, or a busy machine; rerun idle before releasing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
