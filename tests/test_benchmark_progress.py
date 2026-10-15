from __future__ import annotations

import types

import numpy as np
import pytest

lm_eval = pytest.importorskip("lm_eval")

from mechbench_compute.ops import Context
from mechbench_compute.ops.eval import benchmark

VOCAB = 11
REQUESTS = {"alpha": 7, "beta": 4, "empty": 0}


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (VOCAB - 2)) for c in text]


class Model:
    tokenizer = Tokenizer()
    architecture = types.SimpleNamespace(backend="torch")

    def make_ids(self, ids):
        return list(ids)

    def run(self, ids):
        rows = np.stack([np.sin(np.arange(VOCAB) * (t + 1) + i) for i, t in enumerate(ids)])
        return types.SimpleNamespace(logits=rows[None].astype(np.float32))


def fake_simple_evaluate(model, tasks, **kwargs):
    task = tasks[0]
    requests = [types.SimpleNamespace(args=(f"{task} q{i}:", f" a{i}"))
                for i in range(REQUESTS[task])]
    scored = model.loglikelihood(requests) if requests else []
    acc = sum(1 for _, greedy in scored if greedy) / max(len(scored), 1)
    return {"results": {task: {"acc,none": acc,
                               "loglik,none": sum(lp for lp, _ in scored),
                               "loglik_stderr,none": 0.5}},
            "n-samples": {task: {"original": len(requests), "effective": len(requests)}}}


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(lm_eval, "simple_evaluate", fake_simple_evaluate)
    return Model()


def run_benchmark(model, tasks, ctx):
    ctx.loaded = model
    return benchmark.run(ctx, {}, {"tasks": tasks, "variant": "base"})


def test_progress_fires_per_request_within_one_task(harness):
    ticks = []
    totals = []
    ctx = Context(on_item=lambda *a: ticks.append(len(ticks)),
                  on_start=lambda n: totals.append(n))
    run_benchmark(harness, ["alpha"], ctx)
    assert len(ticks) == REQUESTS["alpha"]
    assert totals[0] == 1
    assert totals[-1] == REQUESTS["alpha"]


def test_done_meets_the_total_across_tasks_and_totals_only_grow(harness):
    done = []
    totals = []
    ctx = Context(on_item=lambda *a: done.append(1), on_start=lambda n: totals.append(n))
    run_benchmark(harness, ["alpha", "empty", "beta"], ctx)
    assert totals == sorted(totals)
    assert len(done) == totals[-1] == REQUESTS["alpha"] + 1 + REQUESTS["beta"]


def test_the_results_are_the_same_with_and_without_progress(harness):
    tasks = ["alpha", "beta", "empty"]
    quiet = run_benchmark(harness, tasks, Context())
    loud = run_benchmark(harness, tasks, Context(on_item=lambda *a: None,
                                                 on_start=lambda n: None))
    assert quiet == loud
    assert {r["task"] for r in quiet["rows"]} == set(tasks)
