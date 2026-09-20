"""Attribution patching (task 000606): every (layer, position) of a causal
trace estimated from one forward and one backward pass. On a model whose
metric is linear in its activations the estimate IS the exact patch; on
Gemma it ranks the cells the exact patch ranks."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import interp
from mechbench_compute.interventions import compose


class _Arch:
    n_layers = 3


class _Tok:
    """Words are ids; decode is the word back. The id is a stable digest
    of the word — `hash()` is salted per process and made this flaky."""

    def encode(self, text):
        import hashlib
        return [int(hashlib.sha256(w.encode()).hexdigest()[:4], 16) % 50 + 2 for w in text.split()]

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


class _LinearModel:
    """Residuals are one-hot-ish embeddings of the ids; each layer adds a
    fixed vector; the logits are a linear read of the LAST position's
    final residual — so any metric linear in the activation (the logit)
    is exactly first-order, and attribution equals the patch."""

    tokenizer = _Tok()
    arch = _Arch()
    D, V = 8, 60

    def __init__(self):
        rng = np.random.default_rng(0)
        self.embed = mx.array(rng.normal(size=(60, self.D)).astype(np.float32))
        self.layer_add = [mx.array(rng.normal(size=(self.D,)).astype(np.float32)) for _ in range(3)]
        self.mix = mx.array(rng.normal(size=(self.D, self.D)).astype(np.float32) * 0.3)
        self.unembed = mx.array(rng.normal(size=(self.D, self.V)).astype(np.float32))

    def tokenize(self, prompt, chat_template=False):
        return mx.array([[1, *self.tokenizer.encode(prompt)]])

    def run(self, ids, hooks=None, capture=None, interventions=None):
        hooks_d, caps = compose(interventions, hooks=hooks, capture=capture)
        h = self.embed[ids]                                   # [1, L, D]
        cache = {}
        for layer in range(3):
            name = f"blocks.{layer}.resid_post"
            # Every position reads the previous position's residual, so a
            # patch at (layer, pos) reaches the last position — linearly.
            shifted = mx.concatenate([mx.zeros_like(h[:, :1]), h[:, :-1]], axis=1)
            h = h + self.layer_add[layer] + shifted @ self.mix
            fn = hooks_d.get(name)
            if fn is not None:
                out = fn(h, None)
                h = out if out is not None else h
            if name in caps:
                cache[name] = h
        logits = h @ self.unembed                              # [1, L, V]

        class R:
            pass
        r = R(); r.logits = logits; r.cache = cache
        return r


class TestOnALinearModel:
    def _run(self, method, metric="logit"):
        model = _LinearModel()
        return interp.patch_trace(
            model, [{"id": "p", "a": "the tower is in paris", "b": "the needle is in seattle"}],
            {"layers": "all", "metric": metric, "method": method, "tracked": {"t": "paris"}})

    def test_attribution_equals_the_exact_patch_where_the_metric_is_linear(self):
        exact = np.array(self._run("exact")["items"][0]["measures"]["recovery"])
        est = np.array(self._run("attribution")["items"][0]["measures"]["recovery"])
        assert exact.shape == est.shape == (3, 5)
        assert np.abs(exact - est).max() < 1e-3
        assert np.abs(exact).max() > 0.1     # and there was something to recover

    def test_the_header_says_which_method_ran(self):
        out = self._run("attribution")
        assert out["method"] == "attribution" and out["metric"] == "logit"
        assert self._run("exact")["method"] == "exact"

    def test_on_a_saturating_metric_it_is_an_estimate(self):
        exact = np.array(self._run("exact", "logprob")["items"][0]["measures"]["recovery"])
        est = np.array(self._run("attribution", "logprob")["items"][0]["measures"]["recovery"])
        # Not the same numbers: the log-softmax is where the first order
        # ends. (What survives on a real model — the ranking — is the
        # Gemma test below; a toy with random weights promises nothing
        # about ranks through a softmax.)
        assert np.abs(exact - est).max() > 1e-3

    def test_refusals_by_name(self):
        with pytest.raises(ValueError, match="method"):
            self._run("guess")
        with pytest.raises(ValueError, match="attribution reads"):
            interp.patch_trace(_LinearModel(), [{"id": "p", "a": "a b", "b": "c d"}],
                               {"method": "attribution", "point": "attn.q"})
        with pytest.raises(ValueError, match="not a residual"):
            interp.patch_trace(_LinearModel(), [{"id": "p", "a": "a b", "b": "c d"}],
                               {"method": "exact", "point": "mlp_out"})


E2B = "mlx-community/gemma-4-e2b-it-bf16"


@pytest.mark.skipif(
    os.environ.get("MECHBENCH_MODEL_TESTS") != "1"
    or not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--" + E2B.replace("/", "--"))),
    reason="set MECHBENCH_MODEL_TESTS=1 with gemma-4-e2b cached",
)
def test_on_gemma_the_estimate_ranks_the_cells_the_patch_ranks():
    """The claim the docs make, measured: on a capital-city pair under
    the logit metric, the estimate agrees with the exact trace on WHICH
    cells matter — not on how much, which a first-order estimate of a
    flip cannot say."""
    from scipy.stats import spearmanr

    from mechbench_compute import Model

    model = Model.load(E2B)
    pair = [{"id": "capital", "template": "chat",
             "a": "Answer with one word: what is the capital of France?",
             "b": "Answer with one word: what is the capital of Japan?",
             "tracked": {"answer": "Paris"}}]
    exact_out = interp.patch_trace(model, pair, {"metric": "logit", "method": "exact"})
    exact = np.array(exact_out["items"][0]["measures"]["recovery"])
    est = np.array(interp.patch_trace(model, pair, {"metric": "logit", "method": "attribution"})
                   ["items"][0]["measures"]["recovery"])
    assert exact.shape == est.shape and exact.shape[0] == model.arch.n_layers
    # A pair worth tracing: the clean prompt says Paris, the corrupt one does not.
    assert exact_out["items"][0]["value_a"] > exact_out["items"][0]["value_b"] + 10
    matter = np.abs(exact) > 1.0
    assert matter.sum() >= 10
    rho = spearmanr(exact[matter], est[matter]).statistic
    assert rho > 0.8, rho
    k = 10
    top = lambda a: set(map(tuple, np.dstack(np.unravel_index(np.argsort(-a.ravel())[:k], a.shape))[0]))  # noqa: E731
    assert len(top(exact) & top(est)) >= 6
    # The same sign on nearly every cell where the exact effect is real
    # (one wrong-signed cell in ~25 is the estimate's known miss).
    agree = np.mean(np.sign(exact[matter]) == np.sign(est[matter]))
    assert agree >= 0.9, agree
