from __future__ import annotations

from mechbench_compute.lexicon._base import Kind
from mechbench_compute.lexicon.values import F

KIND = Kind(
    "adapter/operator",
    "A trained operator: a function of what a mask selects on the residual stream after chosen layers, its parameters, the base it was trained on, and the training's methods section.",
    fields={"base_model": F("string", "The base it was trained on."),
            "trained_on": F("object", "`{base, adapters}` — the stack it was trained on."),
            "operator": F("object", "`{point, layers, positions, gate, mask, function, degree?, width?, penalty?, "
                                    "d, params, effective}`: its form, the residual width it acts on, how many "
                                    "parameters it has and how many of them moved off the identity."),
            "parameters": F("object", "Layer → parameter name → values: `a`, `b` for `affine` on dimensions; "
                                      "`P`, `dW`, `b` on a learned subspace (R is P's rows made orthonormal, in "
                                      "order); `c` (one row per power) for `polynomial`; `W1`, `b1`, `W2`, `b2` "
                                      "for `mlp`; `gate.weight` and `gate.bias` when gated."),
            "train": F("object", "Steps, lr, seed, batch, final loss, the target spec, and the counts.")},
    required=("operator", "parameters"),
    doc="What `adapter/train` produces with `operator`. A model reference carries it beside LoRA adapters "
        "(`{\"base\": …, \"adapters\": [\"you/lab/temperature\"]}`), and every node that loads the model "
        "attaches it: on every forward pass, after each of its layers, the residual stream passes through "
        "it, as it did in training. `effective` counts the parameters that moved off the identity (an "
        "`l1` penalty puts the ones the target does not need back on it exactly), and with them each "
        "learned subspace and gate they act through, which is the number to read beside a LoRA's.",
    speak="{operator.function} operator after layer{'' if len(operator.layers) == 1 else 's'} "
          "{', '.join([str(i) for i in operator.layers])}"
          "{'' if operator.mask is None else (' on a learned subspace of rank ' + str(operator.mask.rank) "
          "if 'rank' in operator.mask else ' at dimension' + ('' if len(operator.mask) == 1 else 's') + ' ' "
          "+ ', '.join([str(i) for i in operator.mask]))}"
          "{', gated' if operator.gate else ''}: {operator.effective} of {operator.params} parameters off "
          "the identity{'' if train.final_loss is None else '; final loss ' + str(train.final_loss)}.",
)
