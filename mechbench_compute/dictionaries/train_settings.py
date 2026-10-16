from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

DERIVATIONS = ("sae", "crosscoder")

LATER_DERIVATIONS = ("transcoder",)

TRAINERS = ("batchtopk", "jumprelu")

AUX_COEFFICIENT = 1 / 32

THRESHOLD_DECAY = 0.99

ADAM_BETAS = (0.9, 0.999)

ADAM_EPS = 1e-8

DECAY_SHARE = 0.2

WARMUP_SHARE = 0.05

MAX_WARMUP_STEPS = 1000

DEAD_TOKENS = 10_000_000

DENSE_RATE = 0.1

BANDWIDTH = 0.01

SPARSITY_COEFFICIENT = 1.0

CURVE_POINTS = 1000


@dataclass(frozen=True)
class Settings:
    layer: int
    point: str
    width: int
    activation: str
    k: int
    l0: float
    tokens: int
    context: int
    batch_tokens: int
    buffer_tokens: int
    lr: float
    seed: int
    checkpoint_tokens: int
    heldout_tokens: int
    corpus: Any
    steps: int
    steps_per_chunk: int
    warmup_steps: int
    decay_steps: int
    dead_tokens: int
    rows: int
    heldout_windows: int
    curve_every: int
    derivation: str = "sae"

    @property
    def models(self) -> int:
        return 2 if self.derivation == "crosscoder" else 1

    def lr_at(self, step: int) -> float:
        warm = min(1.0, step / self.warmup_steps) if self.warmup_steps else 1.0
        tail = min(1.0, (self.steps - step + 1) / self.decay_steps) if self.decay_steps else 1.0
        return self.lr * warm * tail

    def sparsity_at(self, step: int) -> float:
        return SPARSITY_COEFFICIENT * (min(1.0, step / self.warmup_steps) if self.warmup_steps else 1.0)

    def bandwidth(self, d: int) -> float:
        return BANDWIDTH * math.sqrt(d)

    def aux_k(self, d: int) -> int:
        return max(1, min(self.width, d // 2))

    def hyperparameters(self) -> dict[str, Any]:
        own = {k: v for k, v in asdict(self).items() if k not in ("corpus", "derivation")}
        return {**own, "adam": {"betas": list(ADAM_BETAS), "eps": ADAM_EPS},
                "aux_coefficient": AUX_COEFFICIENT if self.activation == "batchtopk" else None,
                "threshold_decay": THRESHOLD_DECAY if self.activation == "batchtopk" else None,
                "sparsity_coefficient": SPARSITY_COEFFICIENT if self.activation == "jumprelu" else None,
                "bandwidth_per_sqrt_d": BANDWIDTH if self.activation == "jumprelu" else None,
                "dense_rate": DENSE_RATE}


def read_int(name: str, given: Any, default: Any, least: int) -> int:
    value = default if given is None else given
    if not isinstance(value, int) or isinstance(value, bool) or value < least:
        raise ValueError(f"dictionary/train: `{name}` is a whole number, at least {least}, not {value!r}")
    return value


def read_settings(params: Mapping[str, Any]) -> Settings:
    derivation = params.get("derivation") or "sae"
    if derivation in LATER_DERIVATIONS:
        raise ValueError(f"dictionary/train: the derivation {derivation!r} is not trained yet; this "
                         "operation trains sparse autoencoders (`sae`) so far")
    if derivation not in DERIVATIONS:
        raise ValueError(f"dictionary/train: `derivation` is one of {', '.join(DERIVATIONS + LATER_DERIVATIONS)}, "
                         f"not {derivation!r}")
    if params.get("against") is not None and derivation != "crosscoder":
        raise ValueError("dictionary/train: `against` names a crosscoder's second model; a "
                         f"{derivation!r} reads one model")
    activation = params.get("activation") or "batchtopk"
    if activation not in TRAINERS:
        raise ValueError(f"dictionary/train: `activation` is one of {', '.join(TRAINERS)}, not {activation!r}")
    if params.get("layer") is None:
        raise ValueError("dictionary/train: `layer` names the layer the dictionary reads")
    layer = read_int("layer", params.get("layer"), None, 0)
    width = read_int("width", params.get("width"), 16384, 1)
    k = read_int("k", params.get("k"), 64, 1)
    if k > width:
        raise ValueError(f"dictionary/train: `k` ({k}) is more features than the dictionary has ({width})")
    l0 = params.get("l0") if params.get("l0") is not None else 64
    if not isinstance(l0, (int, float)) or isinstance(l0, bool) or not 0 < l0 <= width:
        raise ValueError(f"dictionary/train: `l0` is the number of features a JumpReLU dictionary aims to "
                         f"fire per token, above 0 and at most `width`, not {l0!r}")
    context = read_int("context", params.get("context"), 1024, 2)
    batch_tokens = read_int("batch_tokens", params.get("batch_tokens"), 4096, 1)
    buffer_tokens = read_int("buffer_tokens", params.get("buffer_tokens"), 262144, 1)
    if buffer_tokens < batch_tokens:
        raise ValueError(f"dictionary/train: `buffer_tokens` ({buffer_tokens}) holds fewer activations than "
                         f"one batch (`batch_tokens`, {batch_tokens})")
    tokens = read_int("tokens", params.get("tokens"), 200_000_000, 1)
    lr = params.get("lr") if params.get("lr") is not None else 2e-4
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not lr > 0:
        raise ValueError(f"dictionary/train: `lr` is a positive number, not {lr!r}")
    seed = read_int("seed", params.get("seed"), 0, 0)
    checkpoint_tokens = read_int("checkpoint_tokens", params.get("checkpoint_tokens"), 10_000_000, 1)
    heldout_tokens = read_int("heldout_tokens", params.get("heldout_tokens"), 262144, 1)
    steps = -(-tokens // batch_tokens)
    steps_per_chunk = buffer_tokens // batch_tokens
    warmup = min(MAX_WARMUP_STEPS, int(steps * WARMUP_SHARE))
    return Settings(
        layer=layer, point=str(params.get("point") or "resid_post"), width=width,
        activation=activation, k=k, l0=float(l0), tokens=tokens, context=context,
        batch_tokens=batch_tokens, buffer_tokens=steps_per_chunk * batch_tokens, lr=float(lr), seed=seed,
        checkpoint_tokens=checkpoint_tokens, heldout_tokens=heldout_tokens, corpus=params.get("corpus"),
        steps=steps, steps_per_chunk=steps_per_chunk, warmup_steps=warmup,
        decay_steps=int(steps * DECAY_SHARE),
        dead_tokens=min(DEAD_TOKENS, max(batch_tokens, tokens // 10)),
        rows=max(1, batch_tokens // context),
        heldout_windows=-(-heldout_tokens // (context - 1)),
        curve_every=max(1, -(-steps // CURVE_POINTS)),
        derivation=derivation,
    )
