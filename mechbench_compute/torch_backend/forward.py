from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mechbench_compute._arch import Arch
from mechbench_compute.cache import ActivationCache
from mechbench_compute.hooks import HookFn, dispatch, parse_hook_name
from mechbench_compute.torch_backend.decoding import (
    read_cache_offset,
    read_cached_kwargs,
)
from mechbench_compute.torch_backend.sites import Site, Sites
from mechbench_compute.torch_backend.tracing import disarm, use_attention, wrap_model

EAGER_POINTS = frozenset({"attn.weights"})

INTERFACE_CALL = re.compile(r"^\s*(attention_interface_(\d+))\s+->", re.MULTILINE)

INTERFACE_NAMES: dict[type, str] = {}

GRADIENTS: ContextVar[bool] = ContextVar("gradients", default=False)


@contextmanager
def tracking_gradients() -> Iterator[None]:
    token = GRADIENTS.set(True)
    try:
        yield
    finally:
        GRADIENTS.reset(token)


def read_grad_mode() -> Any:
    import torch

    return torch.enable_grad() if GRADIENTS.get() else torch.no_grad()


def find_interface_call(attn: Any) -> Any:
    kind = type(attn._module)
    name = INTERFACE_NAMES.get(kind)
    if name is None:
        found = INTERFACE_CALL.findall(str(attn.source))
        if not found:
            raise NotImplementedError(
                f"{kind.__name__} calls no `attention_interface`: the torch backend reads "
                f"attention weights and per-head outputs at that call")
        name = INTERFACE_NAMES[kind] = max(found, key=lambda m: int(m[1]))[0]
    return getattr(attn.source, name)


def read_attention(module: Any, names: Iterable[str], arch: Arch) -> str:
    if any(parse_hook_name(n, arch=arch).point in EAGER_POINTS for n in names):
        return "eager"
    return str(getattr(module.config, "_attn_implementation", None) or "eager")


@dataclass(frozen=True)
class Step:
    names: tuple[str, ...]
    layer: int | None
    points: tuple[str, ...]
    site: Site
    unpack: Callable[[Any], tuple[Any, ...]]
    pack: Callable[[Any, tuple[Any, ...]], Any]

    def run(self, hooks: dict[str, HookFn], capture_set: set[str],
            cache: ActivationCache) -> Any:
        held = self.site.read()
        parts = self.unpack(held)
        out = tuple(dispatch(n, self.layer, p, a, hooks, capture_set, cache)
                    for n, p, a in zip(self.names, self.points, parts))
        if any(a is not b for a, b in zip(out, parts)):
            held = self.pack(held, out)
            self.site.write(held)
        return held


def read_one(value: Any) -> tuple[Any, ...]:
    return (value,)


def pack_one(held: Any, parts: tuple[Any, ...]) -> Any:
    return parts[0]


def read_heads(value: Any) -> tuple[Any, ...]:
    return (value[0].transpose(1, 2),)


def pack_heads(held: Any, parts: tuple[Any, ...]) -> Any:
    return (parts[0].transpose(1, 2), *held[1:])


def pack_pair(held: Any, parts: tuple[Any, ...]) -> Any:
    return (parts[0], parts[1], *held[2:])


def read_scale(norm: Any) -> Callable[[Any], tuple[Any, ...]]:
    eps = float(getattr(norm._module, "eps", getattr(norm._module, "variance_epsilon", 1e-6)))

    def unpack(value: Any) -> tuple[Any, ...]:
        import torch

        f32 = value.float()
        return (torch.sqrt(torch.mean(f32 * f32, dim=-1) + eps),)

    return unpack


def keep_held(held: Any, parts: tuple[Any, ...]) -> Any:
    return held


def make_step(name: str, layer: int | None, point: str, site: Site) -> Step:
    return Step((name,), layer, (point,), site, read_one, pack_one)


def plan_steps(envoy: Any, module: Any, sites: Sites, wanted: Iterable[str],
               arch: Arch) -> list[Step]:
    by_layer: dict[int, set[str]] = {}
    top: set[str] = set()
    for name in wanted:
        info = parse_hook_name(name, arch=arch)
        if info.layer is None:
            top.add(info.point)
        else:
            by_layer.setdefault(info.layer, set()).add(info.point)
    text = sites.read_text(envoy, module)
    steps: list[Step] = []
    if "embed" in top:
        embed = text.embed_tokens
        steps.append(make_step("embed", None, "embed", Site(
            lambda: embed.output, lambda v: setattr(embed, "output", v))))
    for i in range(arch.n_layers):
        points = by_layer.get(i)
        if not points:
            continue
        steps.extend(plan_layer(text.layers[i], i, points, sites))
    norm = text.norm
    if "final_norm.scale" in top:
        steps.append(Step(("final_norm.scale",), None, ("final_norm.scale",),
                          Site(lambda: norm.input, lambda v: None), read_scale(norm), keep_held))
    if "final_norm" in top:
        steps.append(make_step("final_norm", None, "final_norm", Site(
            lambda: norm.output, lambda v: setattr(norm, "output", v))))
    steps.append(make_step("logits", None, "logits", Site(
        lambda: envoy.output.logits, lambda v: setattr(envoy.output, "logits", v))))
    return steps


def make_source_step(op: Any, layer: int, point: str) -> Step:
    return make_step(f"blocks.{layer}.{point}", layer, point, Site(
        lambda: op.output, lambda value: setattr(op, "output", value)))


def plan_rope_qkv(attn: Any, i: int, points: set[str]) -> list[Step]:
    steps: list[Step] = []
    if "attn.v" in points:
        steps.append(make_source_step(attn.source.transpose_2, i, "attn.v"))
    if points & {"attn.q", "attn.k"}:
        rope = attn.source.apply_rotary_pos_emb_0
        named = f"blocks.{i}."
        steps.append(Step((named + "attn.q", named + "attn.k"), i, ("attn.q", "attn.k"), Site(
            lambda: rope.output, lambda value: setattr(rope, "output", value)),
            lambda held: (held[0], held[1]), pack_pair))
    return steps


def plan_layer(layer: Any, i: int, points: set[str], sites: Sites) -> list[Step]:
    steps: list[Step] = []
    named = f"blocks.{i}."
    if "resid_pre" in points:
        steps.append(make_step(named + "resid_pre", i, "resid_pre", Site(
            lambda: layer.input, lambda v: setattr(layer, "input", v))))
    attn = layer.self_attn
    steps.extend((sites.qkv or plan_rope_qkv)(attn, i, points))
    if "attn.weights" in points:
        def read_weights() -> Any:
            return find_interface_call(attn).source.nn_functional_dropout_0

        steps.append(make_step(named + "attn.weights", i, "attn.weights", Site(
            lambda: read_weights().output,
            lambda value: setattr(read_weights(), "output", value),
            lambda: disarm(find_interface_call(attn)))))
    if "attn.per_head_out" in points:
        inner = find_interface_call(attn)
        steps.append(Step((named + "attn.per_head_out",), i, ("attn.per_head_out",), Site(
            lambda: inner.output, lambda value: setattr(inner, "output", value)),
            read_heads, pack_heads))
    if "attn_out" in points:
        steps.append(make_step(named + "attn_out", i, "attn_out", sites.attn_out(layer)))
    if "mlp_out" in points:
        steps.append(make_step(named + "mlp_out", i, "mlp_out", sites.mlp_out(layer)))
    if "gate_out" in points:
        steps.append(make_step(named + "gate_out", i, "gate_out", sites.gate_out(layer)))
    if "resid_post" in points:
        steps.append(make_step(named + "resid_post", i, "resid_post", Site(
            lambda: layer.output, lambda v: setattr(layer, "output", v))))
    return steps


def run_steps(steps: list[Step], hooks: dict[str, HookFn], capture_set: set[str],
          cache: ActivationCache, held: dict[str, Any]) -> None:
    for step in steps:
        held["last"] = step.run(hooks, capture_set, cache)


def run_forward(model: Any, input_ids: Any, *, sites: Sites, hooks: dict[str, HookFn] | None = None,
                capture: list[str] | None = None, arch: Arch | None = None,
                kv_cache: Any = None) -> tuple[Any, ActivationCache]:

    hooks = dict(hooks or {})
    capture_set = set(capture or [])
    cache = ActivationCache(offset=read_cache_offset(kv_cache))
    cached = read_cached_kwargs(kv_cache)
    wanted = set(hooks) | capture_set
    if not wanted - {"logits"}:
        with read_grad_mode():
            logits = model(input_ids=input_ids, **cached).logits
        return dispatch("logits", None, "logits", logits, hooks, capture_set, cache), cache
    envoy = wrap_model(model)
    steps = plan_steps(envoy, model, sites, wanted, arch)
    eager = any(p in EAGER_POINTS for s in steps for p in s.points)
    held: dict[str, Any] = {}
    try:
        with use_attention(model, "eager" if eager else None), read_grad_mode(), \
                envoy.trace(input_ids=input_ids, **cached):
            run_steps(steps, hooks, capture_set, cache, held)
    finally:
        for step in steps:
            if step.site.settle is not None:
                step.site.settle()
    return held["last"], cache
