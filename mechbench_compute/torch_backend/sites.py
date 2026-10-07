from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Read = Callable[[], Any]
Write = Callable[[Any], None]


@dataclass(frozen=True)
class Site:
    read: Read
    write: Write
    settle: Callable[[], None] | None = None


def read_child_output(child: str, layer: Any) -> Site:
    module = getattr(layer, child)
    return Site(lambda: module.output, lambda value: setattr(module, "output", value))


def read_child_first_output(child: str, layer: Any) -> Site:
    module = getattr(layer, child)

    def write(value: Any) -> None:
        held = module.output
        module.output = (value, *held[1:])

    return Site(lambda: module.output[0], write)


@dataclass(frozen=True)
class Sites:
    text_path: Callable[[Any], tuple[str, ...]]
    attn_out: Callable[[Any], Site]
    mlp_out: Callable[[Any], Site]
    gate_out: Callable[[Any], Site] | None = None
    qkv: Callable[..., list[Any]] | None = None

    def read_text(self, root: Any, module: Any) -> Any:
        envoy = root
        for name in self.text_path(module):
            envoy = getattr(envoy, name)
        return envoy
