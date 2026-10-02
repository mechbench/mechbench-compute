from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterator
from typing import Any

ENVOYS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def load_nnsight() -> Any:
    import nnsight

    nnsight.CONFIG.APP.PYMOUNT = False
    if hasattr(object(), "save"):
        from nnsight._c.py_mount import unmount

        unmount("save")
    return nnsight


def wrap_model(module: Any) -> Any:
    envoy = ENVOYS.get(module)
    if envoy is None:
        envoy = ENVOYS[module] = load_nnsight().NNsight(module)
    return envoy


def set_attention(module: Any, name: str) -> None:
    setter = getattr(module, "set_attn_implementation", None)
    if setter is not None:
        setter(name)
    else:
        module.config._attn_implementation = name


@contextlib.contextmanager
def use_attention(module: Any, name: str | None) -> Iterator[None]:
    held = getattr(module.config, "_attn_implementation", None)
    if name is None or name == held or held is None:
        yield
        return
    set_attention(module, name)
    try:
        yield
    finally:
        set_attention(module, held)
