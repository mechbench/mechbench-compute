from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.tools import toolbox_from


def _new_box(image: Any, specs: Sequence[Any], *, block_runner):
    """A fresh toolbox and its session (or None). One per item, so the
    snapshot chain and its provenance belong to that item alone."""
    session = None
    if image is not None:
        from mechbench_compute.sandbox_session import SandboxSession

        session = SandboxSession(image)
    return (toolbox_from(specs, block_runner=block_runner, session=session),
            session)
