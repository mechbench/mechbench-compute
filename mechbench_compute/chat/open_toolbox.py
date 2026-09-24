from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mechbench_compute.tools import build_toolbox


def open_toolbox(image: Any, specs: Sequence[Any], *, block_runner):
    session = None
    if image is not None:
        from mechbench_compute.sandbox_session import SandboxSession

        session = SandboxSession(image)
    return (build_toolbox(specs, block_runner=block_runner, session=session),
            session)
