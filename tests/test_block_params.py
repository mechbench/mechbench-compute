"""The param declarations must not drift behind the code (000438).

Declaring params opt-in per block buys a loud failure when a protocol
asks for something a block cannot do. It also introduces the opposite
bug: a param the block DOES read, left out of the table, is refused
for no reason. This reads the source and asserts the table covers it.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from mechbench_compute.block_params import ACCEPTED, COMMON, check_params

#: block ref -> the module that implements it.
SOURCES = {
    "~canonical/ops/chat/1": "chat.py",
    "~canonical/ops/vectors/mst/1": "trees.py",
    "~canonical/ops/residuals/vectors/1": "interp.py",
    "~canonical/ops/trajectory/capture/1": "trajectory.py",
    "~canonical/ops/trajectory/project/1": "trajectory.py",
    "~canonical/ops/trajectory/compare/1": "trajectory.py",
    "~canonical/ops/trajectory/aggregate/1": "trajectory.py",
    "~canonical/ops/tokenize/stats/1": "tokenizer_stats.py",
}
ROOT = pathlib.Path(__file__).resolve().parent.parent / "mechbench_compute"

#: A module that implements many blocks: only the params read from the
#: named function (to the next top-level `def`) belong to that block.
SCOPED = {
    "~canonical/ops/residuals/vectors/1": "def residual_vectors(",
    "~canonical/ops/trajectory/capture/1": "def capture(",
    "~canonical/ops/trajectory/project/1": "def project(",
    "~canonical/ops/trajectory/compare/1": "def compare(",
    "~canonical/ops/trajectory/aggregate/1": "def aggregate(",
}


def _params_read(path: pathlib.Path, start_at: str | None = None) -> set[str]:
    text = path.read_text()
    if start_at:
        i = text.index(start_at)
        j = text.find("\ndef ", i + 1)
        text = text[i:j if j != -1 else len(text)]
    return (set(re.findall(r'params\.get\(\s*["\']([^"\']+)["\']', text))
            | set(re.findall(r'params\[\s*["\']([^"\']+)["\']\s*\]', text)))


@pytest.mark.parametrize("ref,filename", sorted(SOURCES.items()))
def test_every_param_the_block_reads_is_declared(ref, filename):
    read = _params_read(ROOT / filename, SCOPED.get(ref))
    declared = ACCEPTED[ref] | COMMON
    missing = sorted(k for k in read - declared if not k.startswith("_"))
    assert not missing, (
        f"{ref} reads {missing} but does not declare them — "
        f"a protocol using one would be refused for no reason")


def test_every_declared_block_names_its_source():
    assert set(ACCEPTED) == set(SOURCES), (
        "a block gained a declaration without a source to check it against")


def test_an_executor_injection_is_not_refused():
    # `_block_runner` and friends are added by the executor, never
    # declared by a protocol.
    check_params("~canonical/ops/chat/1", {"n": 1, "_block_runner": object()})
