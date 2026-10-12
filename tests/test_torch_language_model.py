from __future__ import annotations

import pytest

pytest.importorskip("torch")

from mechbench_compute.torch_backend import lora
from tests.tiny_torch_models import KIT_MODELS, build_tiny_model


@pytest.mark.parametrize("name", [name for name, _ in KIT_MODELS])
def test_the_language_model_names_its_parameters(name):
    model = build_tiny_model(name)
    marks = lora.mark_weights(model.lm)
    assert any(key.startswith("model.") for key in marks)
    assert lora.read_changed(model.lm, marks) == []
    assert sum(1 for _ in model.lm.parameters()) >= len(marks)
