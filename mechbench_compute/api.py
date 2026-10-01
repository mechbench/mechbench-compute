from __future__ import annotations

import importlib
from typing import Any

SOURCES: dict[str, tuple[str, str | None]] = {
    "Context": ("mechbench_compute.ops", "Context"),
    "NeedNotDeclared": ("mechbench_compute.ops", "NeedNotDeclared"),
    "ProviderClient": ("mechbench_compute.providers.provider_client", "ProviderClient"),
    "Model": ("mechbench_compute.model", "Model"),
    "HookInfo": ("mechbench_compute.hooks", "HookInfo"),
    "Capture": ("mechbench_compute.interventions", "Capture"),
    "Patch": ("mechbench_compute.interventions", "Patch"),
    "Ablate": ("mechbench_compute.interventions", "Ablate"),
    "compose": ("mechbench_compute.interventions", "compose"),
    "plan_intervention": ("mechbench_compute.intervene.plan", "plan"),
    "compile_intervention": ("mechbench_compute.intervene.compile", "compile"),
    "edit_weights": ("mechbench_compute.intervene.edit_weights", "edit_weights"),
    "sample_completion_cached": ("mechbench_compute.generate", "sample_completion_cached"),
    "render": ("mechbench_compute.distill", "render"),
    "encode": ("mechbench_compute.distill", "encode"),
    "prefill_decision": ("mechbench_compute.distill", "prefill_decision"),
    "TokenReadout": ("mechbench_compute.token_readout", "TokenReadout"),
    "points": ("mechbench_compute.points", None),
    "collection": ("mechbench_compute.lexicon.kinds", "collection"),
    "items_of": ("mechbench_compute.lexicon.kinds", "items_of"),
    "item_kind_of": ("mechbench_compute.lexicon.kinds", "item_kind_of"),
    "read_header": ("mechbench_compute.blocks.read_header", "read_header"),
    "ShardWriter": ("mechbench_compute.tensors", "ShardWriter"),
    "tensor_collection": ("mechbench_compute.tensors", "collection"),
    "read_dictionary_weights": ("mechbench_compute.dictionaries.read_dictionary_weights",
                                "read_dictionary_weights"),
    "read_dictionary_activations": ("mechbench_compute.dictionaries.read_dictionary_activations",
                                    "read_dictionary_activations"),
    "encode_features": ("mechbench_compute.dictionaries.encode_features", "encode_features"),
    "arch_header": ("mechbench_compute.blocks", "arch_header"),
    "flatten_record": ("mechbench_compute.blocks.flatten_record", "flatten_record"),
    "index_by_key": ("mechbench_compute.blocks.index_by_key", "index_by_key"),
    "is_field_match": ("mechbench_compute.blocks.is_field_match", "is_field_match"),
    "MOVING_FIELDS": ("mechbench_compute.blocks.moving_fields", "MOVING_FIELDS"),
    "read_numbers": ("mechbench_compute.calibration.read_numbers", "read_numbers"),
    "compute_version": ("mechbench_compute", "__version__"),
    "Op": ("mechbench_compute.lexicon._base", "Op"),
    "P": ("mechbench_compute.lexicon._base", "P"),
    "In": ("mechbench_compute.lexicon._base", "In"),
    "Output": ("mechbench_compute.lexicon._base", "Output"),
    "Otherwise": ("mechbench_compute.lexicon._base", "Otherwise"),
    "DEFAULT_OUTPUT": ("mechbench_compute.lexicon._base", "DEFAULT_OUTPUT"),
    "Resume": ("mechbench_compute.lexicon._base", "Resume"),
    "Kind": ("mechbench_compute.lexicon._base", "Kind"),
    "Draw": ("mechbench_compute.lexicon._base", "Draw"),
    "Metric": ("mechbench_compute.lexicon._base", "Metric"),
    "Extension": ("mechbench_compute.lexicon.extension", "Extension"),
    "Package": ("mechbench_compute.lexicon.extension", "Package"),
    "F": ("mechbench_compute.lexicon.values", "F"),
    "read_body_level": ("mechbench_compute.resume", "read_body_level"),
    "shapes": ("mechbench_compute.shapes", None),
    "find_architecture": ("mechbench_compute.architectures", "for_type"),
    "read_last_logp": ("mechbench_compute.interp.read_last_logp", "read_last_logp"),
    "read_metric": ("mechbench_compute.interp.read_metric", "read_metric"),
    "METRICS": ("mechbench_compute.interp.read_metric", "METRICS"),
    "METRIC_DOC": ("mechbench_compute.interp.read_metric", "METRIC_DOC"),
    "OUTCOME_METRICS": ("mechbench_compute.interp.read_metric", "OUTCOME_METRICS"),
    "resolve_outcomes": ("mechbench_compute.interp.resolve_outcomes", "resolve_outcomes"),
    "resolve_target": ("mechbench_compute.interp.resolve_target", "resolve_target"),
    "report_own_top1": ("mechbench_compute.interp.report_own_top1", "report_own_top1"),
    "SpecIntervention": ("mechbench_compute.intervene.spec_intervention", "SpecIntervention"),
    "name_component": ("mechbench_compute.intervene.name_component", "name_component"),
    "estimate_bootstrap_ratio": ("mechbench_compute.blocks.estimate_bootstrap_ratio",
                                 "estimate_bootstrap_ratio"),
    "read_model_level": ("mechbench_compute.resume", "read_model_level"),
    "describe_spend_cap": ("mechbench_compute.blocks.describe_spend_cap", "describe_spend_cap"),
    "derive_seed": ("mechbench_compute.seeds", "derive"),
    "serialize_model": ("mechbench_compute.protocol.serialize_model", "serialize_model"),
    "run_try": ("mechbench_compute.live.run_try", "run_try"),
    "TryRefused": ("mechbench_compute.live.run_try", "TryRefused"),
}

__all__ = sorted(SOURCES)


def __getattr__(name: str) -> Any:
    source = SOURCES.get(name)
    if source is None:
        raise AttributeError(f"mechbench_compute.api has no {name!r}; it offers {', '.join(__all__)}")
    module, attr = source
    value = importlib.import_module(module)
    if attr is not None:
        value = getattr(value, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)
