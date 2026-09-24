from .backends import (
    BACKENDS,
    Backend,
    active as active_backend,
    available as available_backends,
    describe_platform,
    require as _require_backend,
)

if active_backend() is not None:
    from ._arch import (
        Arch,
        D_MODEL,
        GLOBAL_LAYERS,
        LAYER_HOOK_POINTS,
        N_HEADS,
        N_LAYERS,
        VOCAB_SIZE,
        all_hook_names,
        layer_type,
    )
    from .cache import ActivationCache
    from .errors import (
        CacheKeyError,
        InterpError,
        InvalidHookName,
        LayerIndexOutOfRange,
    )
    from .geometry import (
        VocabConcentration,
        centroid_decode,
        cluster_purity,
        cohesion,
        cosine_matrix,
        effective_vocab_size,
        entropy_bits,
        fact_vectors,
        fact_vectors_at,
        fact_vectors_at_hook,
        fact_vectors_pooled,
        intra_inter_separation,
        iterate_clusters,
        nearest_neighbor_purity,
        orthogonalize_against,
        silhouette_cosine,
        top_k_mass,
        vocab_concentration,
    )
    from .generate import generate_labeled_corpus, generate_text
    from .head_weights import (
        CircuitAnalysis,
        CircuitComponent,
        HeadSpec,
        PositionWrite,
        get_head_spec,
        head_key_tokens,
        head_ov_actual_writes,
        head_ov_position_writes,
        head_read_tokens,
        ov_circuit,
        qk_circuit,
    )
    from .attribution import (
        accumulated_resid,
        decompose_resid,
        head_results,
        logit_attrs,
    )
    from .probes import Probe
    from .hooks import HookFn, HookInfo, parse_hook_name
    from .interventions import Ablate, Capture, Intervention, Patch, compose
    from .lens import logit_lens_final, logit_lens_per_position
    from .model import Model, RunResult
    from .plot import (
        bar_by_layer,
        grouped_row_heatmap,
        head_heatmap,
        intensity_curve,
        leaderboard_bar,
        lens_trajectory,
        logprob_trajectory,
        pca_scatter,
        position_heatmap,
        probe_diagonal_heatmap,
        similarity_heatmap,
    )
    from .prompts import (
        Prompt,
        PromptSet,
        ValidatedPrompt,
        ValidatedPromptSet,
    )
    from .distill import (
        Example,
        TargetMap,
        TargetTrie,
        first_token_metrics,
        item_metrics,
        render_chat,
        score_items,
        score_items_batched,
        score_items_cached,
        score_items_fast,
        soft_ce,
    )
    from .lora import (
        LoRALinear,
        apply_lora,
        fuse,
        load_adapter,
        restore,
        save_adapter,
    )

else:

    def __getattr__(name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)

        import importlib

        try:
            return importlib.import_module(f".{name}", __name__)
        except ImportError:
            pass

        _require_backend()
        raise AttributeError(name)


def _editable_source_digest() -> str | None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parent
    if {"site-packages", "dist-packages"} & set(root.parts):
        return None
    return _digest_tree(root)


def _digest_tree(root) -> str:
    import hashlib

    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"?")
    return h.hexdigest()[:12]


def _source_version() -> str | None:
    import pathlib
    import re

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("mechbench-compute")
    _src = _editable_source_digest()
    if _src:
        __version__ = f"{_source_version() or __version__}+src.{_src}"
    del _src
except Exception:  # noqa: BLE001
    __version__ = "0.0.0+unknown"

__all__ = [
    "Backend",
    "BACKENDS",
    "active_backend",
    "available_backends",
    "describe_platform",
    "Model",
    "RunResult",
    "ActivationCache",
    "Ablate",
    "Capture",
    "Patch",
    "Intervention",
    "compose",
    "Prompt",
    "PromptSet",
    "ValidatedPrompt",
    "ValidatedPromptSet",
    "TargetMap",
    "TargetTrie",
    "Example",
    "soft_ce",
    "render_chat",
    "score_items",
    "score_items_batched",
    "score_items_cached",
    "score_items_fast",
    "item_metrics",
    "first_token_metrics",
    "LoRALinear",
    "apply_lora",
    "save_adapter",
    "load_adapter",
    "fuse",
    "restore",
    "logit_lens_final",
    "logit_lens_per_position",
    "accumulated_resid",
    "decompose_resid",
    "head_results",
    "logit_attrs",
    "fact_vectors",
    "fact_vectors_at",
    "fact_vectors_at_hook",
    "fact_vectors_pooled",
    "centroid_decode",
    "cohesion",
    "cosine_matrix",
    "intra_inter_separation",
    "iterate_clusters",
    "cluster_purity",
    "silhouette_cosine",
    "nearest_neighbor_purity",
    "orthogonalize_against",
    "Probe",
    "generate_text",
    "generate_labeled_corpus",
    "HeadSpec",
    "CircuitComponent",
    "CircuitAnalysis",
    "PositionWrite",
    "get_head_spec",
    "head_read_tokens",
    "head_key_tokens",
    "qk_circuit",
    "ov_circuit",
    "head_ov_position_writes",
    "head_ov_actual_writes",
    "VocabConcentration",
    "vocab_concentration",
    "top_k_mass",
    "entropy_bits",
    "effective_vocab_size",
    "bar_by_layer",
    "lens_trajectory",
    "logprob_trajectory",
    "position_heatmap",
    "pca_scatter",
    "similarity_heatmap",
    "head_heatmap",
    "probe_diagonal_heatmap",
    "grouped_row_heatmap",
    "intensity_curve",
    "leaderboard_bar",
    "HookInfo",
    "HookFn",
    "parse_hook_name",
    "Arch",
    "N_LAYERS",
    "D_MODEL",
    "N_HEADS",
    "VOCAB_SIZE",
    "GLOBAL_LAYERS",
    "LAYER_HOOK_POINTS",
    "layer_type",
    "all_hook_names",
    "InterpError",
    "InvalidHookName",
    "LayerIndexOutOfRange",
    "CacheKeyError",
]
