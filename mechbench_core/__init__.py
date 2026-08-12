"""mechbench_core — a mechanistic-interpretability framework for
Google's Gemma 4 E4B running locally on Apple Silicon via MLX.

Quick start:

    from mechbench_core import Model

    model = Model.load()
    ids = model.tokenize("Complete this sentence with one word: The Eiffel Tower is in")

    # Forward pass, no instrumentation:
    result = model.run(ids)
    for tok, p in result.top_k(model.tokenizer, k=5):
        print(f'{tok!r:20s} p={p:.4f}')

Capture activations:

    result = model.run(ids, capture=['blocks.23.attn.weights',
                                      'blocks.14.mlp_out'])
    weights = result.cache['blocks.23.attn.weights']  # [1, 8, S, S], bf16
    mlp = result.cache['blocks.14.mlp_out']           # [1, S, 2560], bf16

Modify activations with a hook:

    import mlx.core as mx

    def zero_layer_14_mlp(act, info):
        return mx.zeros_like(act)

    result = model.run(ids, hooks={'blocks.14.mlp_out': zero_layer_14_mlp})

Declarative interventions (Ablate / Capture / Patch), prompt tooling,
logit-lens + geometry helpers, and matplotlib plot helpers are all
re-exported from this module — see README.md for the full API tour.

The full list of hook points is at mechbench_core.all_hook_names().
"""

from ._arch import (
    Arch,
    DEFAULT_MODEL_ID,
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

__version__ = "0.7.0"

__all__ = [
    # Main API
    "Model",
    "RunResult",
    "ActivationCache",
    # Declarative interventions
    "Ablate",
    "Capture",
    "Patch",
    "Intervention",
    "compose",
    # Prompts (specific prompt collections live in experiments.prompts)
    "Prompt",
    "PromptSet",
    "ValidatedPrompt",
    "ValidatedPromptSet",
    # Distributional-target training (distill + lora, task 000114)
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
    # Logit lens
    "logit_lens_final",
    "logit_lens_per_position",
    # Direct logit attribution + residual decomposition
    "accumulated_resid",
    "decompose_resid",
    "head_results",
    "logit_attrs",
    # Fact vectors + geometry
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
    # Probes (persistent concept vectors)
    "Probe",
    # Generation (for corpus-building workflows)
    "generate_text",
    "generate_labeled_corpus",
    # Head-weight analysis (static interp on W_Q/W_K/W_V/W_O)
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
    # Vocabulary-space concentration
    "VocabConcentration",
    "vocab_concentration",
    "top_k_mass",
    "entropy_bits",
    "effective_vocab_size",
    # Plot helpers
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
    # Hook types (for users writing raw callbacks)
    "HookInfo",
    "HookFn",
    "parse_hook_name",
    # Architecture facts
    "Arch",
    "DEFAULT_MODEL_ID",
    "N_LAYERS",
    "D_MODEL",
    "N_HEADS",
    "VOCAB_SIZE",
    "GLOBAL_LAYERS",
    "LAYER_HOOK_POINTS",
    "layer_type",
    "all_hook_names",
    # Errors
    "InterpError",
    "InvalidHookName",
    "LayerIndexOutOfRange",
    "CacheKeyError",
]
