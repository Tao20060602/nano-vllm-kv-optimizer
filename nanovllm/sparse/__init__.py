"""Block-sparse attention laboratory (M8), CPU offload (M9) and approximate
block-retrieval laboratory (M10).

Standalone exact Block-DIPR prototype, synchronous single-layer CPU KV offload,
block representatives, K-to-K KNN / query-guided graphs and a simplified
Block-DIPRS traversal.  Importing this package only requires PyTorch; it must
not pull in CUDA-only engine dependencies.
"""

from nanovllm.sparse.block_sparse import (
    RetrievalBlockMap,
    attention_mass_recovery,
    critical_token_recall,
    dense_decode_attention,
    evaluate_sparse_result,
    exact_block_scores,
    gqa_token_scores,
    kv_head_for_query,
    select_dipr_blocks,
    select_topk_blocks,
    selected_token_indices,
    sparse_decode_attention,
    union_block_mask,
)
from nanovllm.sparse.cpu_offload import (
    CPULayerKVStore,
    PackedCPUStaging,
    RouteAResult,
    SelectorRouteResult,
    route_a_replay,
    route_a_selective,
)
from nanovllm.sparse.representatives import (
    BlockRepresentatives,
    FlatSelectResult,
    flat_mean_select,
    flat_real_select,
)
from nanovllm.sparse.graph_diprs import (
    BlockGraphIndex,
    DIPRSSearchResult,
    GraphBuildInfo,
    union_per_head,
)

__all__ = [
    "RetrievalBlockMap",
    "attention_mass_recovery",
    "critical_token_recall",
    "dense_decode_attention",
    "evaluate_sparse_result",
    "exact_block_scores",
    "gqa_token_scores",
    "kv_head_for_query",
    "select_dipr_blocks",
    "select_topk_blocks",
    "selected_token_indices",
    "sparse_decode_attention",
    "union_block_mask",
    "CPULayerKVStore",
    "PackedCPUStaging",
    "RouteAResult",
    "route_a_replay",
    "BlockRepresentatives",
    "FlatSelectResult",
    "flat_mean_select",
    "flat_real_select",
    "BlockGraphIndex",
    "DIPRSSearchResult",
    "GraphBuildInfo",
    "union_per_head",
]
