"""Block-sparse attention laboratory (M8) and CPU offload laboratory (M9).

Standalone exact Block-DIPR prototype plus synchronous single-layer CPU KV
offload.  Importing this package only requires PyTorch; it must not pull in
CUDA-only engine dependencies.
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
    route_a_replay,
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
]
