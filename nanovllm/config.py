import os
from dataclasses import dataclass, field
from transformers import AutoConfig

from nanovllm.kvdb.fingerprint import build_cache_fingerprint
from nanovllm.kvdb.types import CacheFingerprint


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_cache_metrics: bool = False
    enable_reusable_cache: bool = False
    enable_cpu_cache: bool = False
    cpu_cache_capacity_bytes: int = 256 * 1024 * 1024
    cpu_cache_pinned: bool = False
    # -- M11 opt-in sparse block retrieval + CPU KV offload ---------------
    enable_sparse_attention: bool = False
    sparse_selector: str = "query_guided"
    sparse_retrieval_block_size: int = 64
    sparse_recent_tokens: int = 128
    sparse_first_tokens: int = 0
    sparse_top_k: int = 8
    sparse_beta_raw: float = 48.0
    sparse_num_representatives: int = 4
    sparse_graph_degree: int = 16
    sparse_graph_l0: int = 16
    sparse_graph_max_scored_blocks: int = 32
    sparse_graph_projection_topk: int = 8
    sparse_query_samples: int = 128
    enable_sparse_diagnostics: bool = False
    cache_fingerprint: CacheFingerprint | None = field(default=None, init=False, repr=False)

    _SPARSE_SELECTORS = (
        "full", "exact_dipr", "top_k", "mean", "real", "knn_graph", "query_guided",
    )

    def _validate_sparse(self):
        assert self.sparse_selector in self._SPARSE_SELECTORS, (
            f"invalid sparse_selector {self.sparse_selector!r}; "
            f"choose from {self._SPARSE_SELECTORS}")
        assert self.sparse_retrieval_block_size > 0
        assert self.sparse_recent_tokens >= 0
        assert self.sparse_first_tokens >= 0
        assert self.sparse_top_k >= 0
        assert self.sparse_beta_raw >= 0
        assert self.sparse_num_representatives > 0
        assert self.sparse_graph_degree > 0
        assert self.sparse_graph_l0 > 0
        assert self.sparse_graph_max_scored_blocks > 0
        assert self.sparse_graph_projection_topk > 0
        assert self.sparse_query_samples > 0
        assert self.enforce_eager, "sparse mode requires enforce_eager=True"
        assert self.tensor_parallel_size == 1, "sparse mode requires TP=1"
        assert self.max_num_seqs == 1, "sparse mode requires max_num_seqs=1"
        assert not self.enable_reusable_cache, \
            "sparse mode is not composable with the reusable prefix cache"
        assert not self.enable_cpu_cache, \
            "sparse mode is not composable with the M0-M7 CPU prefix cache"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.enable_sparse_attention:
            self._validate_sparse()
        if self.enable_cpu_cache:
            if self.tensor_parallel_size != 1:
                raise ValueError("CPU cache currently requires tensor_parallel_size=1")
            if not self.enforce_eager:
                raise ValueError("CPU cache currently requires enforce_eager=True")
            self.enable_reusable_cache = True
            self.max_num_seqs = 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.enable_reusable_cache:
            self.cache_fingerprint = build_cache_fingerprint(
                self.model,
                self.hf_config,
                block_size=self.kvcache_block_size,
                tensor_parallel_size=self.tensor_parallel_size,
            )
