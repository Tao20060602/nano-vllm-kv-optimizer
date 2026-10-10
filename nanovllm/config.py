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
    # M12 runtime: sparse_top_k remains the allocation/max budget.
    # Optional prefill-only budget; None preserves the existing behavior.
    sparse_prefill_top_k: int | None = None
    # Optional decode-only fixed/max budget.
    sparse_decode_top_k: int | None = None
    # Decode-only adaptive budget; prefill uses its fixed budget independently.
    sparse_dynamic_top_k: bool = False
    # Decode-only relative-threshold selection (attention >= a * max).
    # None preserves the existing fixed top-k decode behavior.
    sparse_decode_relative_a: float | None = None
    # Hard cap on decode-selected blocks when the relative rule is enabled.
    sparse_decode_relative_max_blocks: int = 48
    # Union mode: keep fixed top-k AND threshold-passing blocks (capped).
    sparse_decode_relative_union: bool = False
    # Decode Q-window routing: average the last W decode queries for routing
    # only. 1 = off (use the current single query).
    sparse_decode_query_window: int = 1
    # Fused Triton block-scoring kernel for the flat sparse selector path.
    sparse_fused_selector: bool = False
    # Opt-in int8 quantization of the CPU history (K per-channel, V per-block).
    sparse_quant_history: bool = False
    # Fuse the dequantize step into one kernel (requires quant history).
    sparse_fused_dequant: bool = False
    # Sort selected block ids before the CPU gather (sequential source access).
    sparse_gather_sort: bool = False
    # Fraction of normalized top-k representative score weight to retain.
    # This is a routing heuristic, not measured attention mass.
    sparse_dynamic_top_k_mass: float = 0.90
    sparse_beta_raw: float = 48.0
    sparse_num_representatives: int = 4
    sparse_graph_degree: int = 16
    sparse_graph_l0: int = 16
    sparse_graph_max_scored_blocks: int = 32
    sparse_graph_projection_topk: int = 8
    sparse_query_samples: int = 128
    enable_sparse_diagnostics: bool = False
    # M12-MVP: use new GPU-representative runtime instead of M11 CPU selector
    use_m12_runtime: bool = False
    # M12 chunked prefill: max tokens per prefill step (0 = one-shot whole prompt)
    sparse_prefill_chunk_size: int = 0
    # M14: number of equal query segments used to route a later prefill chunk.
    # 1 preserves the original whole-chunk mean-query behavior.
    sparse_prefill_query_segments: int = 1
    # M14: later-chunk packed attention backend.  FlashAttention-2 keeps the
    # same selected blocks and one-softmax semantics as the torch reference.
    sparse_prefill_attention_backend: str = "flash"
    # Explicit independent operator checkout, required only for the experiment.
    sparse_operator_root: str | None = None
    # M18: direct gather into pinned staging is the measured M12 default.
    # Set False to reproduce the legacy advanced-indexing path.
    sparse_gather_index_select: bool = True
    # M20 experiment: enqueue K H2D before gathering V during decode only.
    sparse_decode_kv_pipeline: bool = False
    # M19: capture the decode GPU selector; CPU KV gather stays outside graphs.
    sparse_selector_cuda_graph: bool = False
    # M21: reuse a cached GPU protected-index tensor for eager decode selection.
    sparse_selector_static_mask: bool = False
    # Debug-only finite-output assertion; synchronizes the device per layer.
    sparse_check_finite_outputs: bool = False
    # -- M12-MVP: YaRN rope scaling override -------------------------------
    # When set, this dict is injected into hf_config.rope_scaling and the
    # max_model_len cap from hf_config.max_position_embeddings is bypassed.
    rope_scaling_override: dict | None = None
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
        if self.sparse_prefill_top_k is not None:
            assert 1 <= self.sparse_prefill_top_k <= self.sparse_top_k
            assert self.use_m12_runtime, "prefill top-k override requires the M12 runtime"
        if self.sparse_decode_top_k is not None:
            assert 1 <= self.sparse_decode_top_k <= self.sparse_top_k
            assert self.use_m12_runtime, "decode top-k override requires the M12 runtime"
        assert 0.0 < self.sparse_dynamic_top_k_mass <= 1.0
        if self.sparse_dynamic_top_k:
            assert self.use_m12_runtime, "dynamic top-k requires the M12 runtime"
            assert self.sparse_top_k > 0
        if self.sparse_decode_relative_a is not None:
            assert self.use_m12_runtime, "decode relative-a requires the M12 runtime"
            assert 0.0 < self.sparse_decode_relative_a < 1.0, (
                "sparse_decode_relative_a must be in (0, 1)")
            assert self.sparse_decode_relative_max_blocks >= 1
        if self.sparse_decode_query_window < 1:
            raise AssertionError("sparse_decode_query_window must be >= 1")
        if self.sparse_decode_query_window > 1:
            assert self.use_m12_runtime, "decode query window requires the M12 runtime"
        if self.sparse_selector_cuda_graph:
            assert self.use_m12_runtime, "selector CUDA graph requires the M12 runtime"
        if self.sparse_selector_static_mask:
            assert self.use_m12_runtime, "static selector mask requires the M12 runtime"
        if self.sparse_decode_kv_pipeline:
            assert self.use_m12_runtime, "KV pipeline requires the M12 runtime"
            assert self.sparse_gather_index_select, "KV pipeline requires index_select"
        assert self.sparse_beta_raw >= 0
        assert self.sparse_num_representatives > 0
        assert self.sparse_graph_degree > 0
        assert self.sparse_graph_l0 > 0
        assert self.sparse_graph_max_scored_blocks > 0
        assert self.sparse_graph_projection_topk > 0
        assert self.sparse_query_samples > 0
        assert self.sparse_prefill_query_segments > 0
        assert self.sparse_prefill_attention_backend in ("torch", "flash", "flash_reuse", "operator")
        if self.sparse_prefill_attention_backend in ("flash_reuse", "operator"):
            assert self.use_m12_runtime, "experimental prefill requires M12 runtime"
        if self.sparse_prefill_attention_backend == "operator":
            assert self.sparse_operator_root and os.path.isdir(self.sparse_operator_root), (
                "operator backend requires an explicit operator repository path")
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

        # Inject YaRN / rope_scaling override before model construction.
        if self.rope_scaling_override is not None:
            self.hf_config.rope_scaling = dict(self.rope_scaling_override)
            # Ensure rope_parameters dict is also populated for Transformers >= 5.x
            rt = self.rope_scaling_override.get("rope_type",
                                                self.rope_scaling_override.get("type", "yarn"))
            self.hf_config.rope_parameters = {
                "rope_type": rt,
                "rope_theta": self.hf_config.rope_parameters.get("rope_theta", 1000000)
                    if hasattr(self.hf_config, "rope_parameters") else 1000000,
                **self.rope_scaling_override,
            }
            # When YaRN is active, do NOT cap max_model_len to
            # max_position_embeddings (the whole point is to extend beyond it).
            # Only cap when no override is given.
            self.max_model_len = self.max_model_len
        else:
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        if self.enable_reusable_cache:
            self.cache_fingerprint = build_cache_fingerprint(
                self.model,
                self.hf_config,
                block_size=self.kvcache_block_size,
                tensor_parallel_size=self.tensor_parallel_size,
            )
