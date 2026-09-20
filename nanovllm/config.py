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
    cache_fingerprint: CacheFingerprint | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
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
