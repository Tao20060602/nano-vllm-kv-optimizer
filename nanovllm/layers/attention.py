import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.utils.trace import get_tracer


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Logical layer id assigned by ModelRunner when KV caches are wired up.
        self.layer_id = None
        # M11: opt-in per-layer sparse CPU-KV runtime (None when feature-off).
        self.sparse_rt = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        rt = getattr(self, "sparse_rt", None)

        # ---- M11 opt-in sparse path -------------------------------------
        if rt is not None and rt.enabled:
            assert q.shape[0] == (context.max_seqlen_q or q.shape[0]) or context.is_prefill
            if context.is_prefill:
                if getattr(rt, "prefill_chunk", None) is not None and hasattr(rt, "prefill_first"):
                    # M12 chunked prefill:
                    #  - first chunk: dense within-chunk FA2 (no history yet)
                    #  - later chunks: fused exact attention over
                    #    [sink | selected historical blocks | chunk(causal)]
                    if rt.prefill_len == 0:
                        o = flash_attn_varlen_func(q, k, v,
                                                   max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                                   max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                                   softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                        rt.prefill_first(q, k, v)
                        return o
                    return rt.prefill_chunk(q, k, v, q.device)
                # dense prefill output (must remain dense + fit on GPU)
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                # per-layer CPU history init + frozen index build
                rt.prefill(q, k, v)
                return o
            # sparse decode: exactly one query token, no paged cache, no FA
            assert q.shape[0] == 1, "sparse decode supports a single query token"
            assert context.block_tables is None, "sparse decode must not use paged block tables"
            assert self.k_cache.numel() == 0, "sparse mode must allocate zero paged blocks"
            return rt.decode(q, k, v, q.device)

        # ---- feature-off: original dense paged FlashAttention path -------
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            # Opt-in, one-shot real-model trace (cold single-seq prefill only).
            get_tracer().maybe_capture(self.layer_id, q, k, v, o, context)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
