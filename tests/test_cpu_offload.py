"""Tests for the M9 synchronous CPU KV offload laboratory.

CPU/pageable paths run everywhere; pinned and CUDA replay paths run only when
CUDA is available.
"""

import pytest
import torch

from nanovllm.sparse.block_sparse import (
    dense_decode_attention,
    exact_block_scores,
    gqa_token_scores,
    select_dipr_blocks,
    select_topk_blocks,
    sparse_decode_attention,
    union_block_mask,
    selected_token_indices,
)
from nanovllm.sparse.cpu_offload import CPULayerKVStore, route_a_replay


def _kv(seed=0, t=64, hkv=2, hq=4, d=16, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(hq, d, generator=g, dtype=dtype)
    k = torch.randn(t, hkv, d, generator=g, dtype=dtype)
    v = torch.randn(t, hkv, d, generator=g, dtype=dtype)
    return q, k, v


# 1. CPU store preserves dtype, shape and values -------------------------------
def test_cpu_store_preserves_dtype_shape_values():
    q, k, v = _kv(dtype=torch.float32)
    store = CPULayerKVStore(k, v, pinned=False)
    assert store.k_cpu.shape == k.shape and store.v_cpu.shape == v.shape
    assert store.k_cpu.dtype == k.dtype
    assert torch.equal(store.k_cpu, k) and torch.equal(store.v_cpu, v)
    # store owns a copy: mutating input must not change the store
    k.fill_(99.0)
    assert not torch.all(store.k_cpu == 99.0)


def test_cpu_store_rejects_gpu_or_mismatched_tensors():
    q, k, v = _kv()
    with pytest.raises(ValueError):
        CPULayerKVStore(k, v[:, :, :8])          # shape mismatch
    with pytest.raises(ValueError):
        CPULayerKVStore(k, v.to(torch.float16))  # dtype mismatch


# 2. staging contains exactly the requested rows in sorted order ---------------
def test_gather_rows_in_sorted_order():
    q, k, v = _kv()
    store = CPULayerKVStore(k, v, pinned=False)
    idx = torch.tensor([3, 17, 31], dtype=torch.long)
    staging = store.gather(idx)
    assert torch.equal(staging.packed_k, k[idx])
    assert torch.equal(staging.packed_v, v[idx])
    assert staging.packed_k.is_contiguous() and staging.packed_v.is_contiguous()
    assert torch.equal(staging.indices, idx)
    # unsorted / duplicate / out-of-range indices are rejected
    with pytest.raises(ValueError):
        store.gather(torch.tensor([5, 2], dtype=torch.long))
    with pytest.raises(ValueError):
        store.gather(torch.tensor([5, 5], dtype=torch.long))
    with pytest.raises(IndexError):
        store.gather(torch.tensor([64], dtype=torch.long))


# 3. byte counters equal tensor storage sizes ----------------------------------
def test_byte_counters():
    q, k, v = _kv(t=64, hkv=2, d=16, dtype=torch.float32)
    store = CPULayerKVStore(k, v, pinned=False)
    per = 64 * 2 * 16 * 4
    assert store.resident_bytes == 2 * per
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    staging = store.gather(idx)
    assert staging.packed_k_bytes == 4 * 2 * 16 * 4
    assert staging.packed_v_bytes == 4 * 2 * 16 * 4
    assert staging.h2d_bytes == 2 * 4 * 2 * 16 * 4
    assert staging.h2d_bytes / store.resident_bytes == 4 / 64


# 4. packed attention equals M8 sparse attention for the same selection --------
def test_packed_attention_equals_m8_sparse():
    q, k, v = _kv(t=64)
    store = CPULayerKVStore(k, v, pinned=False)
    token_scores = gqa_token_scores(q.float(), k.float())
    bs = exact_block_scores(token_scores, 16)
    union = union_block_mask(select_topk_blocks(bs, 2))
    idx = selected_token_indices(union, 64, 16, recent_tokens=8)
    staging = store.gather(idx)
    packed_out = sparse_decode_attention(q, staging.packed_k, staging.packed_v,
                                         torch.arange(staging.num_selected_tokens))
    sparse_out = sparse_decode_attention(q, k, v, idx)
    assert torch.allclose(packed_out, sparse_out, atol=1e-6)


# 5. all-token packed attention equals dense attention -------------------------
def test_all_token_packed_equals_dense():
    q, k, v = _kv(t=64)
    store = CPULayerKVStore(k, v, pinned=False)
    idx = torch.arange(64)
    staging = store.gather(idx)
    packed_out = sparse_decode_attention(q, staging.packed_k, staging.packed_v,
                                         torch.arange(64))
    dense_out = dense_decode_attention(q, k, v)
    assert torch.allclose(packed_out, dense_out, atol=1e-5)


# 6. pageable Route-A replay works on CPU ---------------------------------------
def test_route_a_pageable_cpu():
    q, k, v = _kv(t=96)
    store = CPULayerKVStore(k, v, pinned=False)
    res = route_a_replay(
        store, q,
        lambda bs: select_dipr_blocks(bs, beta=2.0),
        device=torch.device("cpu"),
        retrieval_block_size=16, recent_tokens=16,
    )
    assert 0 < res.num_selected_tokens <= 96
    assert res.h2d_bytes == 2 * res.num_selected_tokens * 2 * 16 * 4
    assert abs(res.active_byte_ratio - res.num_selected_tokens / 96) < 1e-9
    # output on cpu, finite; dense oracle close on this small example
    dense = dense_decode_attention(q, k, v)
    assert torch.isfinite(res.output).all()
    assert res.output.shape == dense.shape
    for region in ("cpu_search_ms", "cpu_gather_ms", "h2d_ms",
                   "gpu_packed_attention_ms", "total_replay_ms"):
        assert region in res.timings_ms


# 7. pinned mode + CUDA replay when available ----------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_store_and_cuda_replay():
    torch.manual_seed(0)
    t, hq, hkv, d = 128, 4, 2, 32
    g = torch.Generator().manual_seed(1)
    q = torch.randn(hq, d, generator=g, dtype=torch.float16)
    k = torch.randn(t, hkv, d, generator=g, dtype=torch.float16)
    v = torch.randn(t, hkv, d, generator=g, dtype=torch.float16)
    store = CPULayerKVStore(k, v, pinned=True)
    assert store.k_cpu.is_pinned() and store.v_cpu.is_pinned()
    idx = torch.arange(0, t, 2)
    staging = store.gather(idx)
    assert staging.packed_k.is_pinned() and staging.packed_v.is_pinned()
    assert staging.h2d_bytes == 2 * idx.numel() * hkv * d * 2

    res = route_a_replay(
        store, q,
        lambda bs: select_topk_blocks(bs, 4),
        device=torch.device("cuda"),
        retrieval_block_size=16, recent_tokens=16,
    )
    assert res.output.device.type == "cuda"
    assert torch.isfinite(res.output.float()).all()
    assert res.pinned is True
    # full history must never be resident as a packed transfer
    assert res.h2d_bytes < res.full_kv_bytes

    # dense GPU oracle over full K/V (baseline only, not the sparse path)
    dense = dense_decode_attention(q.cuda(), k.cuda(), v.cuda())
    assert dense.shape == res.output.shape
