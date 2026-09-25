"""M16 decode budget decisions; prefill keeps its fixed candidate budget."""

import pytest
import torch

from nanovllm.sparse.m12_runtime import (
    M12Config,
    M12LayerRuntime,
    dynamic_top_k_from_scores,
)


def test_dynamic_top_k_keeps_ambiguous_and_short_histories():
    assert dynamic_top_k_from_scores(torch.ones(32), 32, 0.90) == 32
    assert dynamic_top_k_from_scores(torch.arange(10, 0, -1).float(), 32, 0.90) == 10
    assert dynamic_top_k_from_scores(
        torch.tensor([float("inf")] + [0.0] * 31), 32, 0.90) == 32


def test_dynamic_top_k_reduces_concentrated_scores():
    scores = torch.tensor([10.0] + [0.0] * 31)
    assert dynamic_top_k_from_scores(scores, 32, 0.90) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU selector")
def test_dynamic_selector_changes_decode_budget_only():
    cfg = M12Config(
        block_size=2, r=1, recent_tokens=2, sink_tokens=2,
        top_k_blocks=32, dynamic_top_k=True, max_model_len=128,
        num_heads=1, num_kv_heads=1, head_dim=4, dtype=torch.float32,
    )
    rt = M12LayerRuntime(0, cfg)
    rt.nblocks_filled = 40
    rt.reps_gpu.zero_()
    rt.reps_gpu[0, 0, 0, 0] = 10.0
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")

    prefill_ids, _ = rt._gpu_select(query)
    decode_ids, info = rt._gpu_select(query, dynamic=True)

    assert prefill_ids.numel() == 32
    assert decode_ids.numel() == info["n_selected"] == 16
    assert torch.equal(decode_ids, prefill_ids[:16])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU selector")
def test_fixed_decode_override_keeps_prefill_budget():
    cfg = M12Config(
        block_size=2, r=1, recent_tokens=2, sink_tokens=2,
        top_k_blocks=32, decode_top_k_blocks=24, max_model_len=128,
        num_heads=1, num_kv_heads=1, head_dim=4, dtype=torch.float32,
    )
    rt = M12LayerRuntime(0, cfg)
    rt.nblocks_filled = 40
    rt.reps_gpu.zero_()
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")

    prefill_ids, _ = rt._gpu_select(query)
    decode_ids, info = rt._gpu_select(query, budget=cfg.decode_top_k_blocks)

    assert prefill_ids.numel() == 32
    assert decode_ids.numel() == info["n_selected"] == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU KV storage")
def test_m12_reset_reuses_buffers_without_inheriting_history():
    cfg = M12Config(
        block_size=2, r=1, recent_tokens=2, sink_tokens=2,
        top_k_blocks=2, max_model_len=32,
        num_heads=1, num_kv_heads=1, head_dim=4, dtype=torch.float32,
    )
    rt = M12LayerRuntime(0, cfg)
    kv = torch.arange(16, device="cuda", dtype=torch.float32).view(4, 1, 4)
    rt.prefill_first(kv, kv, kv)
    original_cpu_buffer = rt.k_cpu
    assert rt.valid_len == rt.prefill_len == 4
    assert rt.nblocks_filled == 2

    rt.reset()
    assert rt.k_cpu is original_cpu_buffer
    assert rt.valid_len == rt.prefill_len == rt.nblocks_filled == 0
    assert not rt.protected_blocks
    rt.prefill_first(kv, kv, kv)
    assert rt.valid_len == rt.prefill_len == 4
    assert rt.nblocks_filled == 2
