"""Prefill budget is independent of the fixed decode budget."""

import pytest
import torch

from nanovllm.sparse.m12_runtime import M12Config, M12LayerRuntime


@pytest.mark.parametrize("budget", [0, 33])
def test_prefill_budget_cannot_exceed_allocated_maximum(budget):
    with pytest.raises(ValueError, match="prefill_top_k_blocks"):
        M12Config(top_k_blocks=32, prefill_top_k_blocks=budget)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU runtime")
@pytest.mark.parametrize("prefill_budget,expected", [(None, 32), (24, 24), (16, 16)])
def test_prefill_budget_changes_only_later_chunk_selection(prefill_budget, expected):
    cfg = M12Config(
        block_size=2, r=1, recent_tokens=2, sink_tokens=2,
        top_k_blocks=32, prefill_top_k_blocks=prefill_budget,
        decode_top_k_blocks=32, max_model_len=128,
        num_heads=1, num_kv_heads=1, head_dim=4, dtype=torch.float32,
        scale=0.5, prefill_attention_backend="torch",
    )
    rt = M12LayerRuntime(0, cfg)
    history = torch.randn(80, 1, 4, device="cuda")
    rt.prefill_first(history, history, history)

    next_chunk = torch.randn(2, 1, 4, device="cuda")
    output = rt.prefill_chunk(next_chunk, next_chunk, next_chunk, history.device)
    decode_output = rt.decode(
        next_chunk[:1], next_chunk[:1], next_chunk[:1], history.device)

    assert torch.isfinite(output).all()
    assert torch.isfinite(decode_output).all()
    assert rt.last_prefill_block_ids.numel() == expected
    assert rt.last_block_ids.numel() == 32
    assert rt.valid_len == 83


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU runtime")
def test_prefill_index_select_matches_legacy_gather():
    torch.manual_seed(17)
    history = torch.randn(20, 1, 4, device="cuda")
    next_chunk = torch.randn(4, 1, 4, device="cuda")
    results = []
    for use_index_select in (False, True):
        cfg = M12Config(
            block_size=2, r=1, recent_tokens=2, sink_tokens=2,
            top_k_blocks=4, prefill_top_k_blocks=4,
            decode_top_k_blocks=4, max_model_len=32,
            num_heads=1, num_kv_heads=1, head_dim=4, dtype=torch.float32,
            scale=0.5, prefill_attention_backend="torch",
            use_index_select=use_index_select,
        )
        rt = M12LayerRuntime(0, cfg)
        rt.prefill_first(history, history, history)
        output = rt.prefill_chunk(next_chunk, next_chunk, next_chunk, history.device)
        assert rt.stage_k.is_pinned() and rt.stage_v.is_pinned()
        results.append((output, rt.last_prefill_block_ids))

    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("budget", [0, 33])
def test_decode_budget_cannot_exceed_allocated_maximum(budget):
    with pytest.raises(ValueError, match="decode_top_k_blocks"):
        M12Config(top_k_blocks=32, decode_top_k_blocks=budget)
