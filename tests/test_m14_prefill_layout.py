import pytest
import torch

from nanovllm.sparse.m12_runtime import (
    M12Config,
    M12LayerRuntime,
    prefill_recent_slice,
)


def _full_coverage_tokens(valid_len: int, block_size: int,
                          sink_tokens: int, recent_tokens: int) -> set[int]:
    nblocks = valid_len // block_size
    recent_first_block = max(1, nblocks - recent_tokens // block_size)
    selected_blocks = range(1, recent_first_block)
    selected = {
        token
        for block in selected_blocks
        for token in range(block * block_size, (block + 1) * block_size)
    }
    sink = set(range(min(sink_tokens, valid_len)))

    buffer_start, offset, length = prefill_recent_slice(
        valid_len, sink_tokens, recent_tokens)
    recent = set(range(buffer_start + offset, buffer_start + offset + length))

    assert not (selected & sink)
    assert not (selected & recent)
    assert not (sink & recent)
    return selected | sink | recent


def test_prefill_recent_slice_default_m12_history() -> None:
    # At the second 4096-token chunk, sink owns [0, 64) and recent owns the
    # prior suffix [3584, 4096); selected non-protected blocks fill the middle.
    assert prefill_recent_slice(4096, 64, 512) == (3584, 0, 512)
    assert _full_coverage_tokens(4096, 64, 64, 512) == set(range(4096))


def test_prefill_recent_slice_deduplicates_sink_at_short_history() -> None:
    # The generic helper is safe even before history exceeds sink + recent.
    assert prefill_recent_slice(128, 64, 512) == (0, 64, 64)
    assert prefill_recent_slice(64, 64, 512) == (0, 64, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA sparse runtime")
def test_prefill_chunk_restores_previous_recent_before_storing_current() -> None:
    """The packed later-chunk path covers all prior tokens in this toy setup."""
    torch.manual_seed(7)
    cfg = M12Config(
        block_size=2, r=1, recent_tokens=2, sink_tokens=2, top_k_blocks=2,
        max_model_len=16, num_heads=2, num_kv_heads=1, head_dim=4,
        dtype=torch.float32, scale=0.5,
    )
    rt = M12LayerRuntime(layer_id=0, cfg=cfg)
    device = torch.device("cuda")
    q0 = torch.randn(8, 2, 4, device=device)
    k0 = torch.randn(8, 1, 4, device=device)
    v0 = torch.randn(8, 1, 4, device=device)
    rt.prefill_first(q0, k0, v0)
    rt._gpu_select = lambda _q: (torch.tensor([1, 2]), {"n_selected": 2})

    q1 = torch.randn(4, 2, 4, device=device)
    k1 = torch.randn(4, 1, 4, device=device)
    v1 = torch.randn(4, 1, 4, device=device)
    actual = rt.prefill_chunk(q1, k1, v1, device)
    expected = rt._fused_attention(
        q1, torch.cat((k0, k1)), torch.cat((v0, v1)),
        hist_len=k0.shape[0], sink_len=cfg.sink_tokens,
        causal_from=k0.shape[0],
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert rt.valid_len == 12
