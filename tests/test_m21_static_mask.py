"""Correctness and lifecycle checks for the M21 eager selector mask cache."""

import pytest
import torch

from nanovllm.sparse.m12_runtime import M12Config, M12LayerRuntime


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA selector"
)


def _runtime(*, nblocks=8, dynamic_top_k=False):
    cfg = M12Config(
        block_size=2, r=2, recent_tokens=2, sink_tokens=2,
        top_k_blocks=6, max_model_len=32,
        num_heads=2, num_kv_heads=1, head_dim=4,
        dtype=torch.bfloat16, dynamic_top_k=dynamic_top_k,
        selector_static_mask=True,
    )
    runtime = M12LayerRuntime(0, cfg)
    runtime.nblocks_filled = nblocks
    runtime.reps_gpu[:, :nblocks].copy_(
        torch.randn(1, nblocks, cfg.r, cfg.head_dim, device="cuda")
        .to(torch.bfloat16)
    )
    return runtime


def _select(runtime, query, *, static, dynamic=False, budget=None):
    return runtime._gpu_select(
        query, dynamic=dynamic, budget=budget, use_static_mask=static)


def test_static_mask_matches_legacy_ids_with_ties():
    runtime = _runtime()
    runtime.reps_gpu[:, :runtime.nblocks_filled].zero_()
    runtime.protected_blocks = {0, 4, 100}
    query = torch.zeros(2, 4, device="cuda", dtype=torch.bfloat16)

    legacy_ids, legacy_info = _select(runtime, query, static=False, budget=6)
    static_ids, static_info = _select(runtime, query, static=True, budget=6)

    assert torch.equal(static_ids, legacy_ids)
    assert static_info["n_selected"] == legacy_info["n_selected"]
    assert set(static_ids.tolist()).isdisjoint({0, 4})
    # The mask changes only the protected scores; all unprotected tied scores
    # remain zero before the same top-k operation runs.
    protected = runtime._selector_static_mask
    assert protected.tolist() == [0, 4]


def test_static_mask_matches_legacy_with_dynamic_topk():
    torch.manual_seed(2101)
    runtime = _runtime(dynamic_top_k=True)
    runtime.protected_blocks = {1, 6, 99}
    query = torch.randn(3, 2, 4, device="cuda", dtype=torch.bfloat16)

    legacy_ids, legacy_info = _select(
        runtime, query, static=False, dynamic=True, budget=6)
    static_ids, static_info = _select(
        runtime, query, static=True, dynamic=True, budget=6)

    assert torch.equal(static_ids, legacy_ids)
    assert static_info["n_selected"] == legacy_info["n_selected"]
    assert static_info["n_candidates"] == legacy_info["n_candidates"]


@pytest.mark.parametrize("protected", [set(), {100, 101}])
def test_empty_or_out_of_range_protection_skips_gpu_mask(protected):
    runtime = _runtime()
    runtime.protected_blocks = protected
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)

    legacy_ids, _ = _select(runtime, query, static=False, budget=5)
    static_ids, _ = _select(runtime, query, static=True, budget=5)

    assert torch.equal(static_ids, legacy_ids)
    assert runtime._selector_static_mask is None
    assert runtime._selector_static_mask_key is None


def test_static_mask_matches_legacy_when_every_block_is_protected():
    runtime = _runtime(nblocks=4)
    runtime.protected_blocks = {0, 1, 2, 3, 20}
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)

    legacy_ids, legacy_info = _select(runtime, query, static=False, budget=6)
    static_ids, static_info = _select(runtime, query, static=True, budget=6)

    assert torch.equal(static_ids, legacy_ids)
    assert static_ids.numel() == 1
    assert static_info["n_candidates"] == legacy_info["n_candidates"] == 1


def test_mask_cache_reuses_storage_and_tracks_set_and_history_boundaries():
    runtime = _runtime(nblocks=6)
    runtime.protected_blocks = {0, 2, 100}
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)

    _select(runtime, query, static=True, budget=4)
    first_mask = runtime._selector_static_mask
    first_ptr = first_mask.data_ptr()
    assert first_mask.tolist() == [0, 2]

    _select(runtime, query, static=True, budget=4)
    assert runtime._selector_static_mask is first_mask
    assert runtime._selector_static_mask.data_ptr() == first_ptr

    # In-place set mutation changes the effective tuple and rebuilds indices.
    runtime.protected_blocks.add(4)
    _select(runtime, query, static=True, budget=3)
    second_mask = runtime._selector_static_mask
    assert second_mask is not first_mask
    assert second_mask.tolist() == [0, 2, 4]

    # Crossing a history boundary removes now-invalid protected IDs.
    runtime.nblocks_filled = 3
    _select(runtime, query, static=True, budget=2)
    third_mask = runtime._selector_static_mask
    assert third_mask is not second_mask
    assert third_mask.tolist() == [0, 2]


def test_reset_releases_static_mask_and_allows_a_fresh_sequence_cache():
    runtime = _runtime()
    runtime.protected_blocks = {0, 3}
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    _select(runtime, query, static=True, budget=4)
    assert runtime._selector_static_mask is not None

    runtime.reset()
    assert runtime._selector_static_mask is None
    assert runtime._selector_static_mask_key is None

    runtime.nblocks_filled = 5
    runtime.protected_blocks = {1, 4}
    _select(runtime, query, static=True, budget=3)
    assert runtime._selector_static_mask.tolist() == [1, 4]
