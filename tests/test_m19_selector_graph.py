"""Exactness and cache-lifecycle checks for the M19 selector CUDA graph."""

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
        selector_cuda_graph=True,
    )
    runtime = M12LayerRuntime(0, cfg)
    runtime.nblocks_filled = nblocks
    runtime.reps_gpu[:, :nblocks].copy_(
        torch.randn(1, nblocks, cfg.r, cfg.head_dim, device="cuda")
        .to(torch.bfloat16)
    )
    return runtime


def _ids(runtime, query, *, graph, dynamic=False, budget=None):
    return runtime._gpu_select(
        query, use_graph=graph, dynamic=dynamic, budget=budget)[0]


def test_graph_matches_eager_for_changing_bf16_queries_and_dynamic_topk():
    torch.manual_seed(1901)
    runtime = _runtime(dynamic_top_k=True)
    for _ in range(4):
        query = torch.randn(3, 2, 4, device="cuda", dtype=torch.bfloat16)
        eager_ids = _ids(runtime, query, graph=False, dynamic=True)
        graph_ids = _ids(runtime, query, graph=True, dynamic=True)
        assert torch.equal(graph_ids, eager_ids)
    assert runtime.selector_graph_builds == 1


@pytest.mark.parametrize("budget", [1, 3, 6])
def test_graph_matches_eager_for_small_budgets_and_protected_ids(budget):
    runtime = _runtime()
    runtime.protected_blocks = {0, 3, 100}  # Ignore protected IDs outside history.
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    eager_ids = _ids(runtime, query, graph=False, budget=budget)
    graph_ids = _ids(runtime, query, graph=True, budget=budget)
    assert torch.equal(graph_ids, eager_ids)
    assert not ({0, 3} & set(graph_ids.tolist()))
    assert graph_ids.numel() == budget


def test_graph_ties_follow_eager_topk_exact_ids():
    runtime = _runtime()
    runtime.reps_gpu[:, :runtime.nblocks_filled].zero_()
    query = torch.zeros(2, 4, device="cuda", dtype=torch.bfloat16)
    eager_ids = _ids(runtime, query, graph=False, budget=4)
    graph_ids = _ids(runtime, query, graph=True, budget=4)
    assert torch.equal(graph_ids, eager_ids)


def test_graph_caps_budget_to_small_history_after_protected_mask():
    runtime = _runtime(nblocks=3)
    runtime.protected_blocks = {0, 20}
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    eager_ids = _ids(runtime, query, graph=False, budget=6)
    graph_ids = _ids(runtime, query, graph=True, budget=6)
    assert torch.equal(graph_ids, eager_ids)
    assert graph_ids.numel() == 2
    assert 0 not in graph_ids.tolist()


def test_graph_matches_eager_when_all_in_range_blocks_are_protected():
    runtime = _runtime(nblocks=3)
    runtime.protected_blocks = {0, 1, 2, 20}
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    eager_ids = _ids(runtime, query, graph=False, budget=6)
    graph_ids = _ids(runtime, query, graph=True, budget=6)
    assert torch.equal(graph_ids, eager_ids)
    assert graph_ids.numel() == 1
    assert graph_ids.item() in {0, 1, 2}


def test_graph_reuses_capture_for_current_query_and_in_place_representatives():
    torch.manual_seed(1902)
    runtime = _runtime()
    for _ in range(2):
        query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(
            _ids(runtime, query, graph=True), _ids(runtime, query, graph=False)
        )
    original_builds = runtime.selector_graph_builds

    # Mutate representative values without reallocating their storage. The
    # captured graph must read the current values through the same pointer.
    runtime.reps_gpu[:, :runtime.nblocks_filled].mul_(-1)
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        _ids(runtime, query, graph=True), _ids(runtime, query, graph=False)
    )
    assert runtime.selector_graph_builds == original_builds == 1


def test_graph_rebuilds_for_shape_protection_and_budget_changes():
    runtime = _runtime()
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    _ids(runtime, query, graph=True, budget=4)
    assert runtime.selector_graph_builds == 1

    runtime.nblocks_filled -= 1
    _ids(runtime, query, graph=True, budget=4)
    assert runtime.selector_graph_builds == 2

    runtime.protected_blocks.add(2)
    _ids(runtime, query, graph=True, budget=4)
    assert runtime.selector_graph_builds == 3

    _ids(runtime, query, graph=True, budget=3)
    assert runtime.selector_graph_builds == 4


def test_reset_drops_graph_and_counters_and_representative_build_invalidates():
    runtime = _runtime()
    query = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    _ids(runtime, query, graph=True)
    assert runtime._selector_graph is not None

    runtime.reset()
    assert runtime._selector_graph is None
    assert runtime._selector_graph_key is None
    assert runtime.selector_graph_builds == 0
    assert runtime.selector_graph_setup_ms == 0.0

    runtime.nblocks_filled = 4
    runtime.reps_gpu[:, :4].copy_(torch.randn_like(runtime.reps_gpu[:, :4]))
    _ids(runtime, query, graph=True)
    assert runtime._selector_graph is not None
    assert runtime.selector_graph_builds == 1

    new_keys = torch.randn(2, 2, 1, 4, device="cuda", dtype=torch.bfloat16)
    runtime._build_reps_gpu(new_keys, start=0, nblocks=2)
    assert runtime._selector_graph is None
    assert runtime._selector_graph_key is None
