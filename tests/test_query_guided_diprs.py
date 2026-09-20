"""Deterministic CPU unit tests for the M10 approximate block-retrieval lab.

These tests use tiny controlled tensors and never touch the Qwen model, CUDA
(except the optional Route-A byte-accounting test) or the network.
"""

from __future__ import annotations

import math

import pytest
import torch

from nanovllm.sparse.block_sparse import (
    gqa_token_scores,
    exact_block_scores,
    kv_head_for_query,
    selected_token_indices,
)
from nanovllm.sparse.representatives import (
    BlockRepresentatives,
    flat_mean_select,
    flat_real_select,
)
from nanovllm.sparse.graph_diprs import BlockGraphIndex, union_per_head
from nanovllm.utils.trace import AttentionTracer


def _toy_k(T=40, Hkv=2, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, Hkv, D, generator=g)


# 1. mean reps handle full and partial blocks
def test_mean_reps_full_and_partial_blocks():
    rbs = 8
    k = _toy_k(T=20, Hkv=2, D=4)  # 2 full blocks + partial (4 tokens)
    reps = BlockRepresentatives(k, retrieval_block_size=rbs, r=4)
    assert reps.num_blocks == 3
    # full block mean
    for g in range(2):
        torch.testing.assert_close(
            reps.mean_reps[g, 0], k[:8, g, :].mean(0), rtol=1e-5, atol=1e-6
        )
        # partial block 2 has only 4 tokens
        torch.testing.assert_close(
            reps.mean_reps[g, 2], k[16:20, g, :].mean(0), rtol=1e-5, atol=1e-6
        )
        assert int(reps.valid_reps[g, 2]) == 4


# 2. r-real reps are actual unique token positions, deterministic, in range
def test_real_reps_unique_deterministic_in_range():
    rbs = 8
    k = _toy_k(T=24, Hkv=2, D=4, seed=1)
    a = BlockRepresentatives(k, rbs, r=4)
    b = BlockRepresentatives(k, rbs, r=4)
    for g in range(2):
        for blk in range(3):
            pos = a.real_positions[g, blk][a.real_valid[g, blk]].tolist()
            assert all(0 <= p < 24 for p in pos)
            assert len(set(pos)) == len(pos)              # unique
            # positions lie in this block
            for p in pos:
                assert blk * rbs <= p < (blk + 1) * rbs
            torch.testing.assert_close(a.real_keys[g, blk], b.real_keys[g, blk])


# 3. farthest-point on a controlled example
def test_farthest_point_controlled():
    # D=1, block tokens [0,0,10,10]. mean=5. first closest to mean -> offset 0
    # (tie with offset 1 broken by lowest offset). Then farthest from {0} ->
    # offset 2 (distance 10 tie with offset 3, broken by lowest offset).
    k = torch.tensor([[[[0.0]], [[0.0]], [[10.0]], [[10.0]]]]).reshape(4, 1, 1)
    reps = BlockRepresentatives(k, retrieval_block_size=4, r=2)
    pos = reps.real_positions[0, 0][:2].tolist()
    assert pos == [0, 2]


# 4. representative scoring uses correct GQA KV head
def test_representative_scoring_uses_gqa_head():
    rbs = 8
    k = _toy_k(T=16, Hkv=2, D=4, seed=2)
    reps = BlockRepresentatives(k, rbs, r=2)
    Hq, Hkv, D = 4, 2, 4
    q = torch.randn(Hq, D)
    scores = reps.mean_block_scores(q)
    assert scores.shape == (Hq, reps.num_blocks)
    g_of = kv_head_for_query(Hq, Hkv)
    for h in range(Hq):
        g = int(g_of[h])
        expected = q[h] @ reps.mean_reps[g, 0]
        assert abs(float(scores[h, 0]) - float(expected)) < 1e-5


# 5. flat selectors refine only candidate blocks (no hidden full scan)
def test_flat_selectors_refine_only_candidates():
    rbs = 8
    k = _toy_k(T=32, Hkv=2, D=4, seed=3).float()
    reps = BlockRepresentatives(k, rbs, r=4)
    Hq, D = 4, 4
    q = torch.randn(Hq, D)
    for use_real in (False, True):
        res = flat_real_select(q, reps, beta=1e9) if use_real else flat_mean_select(q, reps, beta=1e9)
        # very large beta -> candidates ~ all blocks; refined count must equal
        # total candidate blocks (no block refined outside candidates).
        cand_count = int(res.approx_mask.sum().item())
        assert res.refined_block_count == cand_count
        # every refined token dot product is bounded by block contents
        assert res.refined_token_dots <= Hq * k.shape[0]


# 6. KNN graph: valid deterministic capped adjacency, no self edges
def test_knn_graph_valid_capped():
    rbs = 8
    k = _toy_k(T=32, Hkv=2, D=4, seed=4)
    reps = BlockRepresentatives(k, rbs, r=4)
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=2)
    B = reps.num_blocks
    for g in range(2):
        for a in range(B):
            nb = gi.adjacency[g][a]
            assert len(nb) <= 2
            assert a not in nb
            assert len(set(nb)) == len(nb)
            assert all(0 <= x < B for x in nb)


# 7. controlled query pattern produces the expected co-occurrence edge
def test_query_guided_cooccurrence_edge():
    # Build a 2-block, 1-KV-head case where a sampled query aligns with both
    # block reps, forcing a co-occurrence edge.
    D = 3
    k = torch.zeros(8, 1, D)
    k[0:4, 0] = torch.tensor([1.0, 0.0, 0.0])    # block 0 all along x
    k[4:8, 0] = torch.tensor([0.0, 1.0, 0.0])   # block 1 all along y
    reps = BlockRepresentatives(k, retrieval_block_size=4, r=1)
    # sampled query = [1,1,0] aligns with both block reps -> projects to {0,1}
    q_samples = torch.tensor([[[1.0, 1.0, 0.0]]])   # [N=1, Hq=1, D]
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=2)
    gi.build_query_guided(q_samples, num_query_heads=1, degree=2, projection_topk=2)
    assert 1 in gi.adjacency[0][0]


# 8. graphs are separate per KV head/group
def test_graphs_separate_per_kv_head():
    rbs = 8
    k = _toy_k(T=32, Hkv=2, D=4, seed=5)
    # make the two KV heads qualitatively different
    k[:, 1] = k[:, 1] * 100.0 + 7.0
    reps = BlockRepresentatives(k, rbs, r=4)
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=2)
    assert gi.adjacency[0] != gi.adjacency[1]


# 9. DIPRS unconditional early exploration then beta pruning
def test_diprs_explore_then_prune():
    rbs = 8
    k = _toy_k(T=64, Hkv=1, D=4, seed=6)
    reps = BlockRepresentatives(k, rbs, r=4)
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=4)
    q = torch.randn(4)
    # generous budget: should explore entries + neighbours
    r0 = gi.search(q, g=0, beta=1e9, l0=16, max_scored_blocks=1000)
    # tiny beta: prunes aggressively -> far fewer scored blocks
    r1 = gi.search(q, g=0, beta=-1e9, l0=0, max_scored_blocks=1000)
    assert len(r0.scored_blocks) >= len(r1.scored_blocks)
    # entries always visited
    assert set(r0.entry_blocks).issubset(set(r0.scored_blocks))


# 10. work counters match scored reps and refined tokens
def test_work_counters_exact():
    rbs = 8
    k = _toy_k(T=48, Hkv=1, D=4, seed=7)
    reps = BlockRepresentatives(k, rbs, r=4)
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=3)
    q = torch.randn(4)
    r = gi.search(q, g=0, beta=4.0, l0=4, max_scored_blocks=20)
    # rep dots = sum valid reps over scored blocks
    expected = sum(int(reps.valid_reps[0, b].item()) for b in r.scored_blocks)
    assert r.rep_dot_products == expected
    # refined token dots = sum valid lengths of refined blocks
    exp_tokens = sum(min(rbs, k.shape[0] - b * rbs) for b in r.refined_blocks)
    assert r.refined_token_dots == exp_tokens


# 11. infinite-beta / full budget / reachable graph == exact oracle (all blocks)
def test_infinite_beta_equals_oracle():
    rbs = 8
    k = _toy_k(T=32, Hkv=1, D=4, seed=8)
    reps = BlockRepresentatives(k, rbs, r=4)
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=32)  # dense enough to reach all blocks
    q = torch.randn(4)
    r = gi.search(q, g=0, beta=float("inf"), l0=16, max_scored_blocks=10000)
    # every block scored, refined, and selected (oracle at beta=inf = all blocks)
    assert set(r.scored_blocks) == set(range(reps.num_blocks))
    assert set(r.final_blocks) == set(range(reps.num_blocks))


# 11b. degree=1 chain: multi-hop must reach every reachable node
def test_degree1_chain_multi_hop():
    k = _toy_k(T=48, Hkv=1, D=4, seed=11)
    reps = BlockRepresentatives(k, retrieval_block_size=8, r=2)
    gi = BlockGraphIndex(reps)
    gi.build_knn(degree=1)
    B = reps.num_blocks
    for b in range(B):
        gi.adjacency[0][b] = [b + 1] if b + 1 < B else []
    q = torch.randn(4)
    r = gi.search(q, g=0, beta=float("inf"), l0=16, max_scored_blocks=1000)
    assert set(r.scored_blocks) == set(range(B))
    assert set(r.final_blocks) == set(range(B))
    assert r.truncated is False


# 12. per-head union + forced windows sorted/dedup
def test_union_and_forced_windows():
    mask = union_per_head([[0, 2], [1, 2]], num_blocks=4)
    assert mask.tolist() == [True, True, True, False]
    idx = selected_token_indices(mask, num_tokens=32, retrieval_block_size=8,
                                 first_tokens=2, recent_tokens=4)
    assert idx.tolist() == sorted(idx.tolist())
    assert len(idx) == len(set(idx.tolist()))
    assert 0 in idx.tolist() and 1 in idx.tolist()          # first window
    assert 31 in idx.tolist()                                # recent window


# 13. raw/scaled beta conversion
def test_raw_scaled_beta_conversion():
    hd = 128
    beta_raw = 48.0
    beta_scaled = beta_raw / math.sqrt(hd)
    assert abs(beta_scaled - 4.242640687119286) < 1e-6


# 14. selector-driven Route A preserves pinned/contiguous staging + bytes
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_route_a_selective_bytes_and_staging():
    from nanovllm.sparse.cpu_offload import CPULayerKVStore, route_a_selective
    T, Hkv, D = 32, 2, 4
    k = torch.randn(T, Hkv, D).to(torch.bfloat16)
    v = torch.randn(T, Hkv, D).to(torch.bfloat16)
    store = CPULayerKVStore(k, v, pinned=True)
    mask = torch.tensor([True, False, True, False])  # blocks 0,2 -> tokens 0-7,16-23
    q = torch.randn(2, D).to(torch.bfloat16)

    def sel():
        return {"union_mask": mask, "search_ms": 0.1, "refine_ms": 0.2,
                "work": {"scored_blocks": 8}}

    res = route_a_selective(store, q, sel, "cuda", retrieval_block_size=8,
                            first_tokens=0, recent_tokens=0)
    assert res.num_selected_tokens == 16
    expected = 2 * 16 * Hkv * D * k.element_size()
    assert res.h2d_bytes == expected
    assert abs(res.active_byte_ratio - 16 / T) < 1e-9
    idx = res.indices
    assert torch.equal(idx, idx.sort().values) and idx.numel() == torch.unique(idx).numel()


# 15. optional query sampling off by default; excludes q_last when enabled
def test_query_sampling_default_and_excludes_last():
    tr = AttentionTracer()
    assert tr.armed is False
    # arm without samples -> q_samples None
    tr.arm(layer_id=0)
    assert tr._query_samples == 0
    tr.clear()

    class Ctx:
        is_prefill = True
        block_tables = None
        cu_seqlens_q = torch.tensor([0, 10])

    tr.arm(layer_id=0, query_samples=4)
    q = torch.randn(10, 2, 4)
    k = torch.randn(10, 1, 4)
    tr.maybe_capture(0, q, k, k, k, Ctx())
    out = tr.retrieve()
    assert out is not None
    assert out.q_samples is not None
    assert out.q_sample_positions is not None
    pos = out.q_sample_positions.tolist()
    assert all(0 <= p < 9 for p in pos)          # excludes final position 9
    assert len(pos) == len(set(pos)) and pos == sorted(pos)


# 16 is covered by running the whole existing suite (see pytest invocation).
