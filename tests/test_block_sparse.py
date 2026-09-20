"""Tests for the M8 exact block-sparse attention laboratory.

Small, deterministic CPU tensors.  Covers dense oracle, physical/retrieval
mapping, exact block scores, top-k, Block-DIPR, critical-token coverage, GQA
head mapping, first/recent-window union, full-selection equivalence and a
controlled sparse example.
"""

import torch

from nanovllm.sparse.block_sparse import (
    RetrievalBlockMap,
    critical_token_recall,
    dense_decode_attention,
    evaluate_sparse_result,
    exact_block_scores,
    gqa_token_scores,
    kv_head_for_query,
    select_dipr_blocks,
    select_topk_blocks,
    selected_token_indices,
    sparse_decode_attention,
    union_block_mask,
)


def _make_tensors(seed=0, num_tokens=10, num_query_heads=4, num_kv_heads=2, head_dim=8):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(num_query_heads, head_dim, generator=g)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, generator=g)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, generator=g)
    return q, k, v


# 1. dense reference matches a direct manual PyTorch calculation ----------------
def test_dense_matches_manual_calculation():
    torch.manual_seed(0)
    q, k, v = _make_tensors(num_query_heads=2, num_kv_heads=2, head_dim=4)
    out = dense_decode_attention(q, k, v)

    scale = 4.0 ** -0.5
    # head 0 shares kv head 0; head 1 shares kv head 1
    manual = []
    for h in range(2):
        scores = (q[h] @ k[:, h, :].T) * scale
        probs = torch.softmax(scores, dim=-1)
        manual.append(probs @ v[:, h, :])
    manual = torch.stack(manual)
    assert torch.allclose(out, manual, atol=1e-6)


# 2. physical/retrieval mapping for 256/64, partial final block ------------------
def test_retrieval_block_mapping_256_64_partial():
    m = RetrievalBlockMap(256, 64)
    assert m.retrieval_blocks_per_physical == 4
    assert m.locate(0, num_tokens=300) == (0, 0, 64)
    assert m.locate(3, num_tokens=300) == (0, 192, 64)
    # partial final retrieval block: block 4 starts at 256, only 44 tokens
    assert m.locate(4, num_tokens=300) == (1, 0, 44)


def test_retrieval_block_mapping_rejects_bad_divisor():
    try:
        RetrievalBlockMap(256, 60)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible sizes")


# 3. exact block scores equal a slow loop implementation ------------------------
def test_exact_block_scores_equal_slow_loop():
    q, k, v = _make_tensors(num_tokens=10, num_query_heads=2)
    token_scores = gqa_token_scores(q, k)  # [2, 10]
    rbs = 4
    bs = exact_block_scores(token_scores, rbs)
    # expected blocks: [0:4],[4:8],[8:10]
    expected = []
    for h in range(2):
        row = []
        for start in range(0, 10, rbs):
            row.append(token_scores[h, start:start + rbs].max())
        expected.append(torch.stack(row))
    expected = torch.stack(expected)
    assert torch.allclose(bs, expected)
    assert bs.shape == (2, 3)


# 4. top-k selects the expected controlled blocks -------------------------------
def test_topk_selects_controlled_blocks():
    block_scores = torch.tensor([[0.0, 5.0, 1.0, 4.0, 2.0]])
    mask = select_topk_blocks(block_scores, top_k=2)
    selected = mask.nonzero(as_tuple=False)[:, 1].tolist()
    assert selected == [1, 3]  # the two largest: 5.0 and 4.0


# 5. Block-DIPR uses per-head max and >= max - beta ------------------------------
def test_dipr_per_head_max_threshold():
    block_scores = torch.tensor([
        [10.0, 8.0, 6.0, 5.0],   # head 0 max = 10
        [1.0, 9.0, 2.0, 0.0],    # head 1 max = 9
    ])
    mask = select_dipr_blocks(block_scores, beta=2.5)
    # head 0: keep >= 7.5 -> [10, 8]
    assert mask[0].tolist() == [True, True, False, False]
    # head 1: keep >= 6.5 -> [9] only
    assert mask[1].tolist() == [False, True, False, False]


# 6. every token-level DIPR-critical token is covered by the block union ---------
def test_critical_tokens_all_covered_by_union():
    torch.manual_seed(1)
    q, k, v = _make_tensors(num_tokens=64, num_query_heads=4, num_kv_heads=2, head_dim=8)
    token_scores = gqa_token_scores(q, k)
    rbs = 16
    block_scores = exact_block_scores(token_scores, rbs)
    beta = 3.0
    per_head_mask = select_dipr_blocks(block_scores, beta)
    union = union_block_mask(per_head_mask)
    token_idx = selected_token_indices(union, num_tokens=64, retrieval_block_size=rbs)
    recall = critical_token_recall(token_scores, beta, token_idx)
    assert recall == 1.0


# 7. GQA maps query heads to the correct shared KV head -------------------------
def test_gqa_maps_query_heads_to_shared_kv_head():
    # 4 query heads, 2 kv heads -> heads 0,1 -> kv0; heads 2,3 -> kv1
    g = kv_head_for_query(4, 2)
    assert g.tolist() == [0, 0, 1, 1]

    # Distinct keys per kv head: kv0 entries = 1, kv1 entries = -1 (same dim)
    num_tokens, num_kv_heads, head_dim = 5, 2, 3
    k = torch.zeros(num_tokens, num_kv_heads, head_dim)
    k[:, 0, :] = 1.0
    k[:, 1, :] = -1.0
    q = torch.zeros(4, head_dim)
    q[0] = 1.0  # head0 shares kv0 -> positive scores
    q[2] = 1.0  # head2 shares kv1 -> negative scores
    scores = gqa_token_scores(q, k)
    assert (scores[0] > 0).all()
    assert (scores[2] < 0).all()


# 8. first/recent-window union is sorted and deduplicated ------------------------
def test_first_recent_union_sorted_deduplicated():
    # one selected block covering tokens 10..13
    block_mask = torch.zeros(4, dtype=torch.bool)
    block_mask[0] = True
    num_tokens = 40
    idx = selected_token_indices(
        block_mask, num_tokens=num_tokens, retrieval_block_size=10,
        first_tokens=2, recent_tokens=5,
    )
    # block0 -> [0..9]; first -> [0,1]; recent -> [35..39]; union sorted unique
    expected = list(range(0, 10)) + list(range(35, 40))
    assert idx.tolist() == expected
    assert idx.dtype == torch.int64
    # overlap between selected block and recent window is deduplicated
    block_mask2 = torch.zeros(4, dtype=torch.bool)
    block_mask2[3] = True  # tokens 30..39
    idx2 = selected_token_indices(
        block_mask2, num_tokens=40, retrieval_block_size=10,
        first_tokens=0, recent_tokens=10,
    )
    assert idx2.tolist() == list(range(30, 40))
    assert len(idx2) == len(set(idx2.tolist()))


# 9. selecting all blocks reproduces dense attention -----------------------------
def test_select_all_reproduces_dense():
    torch.manual_seed(2)
    q, k, v = _make_tensors(num_tokens=48, num_query_heads=4, num_kv_heads=2, head_dim=8)
    dense = dense_decode_attention(q, k, v)
    token_idx = torch.arange(48)
    sparse = sparse_decode_attention(q, k, v, token_idx)
    assert torch.allclose(dense, sparse, atol=1e-5)


# 10. controlled sparse example selects fewer tokens, finite output/metrics ------
def test_controlled_sparse_example_finite_and_fewer_tokens():
    torch.manual_seed(3)
    num_tokens = 128
    q, k, v = _make_tensors(num_tokens=num_tokens, num_query_heads=4,
                            num_kv_heads=2, head_dim=8)
    token_scores = gqa_token_scores(q, k)
    rbs = 32
    block_scores = exact_block_scores(token_scores, rbs)
    per_head = select_topk_blocks(block_scores, top_k=1)
    union = union_block_mask(per_head)
    idx = selected_token_indices(union, num_tokens=num_tokens,
                                 retrieval_block_size=rbs, recent_tokens=16)
    assert 0 < idx.numel() < num_tokens
    sparse = sparse_decode_attention(q, k, v, idx)
    dense = dense_decode_attention(q, k, v)
    scale = 8.0 ** -0.5
    dense_probs = torch.softmax(token_scores * scale, dim=-1)
    metrics = evaluate_sparse_result(dense, sparse, dense_probs, idx)
    assert torch.isfinite(sparse).all()
    assert 0.0 <= metrics["attention_mass_recovery"] <= 1.0
    assert metrics["max_abs_error"] >= 0.0
