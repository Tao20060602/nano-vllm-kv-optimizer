"""Exact block-sparse attention laboratory (milestone M8).

This module is deliberately standalone: it proves the exact Block-DIPR semantics
described in ``docs/block_sparse_design.md`` against a dense single-token decode
oracle.  It does **not** touch the engine scheduler, paged KV allocator, CPU
store or the production ``Attention.forward`` path.

Tensor contract (one decode token, one sequence)::

    q: [num_query_heads, head_dim]
    k: [num_tokens, num_kv_heads, head_dim]
    v: [num_tokens, num_kv_heads, head_dim]

``num_query_heads`` must be divisible by ``num_kv_heads`` (grouped-query
attention).  Query heads map to KV heads in contiguous groups, matching the
model layout.

Retrieval works on raw, *unscaled* inner products; attention softmax uses
``head_dim ** -0.5`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# GQA head mapping
# ---------------------------------------------------------------------------

def kv_head_for_query(num_query_heads: int, num_kv_heads: int, device=None) -> torch.Tensor:
    """Return ``[num_query_heads]`` int tensor: KV head index of each query head."""
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )
    queries_per_kv_head = num_query_heads // num_kv_heads
    return torch.arange(num_query_heads, device=device) // queries_per_kv_head


def _default_scale(head_dim: int) -> float:
    return float(head_dim) ** -0.5


# ---------------------------------------------------------------------------
# Retrieval-block <-> physical-block mapping (preparation for M9)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalBlockMap:
    """Map a global retrieval-block id to ``(logical_block_index, offset, len)``.

    The physical nano-vLLM KV block is 256 tokens; the initial retrieval block
    is 64 tokens, so one *logical* 256-token block holds four retrieval blocks.

    Naming caveat (corrected in M9): ``locate`` returns a **logical** 256-token
    block index, i.e. the position of the block within one sequence's logical
    KV layout.  It is *not* the paged-cache GPU physical block id.  The physical
    slot is obtained from the sequence's block table::

        physical_gpu_block_id = sequence.block_table[logical_block_index]

    Use :meth:`resolve_physical` for that indirection.  This helper is M9/M11
    scaffolding; the laboratory attention tensors themselves remain contiguous.
    """

    physical_block_size: int = 256
    retrieval_block_size: int = 64

    def __post_init__(self) -> None:
        if self.retrieval_block_size <= 0:
            raise ValueError("retrieval_block_size must be positive")
        if self.physical_block_size % self.retrieval_block_size != 0:
            raise ValueError(
                f"physical_block_size ({self.physical_block_size}) must be "
                f"divisible by retrieval_block_size ({self.retrieval_block_size})"
            )

    @property
    def retrieval_blocks_per_physical(self) -> int:
        return self.physical_block_size // self.retrieval_block_size

    def locate(self, retrieval_block_id: int, num_tokens: int) -> tuple[int, int, int]:
        """Return ``(logical_block_index, token_offset, valid_length)``.

        ``logical_block_index`` is the 256-token logical block index within a
        sequence (not a paged physical id); ``token_offset`` is within that
        logical block; ``valid_length`` truncates the final, possibly partial
        retrieval block against ``num_tokens``.
        """
        if retrieval_block_id < 0:
            raise ValueError("retrieval_block_id must be non-negative")
        rp = self.retrieval_blocks_per_physical
        logical_block_index = retrieval_block_id // rp
        within_logical = retrieval_block_id % rp
        token_offset = within_logical * self.retrieval_block_size
        start = retrieval_block_id * self.retrieval_block_size
        valid_length = min(self.retrieval_block_size, num_tokens - start)
        if valid_length <= 0:
            raise ValueError(
                f"retrieval_block_id {retrieval_block_id} lies beyond num_tokens={num_tokens}"
            )
        return logical_block_index, token_offset, valid_length

    def resolve_physical(
        self, block_table, retrieval_block_id: int, num_tokens: int
    ) -> tuple[int, int, int]:
        """Return ``(physical_gpu_block_id, token_offset, valid_length)``.

        ``block_table`` is a sequence's logical->physical mapping
        (``sequence.block_table``); ``physical_gpu_block_id`` is the true paged
        cache slot.  This is the indirection M9 must not skip.
        """
        logical_block_index, token_offset, valid_length = self.locate(
            retrieval_block_id, num_tokens
        )
        if logical_block_index >= len(block_table):
            raise IndexError(
                f"logical block {logical_block_index} is outside the block table "
                f"(len={len(block_table)})"
            )
        physical_gpu_block_id = int(block_table[logical_block_index])
        return physical_gpu_block_id, token_offset, valid_length


# ---------------------------------------------------------------------------
# Core attention primitives
# ---------------------------------------------------------------------------

def gqa_token_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Raw, unscaled inner-product scores for every query head.

    Returns ``[num_query_heads, num_tokens]`` where entry ``[h, j]`` is
    ``q[h] @ k[j, kv_head(h)]``.
    """
    if q.dim() != 2:
        raise ValueError(f"q must be [num_query_heads, head_dim], got {tuple(q.shape)}")
    if k.dim() != 3:
        raise ValueError(f"k must be [num_tokens, num_kv_heads, head_dim], got {tuple(k.shape)}")
    num_query_heads, head_dim = q.shape
    num_tokens, num_kv_heads, k_dim = k.shape
    if head_dim != k_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim}, k={k_dim}")
    g = kv_head_for_query(num_query_heads, num_kv_heads, device=q.device)
    # k_g[h, j, :] = k[j, g(h), :]
    k_g = k[:, g, :].permute(1, 0, 2).contiguous()  # [Hq, T, D]
    return torch.einsum("hd,htd->ht", q, k_g)


def dense_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense single-token decode attention (the correctness oracle).

    Returns ``[num_query_heads, head_dim]``.
    """
    if scale is None:
        scale = _default_scale(q.shape[-1])
    token_scores = gqa_token_scores(q, k)
    probs = torch.softmax(token_scores * scale, dim=-1)
    return _weighted_value(q, v, probs)


def _weighted_value(q: torch.Tensor, v: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """``out[h] = sum_j probs[h, j] * v[j, kv_head(h)]``."""
    num_query_heads = q.shape[0]
    num_tokens, num_kv_heads, head_dim = v.shape
    g = kv_head_for_query(num_query_heads, num_kv_heads, device=v.device)
    v_g = v[:, g, :].permute(1, 0, 2).contiguous()  # [Hq, T, D]
    out = torch.bmm(probs.unsqueeze(1), v_g).squeeze(1)  # [Hq, D]
    return out


def sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention over the packed, selected tokens only.

    ``token_indices`` is a 1-D sorted unique int64 tensor.  Returns
    ``[num_query_heads, head_dim]``.
    """
    if scale is None:
        scale = _default_scale(q.shape[-1])
    if token_indices.numel() == 0:
        raise ValueError("cannot attend to an empty token set")
    if token_indices.dim() != 1:
        raise ValueError("token_indices must be 1-D")
    num_query_heads, head_dim = q.shape
    num_tokens, num_kv_heads, _ = v.shape
    g = kv_head_for_query(num_query_heads, num_kv_heads, device=q.device)

    sel_k = k[token_indices]            # [S, Hkv, D]
    sel_k = sel_k[:, g, :].permute(1, 0, 2).contiguous()  # [Hq, S, D]
    sel_v = v[token_indices][:, g, :].permute(1, 0, 2).contiguous()  # [Hq, S, D]

    scores = torch.bmm(q.unsqueeze(1), sel_k.transpose(1, 2)).squeeze(1) * scale  # [Hq, S]
    probs = torch.softmax(scores, dim=-1)
    return torch.bmm(probs.unsqueeze(1), sel_v).squeeze(1)


# ---------------------------------------------------------------------------
# Block selection
# ---------------------------------------------------------------------------

def exact_block_scores(
    token_scores: torch.Tensor, retrieval_block_size: int
) -> torch.Tensor:
    """Per-block maximum token score, including a partial final block.

    ``token_scores`` is ``[num_query_heads, num_tokens]`` (raw, unscaled).  The
    temporary score tensor is padded with ``-inf`` only; K/V tokens are never
    materialized and no padded index is ever returned.

    Returns ``[num_query_heads, num_retrieval_blocks]``.
    """
    if token_scores.dim() != 2:
        raise ValueError("token_scores must be [num_query_heads, num_tokens]")
    if retrieval_block_size <= 0:
        raise ValueError("retrieval_block_size must be positive")
    num_query_heads, num_tokens = token_scores.shape
    num_blocks = (num_tokens + retrieval_block_size - 1) // retrieval_block_size
    padded_len = num_blocks * retrieval_block_size
    if padded_len != num_tokens:
        pad = torch.full(
            (num_query_heads, padded_len - num_tokens),
            float("-inf"),
            dtype=token_scores.dtype,
            device=token_scores.device,
        )
        scores = torch.cat([token_scores, pad], dim=-1)
    else:
        scores = token_scores
    scores = scores.view(num_query_heads, num_blocks, retrieval_block_size)
    return scores.amax(dim=-1)


def select_topk_blocks(block_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Boolean ``[num_query_heads, num_blocks]`` mask of the top-k blocks per head."""
    if block_scores.dim() != 2:
        raise ValueError("block_scores must be [num_query_heads, num_blocks]")
    num_blocks = block_scores.shape[-1]
    k = min(int(top_k), num_blocks)
    if k <= 0:
        return torch.zeros_like(block_scores, dtype=torch.bool)
    _, idx = block_scores.topk(k, dim=-1)
    mask = torch.zeros_like(block_scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def select_dipr_blocks(block_scores: torch.Tensor, beta: float) -> torch.Tensor:
    """Exact Block-DIPR mask: ``block_score >= per_head_max - beta``.

    ``block_scores`` is ``[num_query_heads, num_blocks]`` (raw, unscaled).
    """
    if block_scores.dim() != 2:
        raise ValueError("block_scores must be [num_query_heads, num_blocks]")
    per_head_max = block_scores.amax(dim=-1, keepdim=True)
    return block_scores >= (per_head_max - float(beta))


def union_block_mask(per_head_block_mask: torch.Tensor) -> torch.Tensor:
    """Boolean ``[num_blocks]`` mask = OR across query heads."""
    return per_head_block_mask.any(dim=0)


# ---------------------------------------------------------------------------
# Token index assembly
# ---------------------------------------------------------------------------

def selected_token_indices(
    block_mask: torch.Tensor,
    num_tokens: int,
    retrieval_block_size: int,
    first_tokens: int = 0,
    recent_tokens: int = 0,
) -> torch.Tensor:
    """Expand a (union) block mask into sorted, deduplicated token indices.

    Adds the ``first_tokens`` leading tokens and the ``recent_tokens`` trailing
    tokens (both clamped to ``[0, num_tokens]``), then sorts and deduplicates.
    Never returns padded, negative or out-of-range indices.
    """
    if block_mask.dim() != 1:
        raise ValueError("block_mask must be 1-D over blocks")
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if retrieval_block_size <= 0:
        raise ValueError("retrieval_block_size must be positive")
    if first_tokens < 0 or recent_tokens < 0:
        raise ValueError("first_tokens and recent_tokens must be non-negative")
    first_tokens = min(int(first_tokens), num_tokens)
    recent_tokens = min(int(recent_tokens), num_tokens)

    expected_blocks = (num_tokens + retrieval_block_size - 1) // retrieval_block_size
    if block_mask.shape[0] != expected_blocks:
        raise ValueError(
            f"block_mask has {block_mask.shape[0]} blocks but num_tokens="
            f"{num_tokens} with retrieval_block_size={retrieval_block_size} "
            f"requires {expected_blocks} blocks"
        )

    device = block_mask.device
    block_ids = block_mask.nonzero(as_tuple=False).squeeze(-1)

    parts: list[torch.Tensor] = []
    if block_ids.numel() > 0:
        offsets = torch.arange(retrieval_block_size, device=device)
        starts = (block_ids * retrieval_block_size).unsqueeze(1)  # [B, 1]
        candidates = (starts + offsets).reshape(-1)              # [B * rbs]
        candidates = candidates[candidates < num_tokens]
        parts.append(candidates)
    if first_tokens > 0:
        parts.append(torch.arange(first_tokens, device=device))
    if recent_tokens > 0:
        parts.append(torch.arange(num_tokens - recent_tokens, num_tokens, device=device))

    if not parts:
        return torch.empty(0, dtype=torch.long, device=device)
    tokens = torch.cat(parts)
    return torch.unique(tokens.sort().values)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def attention_mass_recovery(
    dense_probabilities: torch.Tensor, token_indices: torch.Tensor
) -> float:
    """Mean over query heads of the fraction of dense attention mass kept.

    ``dense_probabilities`` is ``[num_query_heads, num_tokens]`` and already sums
    to 1 over tokens.
    """
    mask = torch.zeros(dense_probabilities.shape[-1], dtype=torch.bool, device=dense_probabilities.device)
    mask[token_indices] = True
    kept = dense_probabilities[:, mask].sum(dim=-1)  # [Hq]
    total = dense_probabilities.sum(dim=-1).clamp_min(1e-12)
    return float((kept / total).mean().item())


def critical_token_recall(
    token_scores: torch.Tensor, beta: float, token_indices: torch.Tensor
) -> float:
    """Fraction of token-level DIPR-critical tokens covered by ``token_indices``.

    A token is critical when its raw score is at least ``per_head_max - beta``.
    The exact block selector must cover all of them (recall 1.0, up to the
    block-superset definition).
    """
    per_head_max = token_scores.amax(dim=-1, keepdim=True)
    critical = token_scores >= (per_head_max - float(beta))  # [Hq, T]
    mask = torch.zeros(token_scores.shape[-1], dtype=torch.bool, device=token_scores.device)
    mask[token_indices] = True
    covered = critical & mask.unsqueeze(0)
    n_critical = int(critical.sum().item())
    if n_critical == 0:
        return 1.0
    n_covered = int(covered.sum().item())
    return n_covered / n_critical


def evaluate_sparse_result(
    dense_output: torch.Tensor,
    sparse_output: torch.Tensor,
    dense_probabilities: torch.Tensor,
    token_indices: torch.Tensor,
) -> dict:
    """Compare a sparse decode result against the dense oracle."""
    diff = dense_output - sparse_output
    max_abs_error = float(diff.abs().max().item())
    dense_norm = float(dense_output.norm().item())
    rel_l2_error = float((diff.norm() / max(dense_norm, 1e-12)).item())
    return {
        "max_abs_error": max_abs_error,
        "rel_l2_error": rel_l2_error,
        "attention_mass_recovery": attention_mass_recovery(dense_probabilities, token_indices),
    }
