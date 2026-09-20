"""Block representative baselines for M10 (CPU, dependency-free).

Given one layer's CPU K tensor ``[num_tokens, num_kv_heads, head_dim]`` (float32)
and a retrieval-block size, build two kinds of *block representative*:

1. **mean-key**: one float32 mean vector per (KV head, retrieval block).  This is
   a synthetic representative, labelled ``mean-key`` (never a real token).
2. **r real-token representatives** (r=4 by default): actual key tokens chosen by
   a deterministic, query-independent farthest-point coverage heuristic.

Retrieval scoring stays the authoritative raw, unscaled inner product
``q . representative``.  Squared L2 is used *only* by the offline coverage
heuristic.  Partial final retrieval blocks are supported; unused representative
slots are marked ``valid=False`` with ``position=-1`` and never participate in
scoring, graph construction or work counters.

This module also implements the two *flat* (non-graph) approximate selectors:
flat mean scan and flat r-real scan.  Both score only representatives over all
blocks, form a best-minus-beta candidate set, then **exactly refine only those
candidate blocks** against the full CPU keys.  They isolate representation error
from graph-search error.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .block_sparse import kv_head_for_query


@dataclass
class FlatSelectResult:
    per_head_mask: torch.Tensor   # [Hq, B] bool, exact-refined DIPR selection
    rep_dot_products: int        # representative dot products computed
    refined_block_count: int     # candidate blocks that were exactly refined
    refined_token_dots: int      # full-token dot products used in refinement
    approx_mask: torch.Tensor    # [Hq, B] bool, representative-only pre-refine set
    rep_scan_ms: float = 0.0
    refine_ms: float = 0.0


class BlockRepresentatives:
    """Holds per-(KV head, retrieval block) mean and real-token reps."""

    def __init__(self, k_f32: torch.Tensor, retrieval_block_size: int, r: int = 4):
        if k_f32.dim() != 3 or k_f32.dtype != torch.float32:
            raise ValueError("k_f32 must be float32 [T, Hkv, D]")
        if r < 1:
            raise ValueError("r must be >= 1")
        self.k_f32 = k_f32
        self.t, self.hkv, self.d = k_f32.shape
        self.rbs = int(retrieval_block_size)
        self.r = int(r)
        self.num_blocks = (self.t + self.rbs - 1) // self.rbs

        B, Hkv, D, rr = self.num_blocks, self.hkv, self.d, self.r
        self.mean_reps = torch.zeros(Hkv, B, D, dtype=torch.float32)
        self.real_keys = torch.zeros(Hkv, B, rr, D, dtype=torch.float32)
        self.real_positions = torch.full((Hkv, B, rr), -1, dtype=torch.long)
        self.real_valid = torch.zeros(Hkv, B, rr, dtype=torch.bool)
        self.valid_reps = torch.zeros(Hkv, B, dtype=torch.long)

        for g in range(Hkv):
            for b in range(B):
                start = b * self.rbs
                L = min(self.rbs, self.t - start)
                tok = k_f32[start:start + L, g, :]          # [L, D]
                mean = tok.mean(dim=0)
                self.mean_reps[g, b] = mean
                chosen = self._farthest_point_reps(tok, mean)
                self.valid_reps[g, b] = len(chosen)
                for slot, off in enumerate(chosen):
                    self.real_keys[g, b, slot] = tok[off]
                    self.real_positions[g, b, slot] = start + off
                    self.real_valid[g, b, slot] = True

    # -- construction heuristic ------------------------------------------
    def _farthest_point_reps(self, tok: torch.Tensor, mean: torch.Tensor) -> list[int]:
        L = tok.shape[0]
        r_eff = min(self.r, L)
        if L == 0:
            return []
        sq = ((tok - mean) ** 2).sum(dim=-1)                 # [L]
        first = int(torch.argmin(sq).item())
        chosen = [first]
        remaining = {j for j in range(L) if j != first}
        while len(chosen) < r_eff:
            # min squared L2 distance of each unchosen token to chosen set.
            best_j, best_d = -1, -1.0
            for j in sorted(remaining):
                d = float(((tok[j].unsqueeze(0) - tok[chosen]) ** 2).sum(dim=-1).min())
                if d > best_d:                               # strict -> ties keep lower offset
                    best_d, best_j = d, j
            chosen.append(best_j)
            remaining.discard(best_j)
        return chosen

    # -- vectorized representative scoring -------------------------------
    def mean_block_scores(self, q_f32: torch.Tensor) -> torch.Tensor:
        """``[Hq, B]`` raw inner products against mean reps via GQA mapping."""
        Hq = q_f32.shape[0]
        g = kv_head_for_query(Hq, self.hkv, device=q_f32.device)      # [Hq]
        reps = self.mean_reps[g]                                     # [Hq, B, D]
        return torch.einsum("hd,hbd->hb", q_f32, reps)

    def real_block_scores(self, q_f32: torch.Tensor) -> torch.Tensor:
        """``[Hq, B]`` max over valid real reps, via GQA mapping."""
        Hq = q_f32.shape[0]
        g = kv_head_for_query(Hq, self.hkv, device=q_f32.device)      # [Hq]
        keys = self.real_keys[g]                                     # [Hq, B, r, D]
        scores = torch.einsum("hd,hbrd->hbr", q_f32, keys)          # [Hq, B, r]
        valid = self.real_valid[g]                                  # [Hq, B, r]
        scores = scores.masked_fill(~valid, float("-inf"))
        return scores.amax(dim=-1)

    def real_score_block(self, g: int, b: int, q_vec: torch.Tensor) -> float:
        """Scalar rep score for one query head's vector on block b (KV head g)."""
        if not bool(self.real_valid[g, b].any()):
            return float("-inf")
        keys = self.real_keys[g, b][self.real_valid[g, b]]          # [r_eff, D]
        return float((keys @ q_vec).max().item())

    def mean_score_block(self, g: int, b: int, q_vec: torch.Tensor) -> float:
        return float(q_vec @ self.mean_reps[g, b])

    # -- exact refinement of candidate blocks ------------------------------
    def exact_refine_blocks(
        self,
        q_f32: torch.Tensor,
        per_head_candidates: list[torch.Tensor],
    ):
        """Exact per-block max over full CPU keys, only for candidate blocks.

        Returns ``(per_head_scores[list[Tensor]], refined_block_count,
        refined_token_dots)``.  ``per_head_candidates[h]`` is a 1-D int tensor of
        candidate block ids for query head h; per_head_scores[h] aligns with it.
        """
        Hq = q_f32.shape[0]
        g_of = kv_head_for_query(Hq, self.hkv)
        refined_block_count = 0
        refined_token_dots = 0
        out_scores = []
        for h in range(Hq):
            cands = per_head_candidates[h]
            g = int(g_of[h].item())
            scores_h = torch.empty(cands.shape[0], dtype=torch.float32)
            for i, b in enumerate(cands.tolist()):
                start = b * self.rbs
                L = min(self.rbs, self.t - start)
                keys = self.k_f32[start:start + L, g, :]            # [L, D]
                blk_scores = keys @ q_f32[h]                        # [L]
                scores_h[i] = float(blk_scores.max().item())
                refined_block_count += 1
                refined_token_dots += L
            out_scores.append(scores_h)
        return out_scores, refined_block_count, refined_token_dots


    def exact_refine_single(
        self, q_vec: torch.Tensor, g: int, cand: list[int]
    ):
        """Exact per-block max over full keys for one (query vec, KV head).

        Returns ``(exact_scores[list[float]], refined_block_count,
        refined_token_dots)`` aligned with ``cand``.
        """
        scores: list[float] = []
        refined_block_count = 0
        refined_token_dots = 0
        for b in cand:
            start = b * self.rbs
            L = min(self.rbs, self.t - start)
            keys = self.k_f32[start:start + L, g, :]            # [L, D]
            blk = keys @ q_vec                                  # [L]
            scores.append(float(blk.max().item()))
            refined_block_count += 1
            refined_token_dots += L
        return scores, refined_block_count, refined_token_dots


# ---------------------------------------------------------------------------
# Flat approximate selectors (no graph)
# ---------------------------------------------------------------------------

def _flat_selector(
    q_f32: torch.Tensor,
    reps: BlockRepresentatives,
    beta: float,
    *,
    use_real: bool,
) -> FlatSelectResult:
    Hq = q_f32.shape[0]
    B = reps.num_blocks
    t0 = perf_counter()
    rep_scores = reps.real_block_scores(q_f32) if use_real else reps.mean_block_scores(q_f32)
    # representative work: count valid reps actually used per block.
    if use_real:
        valid_per_block = reps.valid_reps.sum(dim=-1)            # [Hkv, B]
        g_of = kv_head_for_query(Hq, reps.hkv)
        rep_dot_products = int(valid_per_block[g_of].sum().item())
    else:
        rep_dot_products = Hq * B

    per_head_mask = torch.zeros(Hq, B, dtype=torch.bool)
    approx_mask = torch.zeros(Hq, B, dtype=torch.bool)
    cand_lists: list[torch.Tensor] = []
    approx_best = rep_scores.amax(dim=-1)                        # [Hq]
    for h in range(Hq):
        cands = (rep_scores[h] >= (approx_best[h] - beta)).nonzero(as_tuple=False).squeeze(-1)
        cand_lists.append(cands)
        approx_mask[h, cands] = True

    scan_ms = (perf_counter() - t0) * 1000.0
    t1 = perf_counter()
    refined_scores, refined_blocks, refined_dots = reps.exact_refine_blocks(q_f32, cand_lists)
    for h in range(Hq):
        cands = cand_lists[h]
        if cands.numel() == 0:
            continue
        rs = refined_scores[h]
        best_exact = float(rs.max().item())
        keep = rs >= (best_exact - beta)
        per_head_mask[h, cands[keep]] = True
    refine_ms = (perf_counter() - t1) * 1000.0

    return FlatSelectResult(
        per_head_mask=per_head_mask,
        rep_dot_products=rep_dot_products,
        refined_block_count=refined_blocks,
        refined_token_dots=refined_dots,
        approx_mask=approx_mask,
        rep_scan_ms=scan_ms,
        refine_ms=refine_ms,
    )


def flat_mean_select(q_f32, reps, beta):
    """Flat mean-representative selector."""
    return _flat_selector(q_f32, reps, beta, use_real=False)


def flat_real_select(q_f32, reps, beta):
    """Flat r-real-representative selector."""
    return _flat_selector(q_f32, reps, beta, use_real=True)
