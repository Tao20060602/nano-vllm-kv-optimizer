"""Two block graphs and a simplified Block-DIPRS traversal (M10).

Dependency-free CPU implementation.  One graph is built per **KV head**; all
query heads mapped to the same KV head share it (matching GQA grouping).

Graphs
------
* ``build_knn_graph``: conventional key-to-key KNN block graph.  Block
  similarity is ``max_{a in A, b in B} a . b`` over the real representatives.
* ``build_query_guided_graph``: a *simplified* query-to-key projected block
  graph inspired by RoarGraph's bipartite projection.  It uses ONLY sampled
  prefill queries (never ``q_last``), projects each sampled query's top-k
  representative blocks into a block-to-block co-occurrence edge weight, and
  fills any shortfall from the KNN list.  It is **not** a full RoarGraph.

Traversal (simplified DIPRS)
----------------------------
Deterministic entry blocks; unconditional exploration until ``l0`` nodes have
been scored, then neighbours are appended only if their representative score is
at least ``best_seen - beta_raw``.  Only representative-scored candidates within
``best_representative - beta_raw`` are then exactly refined against the full CPU
keys; final blocks must clear ``best_exact - beta_raw``.  The global exact
full-token scan never runs inside this timed path.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import perf_counter

import torch

from .block_sparse import kv_head_for_query
from .representatives import BlockRepresentatives


@dataclass
class GraphBuildInfo:
    build_ms: float
    logical_persistent_index_bytes: int
    temporary_build_bytes: int
    degree: int
    num_blocks: int
    num_kv_heads: int
    qguided_edges: int = 0
    knn_fill_edges: int = 0
    mean_out_degree: float = 0.0
    mean_in_degree: float = 0.0
    max_out_degree: int = 0
    max_in_degree: int = 0
    projection_topk: int = 0


@dataclass
class DIPRSSearchResult:
    final_blocks: list[int]          # per-head final selected logical block ids
    entry_blocks: list[int]
    scored_blocks: list[int]        # all representative-scored block ids, in order
    refined_blocks: list[int]        # blocks exactly refined
    rep_candidate_blocks: list[int]  # within best_rep - beta before refinement
    rep_dot_products: int
    refined_token_dots: int
    truncated: bool
    best_rep_score: float
    best_exact_score: float


class BlockGraphIndex:
    """Per-KV-head directed block adjacency + DIPRS traversal."""

    def __init__(self, reps: BlockRepresentatives):
        self.reps = reps
        self.adjacency: list[list[list[int]]] = []
        self.info: GraphBuildInfo | None = None


    def _persistent_bytes(self, extra_edge_bytes: int = 0) -> int:
        reps_bytes = self.reps.mean_reps.numel() * 4 + self.reps.real_keys.numel() * 4
        edges = sum(len(nb) for g in self.adjacency for nb in g)
        return reps_bytes + edges * 16 + extra_edge_bytes

    # -- construction ----------------------------------------------------
    def _pairwise_sim(self, g: int) -> torch.Tensor:
        """Symmetric-ish block similarity ``[B,B]`` over real reps."""
        R = self.reps.real_keys[g]                 # [B, r, D]
        V = self.reps.real_valid[g]                # [B, r]
        dots = torch.einsum("asd,btd->abst", R, R)  # [B,B,r,r]
        vm = V[:, None, :, None] & V[None, :, None, :]
        dots = dots.masked_fill(~vm, float("-inf"))
        sim = dots.amax(dim=(-1, -2))               # [B,B]
        sim.fill_diagonal_(float("-inf"))
        return sim

    def build_knn(self, degree: int) -> None:
        t0 = perf_counter()
        B, Hkv = self.reps.num_blocks, self.reps.hkv
        adj = [[[] for _ in range(B)] for _ in range(Hkv)]
        for g in range(Hkv):
            sim = self._pairwise_sim(g)
            for a in range(B):
                pairs = [
                    (float(sim[a, b]), int(b)) for b in range(B)
                    if b != a and torch.isfinite(sim[a, b])
                ]
                pairs.sort(key=lambda x: (-x[0], x[1]))   # score desc, id asc
                adj[g][a] = [b for (_, b) in pairs[:degree]]
        self.adjacency = adj
        self.knn_adj = [ [list(nb) for nb in adj[g]] for g in range(Hkv) ]
        self.info = GraphBuildInfo(
            build_ms=(perf_counter() - t0) * 1000.0,
            logical_persistent_index_bytes=self._persistent_bytes(),
            temporary_build_bytes=0,
            degree=degree, num_blocks=B, num_kv_heads=Hkv,
            projection_topk=0,
        )
        self._finalize_degree_stats()

    def build_query_guided(
        self,
        q_samples: torch.Tensor,
        num_query_heads: int,
        degree: int,
        projection_topk: int = 8,
    ) -> None:
        """Build query-guided adjacency using ONLY sampled queries (not q_last)."""
        t0 = perf_counter()
        B, Hkv, D = self.reps.num_blocks, self.reps.hkv, self.reps.d
        q_per_kv = num_query_heads // Hkv
        if not getattr(self, "knn_adj", None):
            raise RuntimeError("build_knn must be called before build_query_guided")

        adj = [[[] for _ in range(B)] for _ in range(Hkv)]
        qg_edges = fill_edges = 0
        for g in range(Hkv):
            h0, h1 = g * q_per_kv, (g + 1) * q_per_kv
            Q = q_samples[:, h0:h1, :].reshape(-1, D).float()   # [N*qpk, D]
            R = self.reps.real_keys[g]                          # [B, r, D]
            V = self.reps.real_valid[g]                         # [B, r]
            rs = torch.einsum("nd,bsd->nbs", Q, R)             # [Nq, B, r]
            rs = rs.masked_fill(~V, float("-inf")).amax(dim=-1)  # [Nq, B]
            topk = min(projection_topk, B)
            weights: dict[tuple[int, int], float] = defaultdict(float)
            for row in rs:
                order = torch.argsort(row, descending=True)[:topk].tolist()
                P = [int(b) for b in order if torch.isfinite(row[int(b)])]
                for a in P:
                    for b in P:
                        if a != b:
                            weights[(a, b)] += 1.0
            neigh: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for (a, b), w in weights.items():
                neigh[a].append((w, b))
            for a in range(B):
                neigh[a].sort(key=lambda t: (-t[0], t[1]))
                qg = [b for (_, b) in neigh[a][:degree]]
                knn = self.knn_adj[g][a]
                fill = [nb for nb in knn if nb not in qg][: max(0, degree - len(qg))]
                adj[g][a] = qg + fill
                qg_edges += len(qg)
                fill_edges += len(fill)
        self.adjacency = adj
        qg_ms = (perf_counter() - t0) * 1000.0
        # The query-guided graph depends on a KNN fallback list built just
        # before this call; fold that offline cost into the total build time.
        knn_ms = getattr(self, "_prior_knn_ms", 0.0)
        self.info = GraphBuildInfo(
            build_ms=knn_ms + qg_ms,
            logical_persistent_index_bytes=self._persistent_bytes(),
            temporary_build_bytes=q_samples.numel() * 4,
            degree=degree, num_blocks=B, num_kv_heads=Hkv,
            qguided_edges=qg_edges, knn_fill_edges=fill_edges,
            projection_topk=projection_topk,
        )
        self._finalize_degree_stats()

    def _finalize_degree_stats(self) -> None:
        B, Hkv = self.reps.num_blocks, self.reps.hkv
        indeg_all: list[int] = []
        out = []
        for g in range(Hkv):
            indeg = [0] * B
            for a in range(B):
                outs = self.adjacency[g][a]
                out.append(len(outs))
                for b in outs:
                    indeg[b] += 1
            indeg_all.extend(indeg)
        self.info.mean_out_degree = sum(out) / max(1, len(out))
        self.info.max_out_degree = max(out) if out else 0
        self.info.mean_in_degree = sum(indeg_all) / max(1, len(indeg_all))
        self.info.max_in_degree = max(indeg_all) if indeg_all else 0

    # -- traversal --------------------------------------------------------
    def _entries(self) -> list[int]:
        B = self.reps.num_blocks
        raw = [0, B // 2, B - 1]
        out: list[int] = []
        for x in raw:
            if 0 <= x < B and x not in out:
                out.append(x)
        return out

    def traverse(
        self,
        q_vec: torch.Tensor,
        g: int,
        beta: float,
        l0: int,
        max_scored_blocks: int,
    ) -> DIPRSSearchResult:
        """Representative-only multi-hop DIPRS traversal (no exact refinement).

        Three states are kept separate: ``queued`` (discovered, pending),
        ``scored_set`` (representative score computed), and ``expanded``
        (outgoing neighbours iterated).  A block scored when discovered is NOT
        marked expanded, so when dequeued its own neighbours are explored.
        """
        adj = self.adjacency[g]
        reps = self.reps
        entries = self._entries()

        queued: set[int] = set(entries)
        expanded: set[int] = set()
        scored_set: set[int] = set()
        memo: dict[int, float] = {}

        def rep_score(b: int) -> float:
            if b in memo:
                return memo[b]
            s2 = reps.real_score_block(g, b, q_vec)
            memo[b] = s2
            return s2

        queue = deque(entries)
        scored: list[int] = []
        best_rep = float("-inf")
        truncated = False

        while queue:
            if len(scored) >= max_scored_blocks:
                truncated = True
                break
            b = queue.popleft()
            queued.discard(b)
            if b in expanded:
                continue
            if b not in scored_set:
                s2 = rep_score(b)
                scored.append(b)
                scored_set.add(b)
                if s2 > best_rep:
                    best_rep = s2
            expanded.add(b)
            for nb in adj[b]:
                if nb in expanded or nb in queued or nb in scored_set:
                    continue
                if len(scored) >= max_scored_blocks:
                    truncated = True
                    break
                ns = rep_score(nb)
                scored.append(nb)
                scored_set.add(nb)
                if ns > best_rep:
                    best_rep = ns
                # Unconditional exploration up to l0 scored nodes, then prune.
                if len(scored) <= l0 or ns >= best_rep - beta:
                    queue.append(nb)
                    queued.add(nb)

        rep_dots = sum(int(reps.valid_reps[g, b].item()) for b in scored)
        cand = [b for b in scored if rep_score(b) >= best_rep - beta]
        return DIPRSSearchResult(
            final_blocks=[],
            entry_blocks=entries,
            scored_blocks=scored,
            refined_blocks=cand,
            rep_candidate_blocks=cand,
            rep_dot_products=rep_dots,
            refined_token_dots=0,
            truncated=truncated,
            best_rep_score=best_rep,
            best_exact_score=float("-inf"),
        )

    def search(
        self,
        q_vec: torch.Tensor,
        g: int,
        beta: float,
        l0: int,
        max_scored_blocks: int,
    ) -> DIPRSSearchResult:
        """Convenience: traverse then exactly refine representative candidates."""
        partial = self.traverse(q_vec, g, beta, l0, max_scored_blocks)
        final_blocks, ref_dots, best_exact = refine_candidates(
            self.reps, q_vec, g, partial.rep_candidate_blocks, beta
        )
        partial.final_blocks = final_blocks
        partial.refined_token_dots = ref_dots
        partial.best_exact_score = best_exact
        return partial


def refine_candidates(
    reps: BlockRepresentatives,
    q_vec: torch.Tensor,
    g: int,
    cand: list[int],
    beta: float,
):
    """Exactly refine representative candidates; return (final, dots, best_exact)."""
    if not cand:
        return [], 0, float("-inf")
    scores, _nb, dots = reps.exact_refine_single(q_vec, g, cand)
    best_exact = max(scores)
    final = [int(b) for b, sc in zip(cand, scores) if sc >= best_exact - beta]
    return final, dots, best_exact


def union_per_head(per_head_final: list[list[int]], num_blocks: int) -> torch.Tensor:
    mask = torch.zeros(num_blocks, dtype=torch.bool)
    for blocks in per_head_final:
        for b in blocks:
            mask[int(b)] = True
    return mask
