# NanoKV Interview Guide

## 30-second pitch

NanoKV is an educational block-level sparse-attention + CPU-KV-offload
prototype built on a fork of nano-vLLM and inspired by AlayaDB DIPR/DIPRS. For
one decode token it keeps the full K/V history on CPU, selects a small set of
retrieval blocks with sampled-query-guided block graphs, gathers only those rows
to GPU and runs packed attention — while dense FlashAttention remains the
feature-off path. It is a teaching prototype, not a serving engine.

## Two truthful resume bullets (measured values)

- Integrated block-level sparse decode into a real transformer engine
  (Qwen3-0.6B, 28 layers): per-layer CPU K/V history, frozen query-guided block
  graph, pinned packed H2D, packed PyTorch attention feeding o_proj; on a 2048-token
  needle task all 7 selectors matched the dense greedy output (token agreement 1.0)
  while transferring only ~0.33 selected tokens per layer instead of the full KV.
- Built an exact Block-DIPR oracle plus simplified sampled-query-guided graph
  traversal (entry exploration to l0, then best-minus-beta pruning) and showed
  honest cost: packed GPU attention is <1.5 ms/layer while Python traversal +
  CPU refinement dominates TPOT — explicitly no speedup is claimed.

## End-to-end data flow

1. Dense FlashAttention prefill (GPU) computes the first token.
2. Each layer copies post-RoPE K/V to a CPU history buffer.
3. Non-final prefill queries are sampled to build mean/real representatives and
   a per-KV-head block graph (KNN + query-guided projection).
4. Each decode: append current post-RoPE K/V to CPU; run selector on CPU; force
   first + recent windows; gather sorted unique rows into pinned staging; H2D
   only those rows; packed PyTorch attention; result → o_proj → sampler.

## Why raw inner product vs scaled attention differ

Retrieval scores use the **unscaled** raw inner product `q·rep`; attention
softmax uses `(q·k)·(head_dim^-0.5)`. Mixing them would distort the beta
threshold. We report both `beta_raw` and `beta_scaled_logit = beta_raw/sqrt(head_dim)`.

## Why fixed top-k differs from DIPR

Top-k takes the k best blocks regardless of the score gap; Block-DIPR keeps all
blocks within `best_block_score - beta`, so it adapts to how peaked the
attention distribution is (fewer blocks when one block dominates).

## Why OOD motivates query-guided graphs

KNN edges are key-to-key; a query whose nearest keys differ from the block's own
neighborhood can miss. Projecting sampled queries onto blocks (a bipartite
co-occurrence graph) adds edges useful for the actual query distribution.

## GQA sharing + block union

All query heads mapped to one KV head share that KV head's graph; the selected
block masks are unioned across query heads before one packed gather.

## Exact oracle vs approximate

Exact full-token CPU scoring is the correctness oracle (slow); flat
representatives and graph traversal isolate representation error from search
error.

## CPU gather / H2D / attention breakdown

Packed GPU attention is sub-millisecond; H2D scales with selected bytes; the
dominant sparse-TPOT cost is Python CPU selection/refinement (no async overlap).

## Active bytes vs CUDA allocated/reserved

"Active packed K/V bytes" = what crosses H2D this step. `torch.cuda.allocated/reserved`
includes the whole context; in sparse mode the paged KV tensor is 0 bytes.

## Why Python traversal didn't speed up M10/M11

Pure-Python BFS + per-block exact refinement on CPU is slower than FlashAttention;
the value is retrieval byte reduction and integration correctness, not latency.

## What M11 actually integrates

The sparse packed-attention output replaces (not shadows) the paged decode output
inside the shared Attention class, and actually changes sampled tokens.

## Differences from AlayaDB/RoarGraph

We use exact scanning as oracle and a simplified sampled-query graph (no online
insertion, no full HNSW/NSG, no async streams, single sequence). We must not
claim to reproduce production RoarGraph.

## Honest limitations

Dense prefill only (must fit GPU), single sequence, eager TP=1, frozen graph at
prefill (new tokens via recent window), synchronous transfers, no arbitrary-long
context, no batching/TP>1, no CUDA graphs, no custom kernels.

## Likely questions

- *Where does the first token come from?* Dense prefill; sparse decode starts on
  the next forward.
- *How do you prove it isn't a silent dense fallback?* Counters stay
  `dense_decode_fallbacks=0`, `flash_attn_with_kvcache` is not called, paged KV
  bytes are 0, and outputs change token IDs.
- *Why report both beta values?* Because raw inner product and scaled logits live
  on different scales; cross-config comparison needs the scaled value.
