# NanoKV Block-Sparse Attention Design

## Project goal

This phase turns NanoKV from a prefix-KV reuse prototype into an educational
long-context inference prototype inspired by AlayaDB.  The target is a working,
measurable research demo rather than a production serving engine.

The finished demo should show that NanoKV can:

1. divide historical KV into retrieval blocks;
2. select relevant blocks for a decode query;
3. attend only to the selected KV tokens plus a local window;
4. keep historical KV in CPU memory and gather selected slices back to GPU;
5. replace exact block scanning with a query-guided graph search; and
6. report the latency, GPU-memory, retrieval and output-quality trade-offs.

We do **not** claim to reproduce AlayaDB's production RoarGraph index, query
optimizer, vector file system or published performance numbers.  The accurate
description is:

> A block-level sparse-attention and CPU-KV-offload prototype inspired by
> AlayaDB's DIPR/DIPRS design.

## Existing invariants

- The physical nano-vLLM KV-cache block remains 256 tokens.  It is part of the
  allocator, prefix cache, Triton store kernel and FlashAttention paged path.
- The initial retrieval block is 64 tokens.  One physical block therefore maps
  to four retrieval blocks identified by `(physical_block_id, token_offset,
  valid_length)`.
- Retrieval block size is an experiment parameter and will later be swept over
  32, 64, 128 and 256.
- Sparse attention and offload are opt-in.  Existing inference remains the
  default path.
- The first supported sparse path is eager mode, TP=1 and one sequence.
- Qwen3 and Llama use the shared `nanovllm.layers.attention.Attention`, so the
  eventual engine integration belongs in the shared path, not in model-specific
  adapters.

## Block-DIPR semantics

For one decode query head `q`, historical keys are partitioned into contiguous
retrieval blocks `B`.  Use the unscaled inner product for retrieval:

```text
s_B(q) = max(k_j in B) q @ k_j
M(q)   = max_B s_B(q)
```

Exact Block-DIPR selects:

```text
S(q) = { B | s_B(q) >= M(q) - beta }
```

This is a block-contiguous superset of token-level DIPR: every token satisfying
`q @ k_j >= max_s(q @ k_s) - beta` belongs to a selected block.  After block
selection, attention still uses the model's normal softmax scale.

For grouped-query attention, each query head scores the KV head it shares.  The
prototype forms the union of selected blocks across query heads before gathering
tokens.  This preserves every head's selected blocks while producing one packed
K/V buffer for the layer.  The selected-token ratio may therefore be higher
than a per-head custom kernel could achieve.

The most recent local window is always included.  A small sink/first-token
window may also be enabled as an experiment.  Forced tokens are deduplicated
against retrieved blocks.

## Milestone plan

### M8: exact Block-DIPR laboratory

Build a standalone PyTorch algorithm layer and benchmark before changing the
engine's decode path.  It must include:

- dense single-token decode attention as the oracle;
- exact per-block max scores;
- fixed top-k block selection;
- exact Block-DIPR threshold selection;
- union, deduplication and packed K/V gathering;
- sparse attention over the packed tensors;
- selected blocks/tokens, attention-mass recovery and output-error metrics; and
- a deterministic CUDA benchmark with JSON output.

The exact selector scans all keys, so it is a correctness oracle, not an
acceleration result.  M8 must report selection latency separately from sparse
attention latency and must not claim an end-to-end speedup.

### M9: real-model trace and synchronous CPU-offload laboratory

Before changing scheduler ownership, capture one real Qwen attention layer's
last-prefill-token query and complete causal K/V.  Replay that layer through a
synchronous Route-A laboratory:

```text
decode q
  -> select retrieval blocks
  -> read CPU KV slices
  -> synchronous H2D gather into a contiguous GPU staging buffer
  -> union with GPU-resident recent-window KV
  -> sparse attention
```

V1 deliberately omits CUDA-stream overlap, double buffering and NUMA tuning.
Metrics must separate search, gather, H2D and attention time.  GPU residency
must distinguish active K/V tensor bytes from the nano-vLLM allocator's reserved
process memory.  This milestone validates real-model attention quality and the
offload data path but does not yet feed sparse outputs back into generation.

### M10: representatives and Block-DIPRS

Replace exact scanning with approximate retrieval baselines:

1. mean key per retrieval block;
2. multiple real representative key positions per block (start with `r=4`);
3. a simple KNN graph baseline; and
4. a query-guided block graph inspired by RetrievalAttention/RoarGraph.

The query-guided graph is built per `(layer, KV-head/group)` from sampled real
queries.  Exact query-to-representative neighbours create a bipartite graph;
shared query neighbours are projected into representative/block edges.  Degree
is capped (initially 16 or 32), with optional K-to-K edges for connectivity.

DIPRS-style search maintains a best score.  After an initial exploration
budget, it only extends candidates whose score is at least `best - beta`.
Candidate blocks are deduplicated and exactly refined before attention.

### M11: opt-in engine integration, final experiment and presentation

After the retrieval and offload mechanisms are validated independently, add a
single-sequence eager integration path through the shared attention layer.  It
must preserve the default dense path, use the selected packed K/V in the actual
attention result, and document how scheduler ownership differs from the
standalone laboratory.

The initial long prefill still has to fit on GPU unless this milestone also adds
chunked/sparse prefill.  The project must not claim otherwise.

Then compare:

Compare:

- full attention;
- fixed top-k blocks;
- exact Block-DIPR;
- mean representatives;
- multi-representative retrieval;
- KNN graph; and
- query-guided Block-DIPRS.

Report block recall, critical-token recall, attention-mass recovery, selected
tokens, visited nodes, index-build time/memory, search/gather/H2D/attention
latency, GPU KV residency, output/logit error and an end-to-end task metric.
Index build time is separate from decode TPOT.  Quality comparisons must use
the same prompt and context.

## Minimum evidence for the internship demo

The repository is successful when it has:

- a one-command demo on the local Qwen3 model;
- an optional Llama-compatible path through the shared attention layer;
- raw JSON/CSV results plus plots;
- at least one full-attention correctness oracle;
- a clear example where fewer KV tokens are attended;
- measured CPU-offload GPU-memory savings;
- measured retrieval/transfer overhead; and
- documentation of limitations and deviations from AlayaDB.

Code elegance is secondary.  Real execution, reproducible measurements and an
honest explanation of the implementation are mandatory.
