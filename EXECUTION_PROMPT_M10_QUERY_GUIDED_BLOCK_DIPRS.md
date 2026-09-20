# M10 Execution Prompt: Representatives and Query-Guided Block-DIPRS

## Mission

Start from the clean M9 commit 8d07073 and implement one self-contained M10
milestone. Do not begin M11.

M10 replaces the exact full-token CPU scan used by the M9 correctness oracle
with measurable approximate block retrieval baselines:

1. one mean-key representative per retrieval block;
2. multiple real key-token representatives per block, initially r=4;
3. a simple key-to-key KNN block graph;
4. a sampled-query-guided block graph; and
5. a simplified DIPRS-style dynamic graph traversal followed by exact
   refinement of candidate blocks.

This is an educational, block-level approximation inspired by AlayaDB DIPRS,
RetrievalAttention, RoarGraph and InfLLM. It is not a production RoarGraph
implementation and must never be described as one.

The milestone is successful when it shows, on real Qwen tensors, how
representative quality and graph topology affect oracle recall, attention
quality, visited blocks and Route-A latency.

## Mandatory starting checks

Work only in:

    WSL distribution: NanoVLLM-Ubuntu
    repository:       /opt/nano-vllm
    virtualenv:       /opt/nano-vllm/.venv
    model:            /opt/models/Qwen3-0.6B
    HF_HOME:          /opt/models/.cache/huggingface

Before editing:

    bash scripts/check-wsl-environment.sh
    git status --short --branch
    git rev-parse HEAD

Required starting state:

    HEAD: 8d07073
    worktree: clean

Read completely before coding:

    AGENTS.md
    docs/environment.md
    docs/block_sparse_design.md
    EXECUTION_PROMPT_M9_REAL_KV_OFFLOAD.md
    nanovllm/sparse/block_sparse.py
    nanovllm/sparse/cpu_offload.py
    nanovllm/utils/trace.py
    benchmarks/benchmark_real_kv_offload.py

Do not rewrite M8 or M9. Reuse their exact oracle, GQA mapping, selected-token
packing, metrics and CPU-offload path.

## Frozen terminology and mathematical semantics

The physical KV-cache block remains 256 tokens. The initial retrieval block
remains 64 tokens. M10 operates on logical retrieval-block IDs and contiguous
laboratory tensors; it must not confuse a logical block index with a paged-cache
physical block ID.

For each query head h, use its shared KV head g(h). Retrieval uses raw,
unscaled inner products:

    token_score(h,j) = q[h] dot k[j,g(h)]
    exact_block_score(h,B) = max over j in B token_score(h,j)

Exact Block-DIPR remains the global correctness oracle:

    exact block B is selected iff
    exact_block_score(h,B) >= global_exact_max(h) - beta_raw

Attention alone applies head_dim ** -0.5.

Do not replace the authoritative retrieval score with cosine similarity or
normalized keys. A cosine variant is optional only if clearly labelled as an
ablation and must not replace the raw-inner-product results.

Every result must record both:

    beta_raw
    beta_scaled_logit = beta_raw / sqrt(head_dim)

The selection code still uses beta_raw. The scaled value is a reporting unit,
not a different algorithm. For head_dim=128, raw beta 48 is approximately
scaled-logit beta 4.24.

Do not assume one raw beta is comparable across layers or models. Compare
methods primarily at similar selected-token ratios, oracle recall and output
quality. Quantile/adaptive beta may be an explicitly labelled calibration
ablation, but it must not replace standard Block-DIPR.

## Part A: real query samples without changing default execution

M9 captured q_last/k/v/o_last. M10 also needs real query-distribution samples
for offline graph construction.

Extend the existing one-shot tracer minimally:

- keep the existing arm call backward compatible;
- add an optional positive query-sample count used only when explicitly armed;
- deterministically choose stratified positions from the prefill q tensor;
- exclude the final prompt position, because q_last is the held-out evaluation
  query;
- copy only the sampled post-RoPE queries and their positions to CPU;
- keep q_last/k/v/o_last exactly as in M9;
- preserve the cold-prefill, single-sequence, no-block-table conditions;
- when sampling is not requested, do no extra tensor selection or copy.

Suggested trace fields:

    q_samples: CPU tensor [num_samples, num_query_heads, head_dim] or None
    q_sample_positions: CPU int64 tensor [num_samples] or None

Use about 128 samples for the real benchmark, clamped to the available
non-final positions. Positions must be sorted, unique, in range and must not
contain prompt_length - 1.

The sampled queries are index-construction data. q_last is the held-out query
used for the reported attention output and search-quality comparison. Do not
train or tune the graph using q_last.

This is a per-context offline laboratory: the sampled prefill queries and the
indexed keys come from the same prompt, while q_last is excluded and held out.
State this explicitly in the result limitations. Do not claim cross-document or
cross-workload graph generalization from this experiment.

## Part B: block representative baselines

Add a standalone CPU module, suggested location:

    nanovllm/sparse/representatives.py

It must support a partial final retrieval block and preserve the association
between every representative and its logical retrieval block.

### B1. Mean-key baseline

For each KV head and retrieval block, compute one float32 mean key:

    mean_reps: [num_kv_heads, num_blocks, head_dim]

This is a synthetic representative and must be labelled mean-key, not a real
token.

For a query head, the approximate block score is q dot mean_rep for its mapped
KV head.

### B2. Multiple real-token representatives

Start with r=4 actual key tokens per retrieval block and KV head. Implement a
deterministic, query-independent coverage heuristic:

1. cast one block/head's keys to float32 for representative construction;
2. choose as the first representative the actual token closest in squared L2
   distance to that block/head's mean key;
3. repeatedly choose the token whose minimum squared L2 distance to the
   already chosen representatives is largest;
4. break ties by the lowest token offset;
5. for a partial block use min(r, valid_length) unique representatives.

Store both original global token positions and original key vectors. The
retrieval score remains raw q dot representative key; L2 is used only by the
offline coverage heuristic.

A suggested fixed-shape representation is:

    real_keys:      [num_kv_heads, num_blocks, r, head_dim]
    real_positions: [num_kv_heads, num_blocks, r] int64
    real_valid:     [num_kv_heads, num_blocks, r] bool

Use position -1 plus real_valid=False for unused slots in a partial block.
Masked slots must never participate in scoring, graph construction or work
counters. Equivalent list-based storage is acceptable if it preserves the same
invariants.

For one query head and block:

    representative_block_score = max over the block's real representatives
                                 q dot representative

Do not call these representatives an exact implementation of InfLLM. They are
a deterministic multi-real-token coverage baseline inspired by representative
based block retrieval.

### B3. Flat representative selectors

Implement two non-graph approximate selectors:

- flat mean representative scan over all blocks;
- flat r-real-representative scan over all blocks.

For each selector:

1. compute approximate scores without scanning all full-token keys;
2. form approximate candidates by the approximate best-minus-beta rule;
3. exactly refine only those candidate blocks using their full CPU keys;
4. threshold refined blocks against the best exact score found among the
   candidates;
5. union selected blocks across GQA query heads;
6. add/deduplicate first/recent windows through the existing helper.

These baselines isolate representation error from graph-search error.

## Part C: two block graph baselines

Add a small dependency-free CPU implementation, suggested location:

    nanovllm/sparse/graph_diprs.py

Do not add FAISS, HNSW, cuVS, a vector database or another large dependency.
The number of blocks in this laboratory is small enough for exact offline graph
construction.

Build one graph per KV head. All query heads mapped to the same KV head share
that graph, matching the intended GQA grouping.

Represent graph adjacency as deterministic directed integer neighbor lists.
No self edges, duplicate neighbors or out-of-range nodes are allowed. Enforce
the configured maximum out-degree, initially sweep 16 and 32. Report in-degree,
out-degree and reachability/component diagnostics rather than assuming the
graph is connected.

### C1. Key-to-key KNN block graph

This is the conventional baseline and must remain distinct from the
query-guided graph.

Using the r real representatives, define symmetric block similarity:

    key_similarity(A,B) =
        max over representative a in A and b in B of a dot b

For each source block, retain the highest-scoring degree other blocks with
deterministic tie-breaking by block ID.

This is a K-to-K graph. It does not solve query/key distribution mismatch and
must not be presented as query-aware.

### C2. Sampled-query-guided projected block graph

Use only q_samples, never q_last, to construct this graph.

For each KV head/group:

1. pool sampled queries from every query head mapped to that KV head;
2. score each sampled query exactly against every real representative for that
   KV head;
3. reduce representative scores to block scores by max;
4. take the exact top projection_topk blocks for that sampled query;
5. treat the query and those blocks as a temporary bipartite neighbourhood;
6. project it into block-to-block co-occurrence edges by incrementing the edge
   weight for every ordered pair of distinct blocks in that neighbourhood;
7. for each source block, keep highest-weight neighbours first, with
   deterministic tie-breaking;
8. if fewer than degree query-guided neighbours exist, fill remaining outgoing
   slots from that block's K-to-K KNN list.

Initial projection_topk may be 8. Make it configurable.

Record query-guided edge count, KNN-fill edge count, degree statistics and
index-build time/bytes. Index construction is offline and its time must never
be mixed into per-query search or Route-A total.

This is a simplified query-guided projected block graph. It may be described as
inspired by RoarGraph's query-to-key bipartite projection, but not as RoarGraph
itself.

## Part D: simplified Block-DIPRS traversal

Graph nodes are retrieval blocks. Score a visited node by the maximum raw inner
product between the current query head and that block's stored real
representatives.

Use the same search implementation, entry nodes and budgets for the KNN and
query-guided graphs so topology is the only difference.

Use deterministic, query-independent entry block IDs, initially:

    0
    num_blocks // 2
    num_blocks - 1

Deduplicate them for short contexts.

Implement the following simplified DIPRS semantics:

1. initialise candidate list C with the entry blocks;
2. score them and maintain best representative score seen so far;
3. scan C in insertion order;
4. for each unscored outgoing neighbour, compute its representative score and
   mark it visited/scored;
5. while C has not exceeded initial exploration threshold l0, append every
   newly scored neighbour;
6. afterwards append a neighbour only if its score is at least
   best_seen - beta_raw;
7. stop when C is exhausted or max_scored_blocks is reached;
8. from C, retain representative candidates within final
   best_representative - beta_raw;
9. exactly refine only those candidate blocks using their full CPU keys;
10. return refined blocks whose exact score is at least
    best_exact_candidate - beta_raw.

Initial suggested values:

    l0: 16
    max_scored_blocks: 32 and 64
    graph degree: 16 and 32

The search result must report at least:

- entry blocks;
- scored/visited blocks;
- appended candidate blocks;
- representative-threshold candidates;
- exact-refined blocks;
- final selected blocks;
- representative dot products;
- full-token dot products used in exact refinement;
- whether max_scored_blocks truncated the search.

Run search independently per query head on the graph of its mapped KV head,
then union final selected blocks across heads before gathering K/V.

The global exact full-token scan is allowed only for the separately labelled
oracle and evaluation metrics. It must not run inside the timed approximate
selector or approximate Route-A path.

With beta_raw=infinity, max_scored_blocks at least num_blocks and a graph from
which all blocks are reachable from the entries, traversal plus refinement must
match the exact all-block oracle. Add a controlled test for this invariant.

## Part E: exact oracle and evaluation definitions

For q_last, compute the existing exact full-token Block-DIPR oracle outside all
approximate timing regions.

Report both per-head and GQA-union retrieval metrics:

    oracle block recall
    oracle block precision
    critical-token recall
    selected block/token count and ratio

Compute graph/representative retrieval precision and recall before adding the
forced first/recent windows. Also report the final attended blocks/tokens after
forced-window union. A recent-window hit must not be credited to graph search.

Also report:

    attention-mass recovery
    max absolute output error
    relative L2 output error
    visited block count and ratio
    refined block count and ratio
    representative and refined token dot-product counts

Critical-token recall continues to use the global exact token scores. A value
below 1.0 is expected for approximate methods and must be reported honestly.
Do not assert approximate recall equals 1.0.

For fair method comparisons:

- use the same q_last, K/V, recent window and beta for every method;
- compare KNN graph and query-guided graph with identical r, degree, l0,
  max-scored budget and entries;
- additionally identify closest pairs of configurations by selected-token
  ratio and compare recall/output quality there;
- do not claim one method wins from a different active-token budget.

## Part F: Route-A replay without hidden exact scan

Integrate approximate selectors with the M9 CPU store and packed transfer
laboratory without changing generation.

It is acceptable to add a generic selector-driven Route-A helper while keeping
the M9 route_a_replay API working unchanged.

The timed approximate path must be:

    q to CPU
      -> representative/graph search on CPU
      -> exact refinement of candidate blocks only on CPU
      -> block union plus forced windows
      -> contiguous pinned CPU gather
      -> selected packed K/V H2D only
      -> GPU packed PyTorch attention

Measure separately:

    approximate representative/graph search
    candidate exact refinement
    CPU packed gather
    synchronized packed K/V H2D
    GPU packed attention
    whole approximate Route-A replay

The whole total must be measured around one complete replay and must not be the
sum of independently selected minima.

Retain the M9 byte invariants:

    h2d_bytes = packed_k_bytes + packed_v_bytes
    h2d_bytes = 2 * selected_tokens * Hkv * D * element_size
    packed/full ratio = selected_tokens / prompt_tokens

The full historical K/V must not be transferred to CUDA inside an approximate
path. Full K/V on GPU is allowed only for the explicitly labelled correctness
and dense baseline outside approximate timing.

## Part G: real-Qwen benchmark

Add:

    benchmarks/benchmark_query_guided_diprs.py
    benchmarks/results/m10_query_guided_diprs.json

Use:

    Qwen3-0.6B
    eager mode
    TP=1
    batch size 1
    reusable prefix cache off
    CPU prefix cache off
    one configurable middle layer, default layer 14
    retrieval block size 64
    r=4 real representatives
    about 128 sampled non-final prefill queries
    recent window 128
    pinned CPU staging

Prefer prompt length 8192 so graph traversal is non-trivial. If the verified
hardware cannot run 8192 safely, use 4096 and report the reason; do not silently
change it.

Suggested beta sweep:

    raw beta: 16,24,32,48,64,80

Always derive and store scaled-logit beta. Suggested graph sweeps:

    degree: 16,32
    max_scored_blocks: 32,64
    l0: 16
    projection_topk: 8

Keep the sweep small enough to finish reliably. At minimum the saved result
must contain:

1. exact full-token Block-DIPR oracle;
2. flat mean representatives;
3. flat r=4 real representatives;
4. r=4 K-to-K KNN graph DIPRS;
5. r=4 query-guided graph DIPRS;
6. the GPU-resident PyTorch dense-attention baseline.

Do not name the dense baseline full_gpu_attention without clarification. Use a
name such as gpu_resident_pytorch_dense_attention and state explicitly that it
is dense_decode_attention, not FlashAttention. The traced FlashAttention output
remains the real-engine numerical reference.

Save raw JSON and print a compact human-readable summary. Record:

- exact command and Git commit;
- model/environment/configuration;
- trace shapes and sampled positions summary;
- dense replay versus traced FlashAttention error;
- representative/index build time and bytes;
- graph degree/connectivity and edge-source statistics;
- raw and scaled beta;
- all retrieval, quality, work-count, timing and byte metrics listed above;
- closest selected-token-ratio comparisons;
- limitations and whether any budget truncated search.

Abort without overwriting the final JSON if a self-check fails. Write through a
temporary file and replace the result only after all checks pass.

Expected interpretation is evidence, not a required outcome:

- mean representatives may have poor oracle recall;
- r=4 real representatives should usually improve coverage;
- K-to-K graph may struggle because Q and K distributions differ;
- query-guided projection should improve recall at similar visited budget;
- approximate search may or may not beat exact scan at only 4K/8K context in
  Python;
- no end-to-end speedup may be claimed unless measured.

## Required unit tests

Add focused deterministic CPU tests and CUDA tests only where needed:

1. mean representatives handle full and partial blocks;
2. r-real representatives are actual unique token positions, deterministic and
   in range;
3. farthest-point selection follows a controlled synthetic example;
4. representative scoring uses the correct GQA KV head;
5. flat mean and flat multi-representative selectors never perform hidden full
   refinement outside their candidate blocks;
6. KNN graph has valid deterministic capped adjacency with no self edges;
7. a controlled sampled-query pattern creates the expected query-guided
   co-occurrence edge;
8. graphs are separate per KV head/group;
9. DIPRS performs unconditional early exploration and later beta pruning;
10. work counters exactly match scored representatives and refined full tokens;
11. infinite-beta/full-budget reachable traversal matches the exact oracle;
12. per-head results union correctly and forced windows remain sorted and
    deduplicated;
13. raw/scaled beta conversion is correct;
14. selector-driven Route A preserves pinned/contiguous staging and exact byte
    accounting;
15. optional query sampling is off by default and excludes q_last when enabled;
16. all pre-existing tests still pass.

Use small controlled tensors for semantics. Do not make unit tests depend on
the Qwen model or network.

## Explicit boundaries

Do not implement in M10:

- M11 shared-attention generation integration;
- scheduler KV ownership, refcount or block-release changes;
- sparse outputs feeding actual token generation;
- chunked or sparse prefill;
- CUDA streams, async overlap or double buffering;
- custom CUDA/Triton kernels;
- full HNSW, NSG, RoarGraph, FAISS or cuVS;
- production persistence or incremental graph updates;
- multi-sequence, tensor-parallel or CUDA-graph support;
- Llama adapter changes;
- a switch from raw inner product to cosine as the main method;
- claims of reproducing AlayaDB or its published speedups.

Do not optimize code style at the expense of finishing the measurable
laboratory. Correct semantics, honest counters, reproducible data and clear
method labels matter more than production abstraction.

## Required self-review before commit

Answer every item explicitly in the completion report:

1. Did graph construction use sampled real post-RoPE queries and exclude
   q_last?
2. Are graphs built/shared per KV head group rather than incorrectly mixing all
   KV heads?
3. Are multi representatives actual unique K tokens, with partial blocks
   handled?
4. Is the K-to-K graph clearly separated from the query-guided projected graph?
5. Does the query-guided graph use exact sampled-Q-to-representative neighbours
   before bipartite projection?
6. Is out-degree capped and are entry points/search budgets identical in the
   graph comparison?
7. Does DIPRS explore unconditionally until l0, then prune using
   best_seen - beta_raw?
8. Does exact refinement read only approximate candidate blocks rather than all
   historical keys?
9. Does any timed approximate path call the global exact full-token oracle or
   transfer full historical K/V to CUDA?
10. Are raw beta, scaled-logit beta and selected-token ratio all reported?
11. Are oracle recall, critical-token recall, visited/refined work and output
    error reported without forcing approximate recall to 1.0?
12. Are index-build time and bytes excluded from per-query latency?
13. Is the dense GPU baseline labelled as PyTorch dense attention rather than
    FlashAttention?
14. Is the query-guided method described as simplified/inspired rather than as
    a full RoarGraph reproduction?
15. Is default nano-vLLM behavior unchanged and is the M9 API backward
    compatible?
16. Did the complete pytest suite pass, did the real benchmark finish, and is
    the worktree clean after one M10 commit?

## Completion contract

Commit all M10 implementation and measured results in one milestone commit:

    milestone 10: add query-guided block DIPRS laboratory

Do not begin M11.

Return:

- commit SHA;
- changed-file list;
- full pytest summary;
- exact benchmark command;
- compact table for exact oracle, mean, r=4 flat, KNN graph and query-guided
  graph;
- raw/scaled beta and matched selected-token-ratio comparisons;
- graph/index build time and memory;
- visited/refined ratios and latency breakdown;
- dense-vs-FlashAttention validation;
- all 16 self-review answers;
- any honest limitation or result that affects M11.
