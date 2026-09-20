# M11 Execution Prompt: Opt-in Engine Integration and Final NanoKV Demo

## Mission

Start from the clean, accepted M10 commit 7701ae3 and complete the final
NanoKV milestone in one implementation commit. M11 must make sparse attention
affect real token generation through the shared Attention layer, demonstrate a
real CPU-resident KV history with packed H2D retrieval, and package the project
for an AlayaDB-related internship.

This remains an educational single-request prototype, not a production serving
engine. The final accurate description is:

> NanoKV is an educational block-level sparse-attention and CPU-KV-offload
> prototype inspired by AlayaDB DIPR/DIPRS. It uses exact scanning as an oracle
> and a simplified sampled-query-guided graph rather than reproducing
> AlayaDB's production RoarGraph system.

Do not begin unrelated productionization after the required M11 deliverables.

## Mandatory starting state

Use only:

    WSL distribution: NanoVLLM-Ubuntu
    repository:       /opt/nano-vllm
    virtualenv:       /opt/nano-vllm/.venv
    model:            /opt/models/Qwen3-0.6B
    HF_HOME:          /opt/models/.cache/huggingface

Before editing:

    bash scripts/check-wsl-environment.sh
    git status --short --branch
    git rev-parse HEAD

Required:

    HEAD: 7701ae3
    worktree: clean

Read completely:

    AGENTS.md
    docs/environment.md
    docs/block_sparse_design.md
    EXECUTION_PROMPT_M9_REAL_KV_OFFLOAD.md
    EXECUTION_PROMPT_M10_QUERY_GUIDED_BLOCK_DIPRS.md
    nanovllm/config.py
    nanovllm/layers/attention.py
    nanovllm/engine/model_runner.py
    nanovllm/engine/llm_engine.py
    nanovllm/engine/scheduler.py
    nanovllm/engine/sequence.py
    nanovllm/sparse/cpu_offload.py
    nanovllm/sparse/representatives.py
    nanovllm/sparse/graph_diprs.py

Do not rework M8-M10 algorithms unless required for the integration adapter.
Do not modify either model adapter. Qwen and Llama must continue to reach the
same shared Attention class.

## Completion definition

M11 is complete only if all of the following are true:

1. feature-off execution remains the original dense nano-vLLM path;
2. an explicit opt-in mode runs only in eager, TP=1, single-sequence mode;
3. dense prefill remains dense and must fit on GPU;
4. every participating layer copies its post-RoPE prefill K/V to CPU;
5. sparse decode appends the current token's post-RoPE K/V to CPU history;
6. the selected packed K/V, not the dense FlashAttention output, produces the
   attention tensor returned to the model during decode;
7. at least one generation produces two or more decode iterations through the
   sparse path and records invocation counters;
8. sparse mode does not maintain or silently read a full paged GPU KV cache;
9. full, exact-DIPR, top-k, mean, real-representative, KNN-graph and
   query-guided-graph modes are measured honestly;
10. final JSON/CSV/plots, README, one-command demo and interview guide are
    generated from real runs.

## Frozen scope and semantics

The existing physical block size remains 256 tokens. Retrieval block size is
configurable and initially 64.

Retrieval continues to use raw, unscaled inner product. Attention uses
head_dim ** -0.5. Do not make cosine the default.

Report:

    beta_raw
    beta_scaled_logit = beta_raw / sqrt(head_dim)

For GQA, each query head searches the graph/index of its mapped KV head. Union
selected retrieval blocks across query heads before one packed K/V gather.

First/sink and recent windows are forced and deduplicated. Retrieval recall is
measured before forced-window union; final active-token and output metrics are
measured after it.

The graph index is frozen at the end of prefill. Newly generated tokens are not
inserted into the graph in M11; they are always covered by the recent window.
State this limitation.

## Part A: explicit opt-in configuration

Add small Config fields with safe defaults. Suggested shape:

    enable_sparse_attention: bool = False
    sparse_selector: str = "query_guided"
    sparse_retrieval_block_size: int = 64
    sparse_recent_tokens: int = 128
    sparse_first_tokens: int = 0
    sparse_top_k: int = 8
    sparse_beta_raw: float = 48.0
    sparse_num_representatives: int = 4
    sparse_graph_degree: int = 16
    sparse_graph_l0: int = 16
    sparse_graph_max_scored_blocks: int = 32
    sparse_graph_projection_topk: int = 8
    sparse_query_samples: int = 128
    enable_sparse_diagnostics: bool = False

Allowed selectors:

    full
    exact_dipr
    top_k
    mean
    real
    knn_graph
    query_guided

When enable_sparse_attention is false, none of these fields may alter execution.

When true, validate and fail early unless:

    enforce_eager is True
    tensor_parallel_size == 1
    max_num_seqs == 1
    enable_reusable_cache is False
    enable_cpu_cache is False

Sparse integration and the M0-M7 reusable-prefix/CPU-prefix cache are separate
prototypes in M11. Do not compose them silently.

Reject invalid selectors, negative windows/beta, non-divisible retrieval block
sizes, non-positive graph budgets and sparse chunked prefill. Do not silently
fall back to dense decode when sparse mode fails.

## Part B: sparse scheduler mode without a paged GPU KV history

The existing scheduler assumes every logical sequence block owns a physical
GPU cache block. That must remain unchanged when the feature is off.

For opt-in sparse mode, add an explicit single-sequence virtual-allocation
branch:

- add a clear Sequence/Scheduler state such as sparse_virtual_allocated;
- do not allocate physical blocks for the prompt or generated tokens;
- do not call BlockManager hash, append, preempt or deallocate operations on
  virtual sparse state;
- prepare prefill/decode slot_mapping as -1 because sparse Attention skips the
  paged-cache store;
- sparse decode must not require a block table;
- finish/reset must clear the virtual state;
- reject batching and preemption instead of attempting unsupported behavior.

Do not fill sequence.block_table with fake physical IDs that later reach normal
refcount/hash code.

In ModelRunner.allocate_kv_cache, sparse mode may allocate a zero-block empty
cache tensor and set num_kvcache_blocks to zero, because shared Attention must
not read or write it. BlockManager(0, ...) is acceptable only behind the sparse
scheduler branch. Default dense allocation must remain byte-for-byte equivalent
in behavior.

This change provides a real distinction:

    dense feature-off: paged K/V history is GPU-resident
    sparse feature-on: full layer histories are CPU-resident; only packed
                       selections are transiently moved to GPU

The CUDA allocator may still reserve model/workspace memory. Report paged-cache
tensor bytes, active packed K/V bytes and process allocated/reserved memory as
different quantities.

Do not claim arbitrary context length: dense prefill still must fit on GPU and
the configured max_model_len still applies.

## Part C: per-layer appendable CPU KV history

Add a runtime component, suggested location:

    nanovllm/sparse/engine_runtime.py

One runtime instance belongs to each shared Attention module/layer. Because
M11 supports only one active sequence, it does not need a production session
map.

Use original model dtype for stored K/V:

    k_cpu/v_cpu: [capacity, Hkv, D]

Requirements:

- allocate or grow CPU history safely up to max_model_len;
- pageable full history is acceptable and preferred over pinning every layer;
- packed staging returned by gather must itself be contiguous and pinned when
  CUDA is available;
- track current valid length separately from capacity;
- prefill initializes exactly T positions;
- each decode call appends exactly one current post-RoPE K/V position before
  attention;
- reset removes the previous request's history and indexes;
- bounds, dtype, device, duplicate append and capacity failures must raise.

At 8192 tokens Qwen3-0.6B has about 32 MiB K/V per layer and about 896 MiB over
28 layers. Record the measured CPU history bytes; do not confuse them with GPU
memory.

### Prefill behavior

Prefill output remains the existing FlashAttention output. For each participating
layer:

1. receive post-RoPE q/k/v in shared Attention.forward;
2. run normal dense FlashAttention for the prompt;
3. copy k/v to that layer's CPU history;
4. deterministically sample non-final q positions for index construction;
5. exclude q[-1] from sampled graph queries;
6. build only the selector state required by the configured mode;
7. record CPU-copy and index-build time separately from dense prefill compute.

The first generated token comes from dense prefill. M11 sparse decode begins on
the next model forward pass. A max_tokens=1 run proves nothing about integration
and must not be used as the demo.

### Decode behavior

For one decode token and layer:

1. receive q/k/v with shape [1,Hq,D], [1,Hkv,D], [1,Hkv,D];
2. copy the tiny q and current k/v to CPU;
3. append current k/v to CPU history;
4. select historical prefix blocks using the configured selector;
5. explicitly add every token in the current recent window, including the
   just-appended current token;
6. gather sorted unique token indices into pinned CPU staging;
7. copy only packed K/V to GPU, except in the explicitly labelled full selector;
8. run packed PyTorch attention on GPU;
9. return the result with shape [1,Hq,D] from shared Attention.forward.

The returned packed-attention tensor must flow into o_proj and the rest of the
model. Do not compute sparse output only for metrics while returning dense
FlashAttention output.

No call to flash_attn_with_kvcache is allowed in enabled sparse decode.

## Part D: selector adapter

Reuse M8-M10 implementations through a thin runtime adapter.

### full

Select all valid CPU-history tokens. This deliberately transfers full K/V and
is the CPU-offload numerical integration baseline, not a sparse result.

### exact_dipr

Run exact full-token CPU scoring and exact Block-DIPR. This is the correctness
oracle and may be slow.

### top_k

Use exact block scores followed by fixed top-k block selection.

### mean and real

Use M10 flat representative selectors with candidate-only exact refinement.

### knn_graph and query_guided

Use the M10 graph traversal, identical entry points and configured budgets.
Build one graph per KV head. The query-guided graph uses sampled non-final
prefill queries only.

For mean/real/graph modes, the representative/indexed prefix is frozen at
prefill length. Generated tokens and any tokens beyond that frozen prefix are
included through the explicit recent window rather than index mutation.

All modes must return a common selection result containing:

- pre-window per-head and union blocks;
- final sorted unique token indices;
- selected-token ratio;
- representative/scored/refined work counters;
- search/refine/gather/H2D/attention times;
- packed/full K/V bytes;
- truncation flag;
- selector name, layer ID and decode step.

For graph modes, preserve both block-head-pair and union-block work counts.

## Part E: shared Attention integration

Modify only the shared nanovllm.layers.attention.Attention path.

Feature-off:

    current store_kvcache + FlashAttention behavior

Feature-on prefill:

    dense FlashAttention output + CPU runtime initialization

Feature-on decode:

    append CPU K/V + select/gather/H2D + packed attention output

ModelRunner may assign runtime/config objects to Attention modules when it
assigns layer_id. Do not edit qwen3.py or llama.py.

Add hard assertions:

- enabled sparse decode has exactly one query token;
- runtime was initialized by a complete single-shot prefill;
- CPU history length equals the model position plus one;
- paged k_cache/v_cache contain zero blocks in sparse mode;
- context block_tables is None in sparse mode;
- packed output is finite and has the expected dtype/device/shape.

Add counters proving the path is live:

    sparse_prefill_initializations
    sparse_decode_layer_calls
    sparse_generated_steps
    dense_decode_fallbacks  # must remain zero when enabled

For N completion tokens, the first token is produced by prefill. Therefore an
all-layer run should normally observe:

    sparse_generated_steps = max(N - 1, 0)
    sparse_decode_layer_calls =
        sparse_generated_steps * num_hidden_layers

Do not hard-code this equality if EOS ends generation early; derive it from the
actual completion length.

## Part F: diagnostics and quality comparison

Add opt-in diagnostics only; default generation must not copy logits to CPU.

When enable_sparse_diagnostics is true, retain compact per-step information:

- top-1 token ID;
- top-5 token IDs and logits;
- finite/logit norm checks;
- optionally the full CPU logits for the short benchmark only.

Compare every sparse run with a separate feature-off dense run using the exact
same prompt, greedy sampling and max_tokens.

Report:

- completion token IDs and decoded text;
- exact sequence match;
- token agreement rate;
- first divergent completion position;
- top-1 agreement per aligned step;
- logit max-absolute and relative-L2 error while the preceding generated token
  histories are still identical;
- sparse invocation counters.

Once generated histories diverge, do not compare logits as if they came from
the same input. Continue reporting end-to-end token/text agreement separately.

The full CPU selector is the most important integration check. It should be
close to dense engine output, subject to PyTorch-vs-FlashAttention numerical
differences. Record actual error; do not demand bit identity.

## Part G: final benchmark matrix

Add:

    benchmarks/benchmark_engine_sparse.py
    benchmarks/results/m11_engine_sparse.json
    benchmarks/results/m11_engine_sparse.csv

Use Qwen3-0.6B, eager, TP=1, batch one and greedy decoding.

### G1. Integration run

Use a deterministic 2048-token prompt and at least 8 completion tokens. Run:

1. dense feature-off GPU-paged baseline;
2. full CPU-history selector;
3. exact Block-DIPR;
4. fixed top-k;
5. flat mean;
6. flat r=4 real representatives;
7. KNN graph;
8. query-guided graph.

Start with:

    retrieval block size: 64
    recent window: 128
    r: 4
    top-k: 8
    raw beta: 48 and one higher-quality setting such as 64 or 80
    graph degree: 16 or 32
    l0: 16
    max scored blocks: 32 or 64
    projection top-k: 8

Keep the matrix small enough to finish reliably. Prefer one representative
configuration per method plus a higher-quality beta comparison rather than a
large Cartesian sweep.

### G2. Block-size sweep

Using the same real prompt/layer data, sweep:

    32, 64, 128, 256

The block-size sweep may use a single-layer replay rather than reloading and
generating with a new full engine for every point. Label it offline replay and
do not mix its latency with integrated TPOT.

At minimum compare exact oracle, mean and query-guided graph for:

    selected-token ratio
    oracle block recall
    critical-token recall
    attention mass
    output relative-L2
    visited/refined work
    search/refine/gather/H2D/attention time

### G3. CPU offload on/off and memory

Compare:

    off: feature-off dense GPU-paged engine
    on:  feature-on CPU-history engine

This is an architectural comparison, not a controlled same-kernel ablation.
State that clearly.

Record after engine initialization and after prefill:

- paged KV-cache tensor bytes;
- CPU history bytes;
- representative/index logical bytes;
- packed active GPU K/V bytes per decode;
- torch.cuda.memory_allocated;
- torch.cuda.memory_reserved;
- peak allocated and peak reserved;
- TTFT, median/p50/p95 TPOT and total generation time.

Index build and CPU-copy time must be reported separately and also honestly
included in observed TTFT. Do not subtract them from the user-visible TTFT.

Full K/V CPU-to-GPU transfer is allowed only for selector=full and must be
labelled. Every other sparse mode must assert H2D bytes equal selected packed
K/V bytes.

### G4. Small end-to-end task

Include at least one deterministic text prompt containing a simple passkey or
needle followed by a direct question. Record:

- whether dense output contains the expected answer;
- whether each sparse output contains it;
- dense-output agreement.

If Qwen3-0.6B fails the task even with dense attention, report that outcome and
do not present sparse failure as a retrieval regression. Do not invent a task
accuracy percentage from one example.

### Benchmark self-checks

Abort before replacing final JSON/CSV if:

- enabled sparse mode has zero sparse decode steps;
- any enabled sparse layer falls back to dense decode;
- history length is inconsistent;
- selected indices are not sorted/unique/in range;
- current token is missing from attention;
- H2D accounting is inconsistent;
- outputs/logits are non-finite;
- feature-off dense baseline fails normal generation;
- graph work exceeds configured budgets without a recorded reason.

Write results through temporary files and atomically replace final outputs only
after all checks pass.

## Part H: tests

Add focused tests for at least:

1. Config defaults leave sparse attention disabled.
2. Invalid sparse combinations fail early.
3. Dense scheduler and block manager behavior remain unchanged feature-off.
4. Sparse scheduler allocates no physical paged blocks.
5. Sparse prefill is single-shot and chunked prefill is rejected.
6. CPU history initializes, appends once, grows safely and resets.
7. Current decode K/V is appended before selection and always attended.
8. Full CPU selector matches dense_decode_attention on controlled tensors.
9. exact/top-k/mean/real/KNN/query-guided adapters return valid common results.
10. Graph/index remains frozen while generated tokens enter through recent
    window.
11. Shared Attention returns the sparse tensor, verified with a sentinel/mock;
    it does not discard it and return dense output.
12. Sparse decode never calls flash_attn_with_kvcache.
13. Sparse mode never reads/writes a full paged GPU KV cache.
14. Only packed K/V is transferred for non-full selectors.
15. Invocation counters equal actual generated sparse steps/layer calls.
16. Runtime state does not leak into a second request.
17. Diagnostics stop aligned-logit comparison after token divergence.
18. Retrieval block sizes 32/64/128/256 handle partial final blocks.
19. Existing M8-M10 unit tests remain unchanged and pass.
20. A CUDA integration smoke test generates at least 3 tokens through the real
    shared sparse Attention path.

Use small deterministic tensors for unit tests. Only the integration smoke test
may load the local Qwen model, and it should be marked so ordinary CPU-only test
runs can skip it when CUDA/model prerequisites are unavailable.

## Part I: final artifacts

Add:

    benchmarks/plot_m11_results.py
    benchmarks/results/m11_quality_tradeoff.png
    benchmarks/results/m11_latency_breakdown.png
    benchmarks/results/m11_memory_comparison.png
    scripts/run_m11_demo.sh
    docs/nanokv_m11_results.md
    docs/nanokv_interview_guide.md

Update README.md with:

- the exact project positioning;
- architecture/data-flow diagram in text or Mermaid;
- feature-off and feature-on commands;
- one-command demo;
- measured hardware/software environment;
- concise M8/M9/M10/M11 result tables;
- explanation of the PyTorch dense baseline versus real traced FlashAttention;
- limitations and non-claims.

The demo script must:

- verify it is running in /opt/nano-vllm and the expected virtualenv;
- use the local Qwen model and HF cache;
- run a short dense baseline and short sparse generation;
- print token outputs, sparse counters, selected ratio, latency and memory;
- fail loudly on missing prerequisites;
- never download another model or install packages.

Plots must be generated from saved JSON/CSV, not manually typed values.

The interview guide must include:

- a 30-second project pitch;
- two truthful resume bullets using measured values;
- the end-to-end call/data flow;
- why raw inner product and scaled attention are different;
- why fixed top-k differs from DIPR;
- why query/key OOD motivates query-guided graph construction;
- GQA index sharing and block-union trade-off;
- exact oracle versus approximate search;
- CPU gather/H2D/attention breakdown;
- active bytes versus CUDA allocated/reserved memory;
- why Python graph traversal did not speed up M10;
- what M11 truly integrates;
- differences from AlayaDB/RoarGraph;
- limitations: dense prefill, single sequence, eager TP=1, frozen graph,
  synchronous transfers and no arbitrary-long-context claim;
- likely interviewer questions with concise answers.

Do not write resume claims until the final benchmark values exist.

## Explicit non-goals

Do not add:

- chunked or sparse prefill;
- arbitrary-long-context claims;
- multi-sequence batching;
- TP>1 sparse support;
- CUDA graphs for sparse mode;
- asynchronous streams, overlap or double buffering;
- custom CUDA/Triton attention kernels;
- full HNSW/NSG/RoarGraph;
- online graph insertion for generated tokens;
- production refcounting/persistence;
- remote models, datasets or services;
- changes to qwen3.py or llama.py;
- benchmark numbers copied from papers;
- claims of end-to-end speedup unless directly measured.

Do not hide Python overhead. A functional slow integration with honest
breakdown is acceptable; a fake fast path or silent dense fallback is not.

## Required self-review before commit

Answer every item:

1. Is feature-off behavior still the original paged FlashAttention path?
2. Does sparse configuration reject unsupported eager/TP/batch/cache modes?
3. Does sparse scheduling allocate zero physical paged KV blocks?
4. Does every enabled layer own a CPU history with correct dtype, length and
   reset behavior?
5. Is the prompt prefill still dense and explicitly limited to fitting GPU?
6. Does sparse decode append current post-RoPE K/V before attention?
7. Is the current token always included in selected indices?
8. Does shared Attention return packed sparse output into o_proj/generation?
9. Is flash_attn_with_kvcache never called during enabled sparse decode?
10. Do counters prove at least two actual sparse decode iterations occurred?
11. Does full CPU selector provide a credible dense integration comparison?
12. Do non-full selectors transfer only packed selected K/V?
13. Are graph indexes per KV head, built without q_last and frozen after
    prefill?
14. Are generated tokens covered by the recent window without graph insertion?
15. Are retrieval metrics computed before forced windows and output metrics
    after them?
16. Are raw/scaled beta and per-layer/aggregate selection statistics recorded?
17. Are index build/CPU copy included in TTFT but separately identified?
18. Are CPU history, paged-cache, active packed and CUDA reserved bytes kept
    distinct?
19. Are dense-vs-sparse logits compared only while token histories align?
20. Do saved JSON/CSV, plots and Markdown come from the same final run?
21. Does the one-command demo use only the authoritative local environment?
22. Are all AlayaDB/RoarGraph similarities and differences stated accurately?
23. Did full pytest and the CUDA integration smoke test pass?
24. Did the complete benchmark finish and leave one clean M11 commit?

## Completion contract

Commit once:

    milestone 11: integrate sparse generation and finalize NanoKV demo

Do not tag the commit yet; Codex will decide after final independent review.

Return:

- final commit SHA and changed-file list;
- full pytest summary and CUDA integration-smoke result;
- exact benchmark and demo commands;
- dense/full/exact/top-k/mean/real/KNN/query-guided comparison table;
- block-size sweep;
- token agreement, first divergence and aligned-logit error;
- sparse invocation counters proving output entered generation;
- TTFT/TPOT/total latency breakdown;
- paged GPU KV, CPU history, packed active and CUDA allocated/reserved bytes;
- task result;
- plot/document paths;
- all 24 self-review answers;
- every limitation or negative result relevant to an interview.
