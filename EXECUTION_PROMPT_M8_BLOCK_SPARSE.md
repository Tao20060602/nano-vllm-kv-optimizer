# Execution Prompt: M8 Exact Block-Sparse Attention Laboratory

You are the implementation agent for NanoKV milestone M8.  Implement this
milestone completely, run its tests and benchmark, then commit the result.  Do
not start M9.

## Environment and base

- WSL distribution: `NanoVLLM-Ubuntu`
- repository: `/opt/nano-vllm`
- virtual environment: `/opt/nano-vllm/.venv`
- base commit at task publication: `75569dc`
- model/cache environment is documented in `AGENTS.md` and
  `docs/environment.md`

Before editing, read the two environment documents and
`docs/block_sparse_design.md`.  Confirm the worktree is clean.  If HEAD has
moved forward only because the design/prompt files were committed, that is the
expected task base; otherwise report the mismatch before proceeding.

## Goal

Create a small, readable PyTorch implementation that proves the exact
Block-DIPR semantics and compares it with full attention and fixed top-k block
selection.  This milestone is deliberately standalone: do not modify the
engine scheduler, paged KV allocator, CPU store, Llama adapter or production
`Attention.forward` path yet.

## Suggested files

You may choose equivalent names, but keep the implementation concentrated:

```text
nanovllm/sparse/__init__.py
nanovllm/sparse/block_sparse.py
tests/test_block_sparse.py
benchmarks/benchmark_block_sparse.py
benchmarks/results/m8_block_sparse.json
```

Update a small amount of documentation if useful.  Avoid unrelated refactors.

## Tensor contract

Support one decode token and one sequence:

```text
q: [num_query_heads, head_dim]
k: [num_tokens, num_kv_heads, head_dim]
v: [num_tokens, num_kv_heads, head_dim]
```

Require `num_query_heads % num_kv_heads == 0`.  Query heads are assigned to KV
heads in contiguous GQA groups, matching the model layout.

Implement these logical operations with readable, vectorized PyTorch:

1. Dense decode attention reference.
2. Raw, unscaled token inner-product scores per query head.
3. Exact block score: maximum token score inside each contiguous retrieval
   block, including a partial final block.
4. Fixed top-k block selection per query head.
5. Exact Block-DIPR selection per query head:
   `block_score >= per_head_global_max - beta`.
6. Union selected blocks across query heads.
7. Expand block IDs into token indices, add optional first/recent-window token
   indices, sort and deduplicate them.
8. Gather packed K/V and compute attention only over those tokens.

Attention uses `head_dim ** -0.5` by default.  Retrieval thresholds operate on
the raw inner product before this scale.

Use this concrete internal representation unless there is a compelling reason
to deviate:

```text
token_scores:          [num_query_heads, num_tokens]
block_scores:          [num_query_heads, num_retrieval_blocks]
per_head_block_mask:   bool[num_query_heads, num_retrieval_blocks]
union_block_mask:      per_head_block_mask.any(dim=0)
selected_token_indices: sorted unique int64[num_selected_tokens]
```

The GQA head mapping is:

```python
queries_per_kv_head = num_query_heads // num_kv_heads
kv_head_for_query = torch.arange(num_query_heads) // queries_per_kv_head
```

For a context whose length is not divisible by the retrieval block size, pad
only the temporary score tensor with `-inf` before the block maximum.  Never
materialize padded K/V tokens and never return padded token indices.

Provide a small pure mapping helper or dataclass for converting a global
retrieval-block ID to:

```text
(physical_block_id, token_offset, valid_length)
```

For the M8 default, physical size is 256 and retrieval size is 64.  Validate
that the physical size is divisible by the retrieval size.  This helper is
preparation for M9; M8's attention tensors themselves remain contiguous.

A recommended public API is shown below.  Exact spelling may change, but keep
the stages separately callable so tests and benchmarks can time them without
copying algorithm logic:

```python
dense_decode_attention(q, k, v, scale=None)
gqa_token_scores(q, k)
exact_block_scores(token_scores, retrieval_block_size)
select_topk_blocks(block_scores, top_k)
select_dipr_blocks(block_scores, beta)
selected_token_indices(block_mask, num_tokens, retrieval_block_size,
                       first_tokens=0, recent_tokens=0)
sparse_decode_attention(q, k, v, token_indices, scale=None)
evaluate_sparse_result(dense_output, sparse_output, dense_probabilities,
                       token_indices)
```

Avoid recomputing dense token scores inside several helper functions during one
benchmark iteration.  The benchmark should make the stages visible, even if a
convenience wrapper also exists.

Return enough structured information for the benchmark to report:

- selected block IDs;
- selected token indices/count/fraction;
- per-head full-attention probability mass covered by selected tokens;
- token-level DIPR critical-token recall (the exact block selector should be
  1.0, up to the definition above);
- sparse output;
- maximum absolute output error and relative L2 error against dense output.

Do not hide the implementation behind a fake speedup.  Exact block scoring is
O(context length) and must be timed as retrieval/selection work.

## Required tests

Use small deterministic CPU tensors.  At minimum cover:

1. dense reference matches a direct manual PyTorch calculation;
2. physical/retrieval mapping behaviour for 256/64, including a partial final
   retrieval block;
3. exact block scores equal a slow loop implementation;
4. top-k selects the expected controlled blocks;
5. Block-DIPR uses a per-query-head maximum and `>= max - beta`;
6. every token-level DIPR-critical token is covered by the selected block union;
7. GQA maps query heads to their correct shared KV head;
8. first/recent-window union is sorted and deduplicated;
9. selecting all blocks reproduces dense attention within a reasonable floating
   point tolerance; and
10. a sparse controlled example selects fewer tokens and still returns finite
    output and metrics.

Run the complete existing test suite as well as the new test file.

## Benchmark

Provide a deterministic CLI benchmark that runs on CUDA when available and can
fall back to CPU.  It should accept useful flags including context length,
retrieval block size, query/KV head counts, head dimension, beta, top-k, warmup,
repeat count and output path.

For at least full attention, top-k and exact Block-DIPR, record:

- configuration and device;
- dense attention latency;
- selection latency;
- sparse attention latency;
- total selection + sparse-attention latency;
- selected blocks/tokens and selected-token ratio;
- attention-mass recovery;
- max absolute and relative L2 output error.

Synchronize CUDA around timed regions.  Label exact Block-DIPR as an oracle and
state that its full scan is not expected to improve end-to-end latency.

Run one practical local-GPU benchmark that completes quickly.  Save its raw JSON
under `benchmarks/results/`.  Do not fabricate favourable numbers.

Use this as the initial practical run unless memory or runtime requires a small
adjustment:

```bash
python benchmarks/benchmark_block_sparse.py \
  --context-length 8192 \
  --retrieval-block-size 64 \
  --num-query-heads 16 \
  --num-kv-heads 4 \
  --head-dim 128 \
  --beta 4.0 \
  --top-k 8 \
  --recent-tokens 128 \
  --warmup 10 \
  --repeats 50 \
  --output benchmarks/results/m8_block_sparse.json
```

## Completion

Commit everything in one commit with a message similar to:

```text
milestone 8: add exact block-sparse attention laboratory
```

Then report only:

1. commit SHA;
2. files added/changed;
3. pytest summary;
4. benchmark command;
5. headline measurements; and
6. any limitation or design question that should be reviewed before M9.

Do not implement CPU offload, graph search, engine integration or custom CUDA
kernels in this milestone.
