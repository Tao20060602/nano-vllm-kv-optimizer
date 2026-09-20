# Execution Prompt: M9 Real-Model KV Trace and CPU-Offload Laboratory

Implement NanoKV milestone M9 completely, run its tests and real-model
benchmark, and commit the result.  Do not start graph search or scheduler-level
offload integration.

## Environment and starting point

- WSL distribution: `NanoVLLM-Ubuntu`
- repository: `/opt/nano-vllm`
- virtual environment: `/opt/nano-vllm/.venv`
- local model: `/opt/models/Qwen3-0.6B`
- HF cache: `/opt/models/.cache/huggingface`
- M8 implementation commit: `0d51df2`

Read `AGENTS.md`, `docs/environment.md`, `docs/block_sparse_design.md`, the M8
implementation and this file before editing.  The expected implementation base
is the clean HEAD containing this prompt and the design update.

## Why M9 is a laboratory first

M8 used synthetic Gaussian tensors.  M9 must answer two questions with real
model tensors before we modify the scheduler and KV ownership:

1. What quality/selection curve does Block-DIPR produce on an actual Qwen
   attention layer?
2. Does Route A work mechanically: CPU-resident historical K/V -> selected
   contiguous pinned slices -> H2D staging -> GPU attention?

M9 does not claim end-to-end generation speedup or process-level GPU-memory
savings.  nano-vLLM still owns a preallocated paged KV tensor during capture.
Report active replay-buffer bytes separately and say this explicitly.

## Part A: correct the M8 mapping terminology

`RetrievalBlockMap.locate()` currently derives an integer directly from the
global retrieval-block ID and calls it a `physical_block_id`.  In paged
nano-vLLM this is only a **logical 256-token block index**.  A true GPU physical
block ID requires:

```text
physical_gpu_block_id = sequence.block_table[logical_block_index]
```

Correct the variable names/docstrings/tests so M9 does not accidentally use a
logical index as a physical cache ID.  Keep compatibility simple; changing the
tuple's semantic documentation is sufficient if no public named field exists.

Also harden M8 token-index assembly while touching its tests:

- reject `num_tokens <= 0`;
- clamp `first_tokens` and `recent_tokens` to `[0, num_tokens]`;
- reject negative window sizes; and
- reject a block-mask length that cannot describe `num_tokens` at the supplied
  retrieval block size.

Add tests where first/recent windows exceed the context length.  No negative or
out-of-range token index may be returned.

## Part B: opt-in real attention trace

Add a minimal, disabled-by-default tracing mechanism to the shared
`nanovllm.layers.attention.Attention` path.  Do not modify Qwen- or Llama-specific
attention implementations.

For exactly one chosen layer during a cold, single-sequence prefill, capture:

```text
q_last: [num_query_heads, head_dim]
k:      [prompt_tokens, num_kv_heads, head_dim]
v:      [prompt_tokens, num_kv_heads, head_dim]
o_last: [num_query_heads, head_dim]
```

The tensors must be after RoPE because that is what `Attention.forward`
receives.  `q_last` and `o_last` refer to the final prompt token, whose causal
attention covers the entire prompt.  Copy the trace to CPU and detach it so the
full prompt K/V is not retained on GPU by the trace.

The exact capture point is:

```text
Qwen/Llama attention applies RoPE
  -> shared Attention.forward(q, k, v)
  -> capture q[-1], k, v for a cold prefill
  -> FlashAttention computes o
  -> capture o[-1]
```

Only capture when all of these are true:

```text
trace explicitly armed
context.is_prefill is True
context.block_tables is None
exactly one sequence is being traced
```

The trace must be one-shot: after one successful capture, disarm it.  Enabling
the trace happens after `ModelRunner` construction, so model warmup must never
populate the trace.  Retrieving the trace must not return live GPU views.

Expose small ModelRunner/LLMEngine methods to enable one layer's next cold
prefill trace and retrieve/clear it.  The trace path must have no work when
disabled.  The benchmark must use:

- eager mode;
- TP=1;
- batch size one;
- reusable prefix cache disabled;
- CPU prefix cache disabled; and
- a fresh prompt with no paged prefix hit.

Validate that M8 `dense_decode_attention(q_last, k, v)` approximately matches
the traced FlashAttention `o_last`.  Use a tolerance appropriate for the real
model dtype and record the actual max/relative error rather than assuming exact
bit identity.

Do not trace every layer.  One configurable layer (default to a middle layer)
is enough and avoids unnecessary RAM/copy overhead.

## Part C: synchronous layer KV offload

Add a small standalone component, suggested location:

```text
nanovllm/sparse/cpu_offload.py
```

It should own one layer's historical K/V in CPU memory with original model
dtype and layout:

```text
k_cpu/v_cpu: [num_tokens, num_kv_heads, head_dim]
```

Pinned memory should be supported and used by the CUDA benchmark.  A pageable
mode may remain available as a baseline.

Use an explicit container similar to:

```python
class CPULayerKVStore:
    k_cpu: Tensor  # [T, Hkv, D]
    v_cpu: Tensor  # [T, Hkv, D]
    pinned: bool

    @property
    def resident_bytes(self) -> int: ...

    def gather(self, sorted_unique_cpu_indices) -> PackedCPUStaging: ...
```

`gather()` must allocate/copy into contiguous CPU tensors.  In pinned mode the
returned packed tensors themselves must be pinned; taking `k_cpu[indices]` and
assuming the result stays pinned is not sufficient.  Validate contiguity,
device, dtype and bounds.

For a query:

1. copy the tiny query to CPU;
2. compute exact raw GQA token/block scores on CPU (float32 scoring is allowed
   if CPU bfloat16 matmul is unsuitable);
3. select fixed top-k or exact Block-DIPR blocks;
4. union blocks across query heads and add/deduplicate the recent window;
5. gather selected K/V into contiguous CPU staging tensors;
6. make staging pinned when requested;
7. copy only packed selected K/V to GPU;
8. compute attention over packed K/V on GPU; and
9. return the result and timings/byte counts.

The allowed sparse data flow is exactly:

```text
full k_cpu/v_cpu remain on CPU
         |
         +-- CPU scoring reads k_cpu
         |
selected CPU indices
         |
CPU index_select/copy into packed pinned staging
         |
only packed_k/packed_v .to(cuda)
         |
packed GPU attention
```

The following implementation is forbidden because it is not CPU offload:

```python
k_gpu = k_cpu.to("cuda")       # full history copied
v_gpu = v_cpu.to("cuda")       # full history copied
packed = k_gpu[selected]        # selection happens after H2D
```

For `S` selected tokens, H2D bytes must equal:

```text
2 * S * num_kv_heads * head_dim * element_size
```

Full K/V bytes are the same formula with `S = num_tokens`; therefore the active
packed/full byte ratio should equal `S / num_tokens`.  Assert these equalities in
unit tests or benchmark self-checks so a hidden full transfer cannot be reported
as sparse offload.

Keep these timing regions separate:

```text
CPU search/selection
CPU gather into contiguous staging
H2D transfer (synchronised for measurement)
GPU packed attention
total Route-A replay
```

Use `perf_counter` for CPU regions.  Synchronize CUDA immediately before and
after H2D/attention timing.  The reported total should be measured around the
whole replay once as well as accompanied by the component timings; do not only
add independently measured minima.

Add a direct packed-attention primitive if useful.  Do not transfer the full K/V
to GPU inside the sparse path and then index it there.  That would invalidate
the offload demonstration.

The exact CPU scan is an oracle and may be slow.  M10's approximate graph exists
to replace it; do not conceal this cost.

## Part D: real-model benchmark

Create a deterministic benchmark, suggested name:

```text
benchmarks/benchmark_real_kv_offload.py
```

It should:

1. load `/opt/models/Qwen3-0.6B` through nano-vLLM;
2. build a deterministic prompt of a configurable token length;
3. capture one chosen layer's real `q_last/k/v/o_last`;
4. verify the dense PyTorch replay against `o_last`;
5. store historical K/V in CPU memory;
6. compare full-GPU attention, exact top-k Route A and exact Block-DIPR Route A;
7. sweep several useful beta values (for example 0.5, 1, 2, 4, 8) or accept a
   comma-separated CLI sweep;
8. save raw JSON under `benchmarks/results/`.

Record at least:

- model, prompt length, layer, dtype and head layout;
- full K/V tensor bytes;
- CPU store bytes and whether pinned;
- selected blocks/tokens and selected-token ratio;
- attention-mass recovery and critical-token recall;
- max absolute and relative L2 output error;
- search, gather, H2D, packed-attention and total latency;
- packed GPU K/V bytes and `packed/full` active-byte ratio; and
- dense replay vs traced FlashAttention error.

For every beta/top-k result, add benchmark self-checks:

```text
selected indices are sorted and unique
0 < selected_tokens <= prompt_tokens
h2d_bytes == packed_k_bytes + packed_v_bytes
packed/full byte ratio == selected_tokens/prompt_tokens
all reported outputs and errors are finite
exact Block-DIPR critical-token recall == 1.0
```

Abort the benchmark instead of writing JSON if a self-check fails.

Do not call active tensor-byte reduction a reduction in total process
`torch.cuda.memory_reserved()`.  The existing engine preallocates its cache, so
those are different measurements.

Do not assert that real Qwen attention will necessarily have better quality than
the M8 Gaussian benchmark.  Measure it and let the curve decide.

Use a prompt length that safely completes on the local RTX 3080 Laptop GPU,
initially 2048 or 4096 tokens.  Keep the saved benchmark reasonably quick.

## Tests

Add focused CPU tests for the offload/gather component and packed attention.
At minimum verify:

1. CPU K/V store preserves dtype, shape and values;
2. selected staging contains exactly the requested token rows in sorted order;
3. byte counters equal tensor storage sizes;
4. packed attention equals M8 sparse attention for the same selection;
5. all-token packed attention equals dense attention;
6. pageable mode works; and
7. pinned mode is tested when CUDA is available.

Run the complete pytest suite.  The real-model trace/replay check belongs in the
benchmark and saved JSON; it does not need to run in every unit-test invocation.

## Boundaries

Do not implement in M9:

- HNSW, RoarGraph or query-guided graph search;
- CUDA streams, double buffering or asynchronous overlap;
- scheduler block release/refcount changes;
- replacement of the actual generation output with sparse output;
- custom CUDA/Triton kernels; or
- changes to the Llama model adapter.

## Known traps: check these before coding

1. A retrieval block ID is not a paged-cache physical block ID.
2. Advanced indexing of pinned tensors may return pageable output; verify the
   actual staging tensor with `is_pinned()`.
3. The captured prefill `k/v` are valid only for a cold single-sequence path;
   a prefix-cache block table changes the meaning of the tensors passed to
   FlashAttention.
4. `q/k` retrieval scores are raw inner products, while attention softmax uses
   `head_dim ** -0.5`.
5. CPU float32 retrieval scores and GPU model-dtype attention are intentionally
   different stages; document the cast.
6. `torch.cuda.memory_reserved()` from the live engine cannot demonstrate this
   layer laboratory's active-byte savings.
7. Do not assume beta 4 is useful for real Qwen.  The beta sweep is the result.
8. Do not silently fall back to full GPU K/V if pinned allocation or transfer
   fails; fail the benchmark with a clear error.

## Required self-review before commit

Before committing, inspect your own diff and answer each item in the completion
report:

```text
[ ] Did any sparse/offload code transfer full historical K/V to CUDA?
[ ] Are packed CPU staging tensors truly contiguous and pinned in pinned mode?
[ ] Do measured H2D bytes exactly match selected K/V tensor bytes?
[ ] Was the trace captured after RoPE and outside warmup/prefix reuse?
[ ] Does dense replay numerically match traced FlashAttention o_last?
[ ] Are selection, gather, H2D and attention timings separate?
[ ] Are real-model results measured rather than inferred from M8 random data?
[ ] Is default nano-vLLM execution unchanged when tracing is disabled?
[ ] Did the full existing pytest suite pass?
[ ] Is the worktree clean after one M9 commit?
```

## Completion

Commit in one milestone commit, for example:

```text
milestone 9: add real-model CPU KV offload laboratory
```

Report the commit SHA, files changed, full pytest result, exact benchmark
command, dense-vs-FlashAttention validation error, the beta/quality/selection
curve, timing breakdown, active K/V byte ratios and any obstacle that affects
M10 or eventual engine integration.  Do not begin M10.
