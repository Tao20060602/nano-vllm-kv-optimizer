# Milestone 5: end-to-end CPU-backed prefix reuse

Milestone 5 connects `ContextDB` to nano-vLLM's scheduler and model runner for
the supported first-version scope: one GPU, tensor parallel size one, eager mode,
and one scheduled sequence at a time.

## Request path

```text
LLMEngine.add_request(token IDs)
  -> ContextDB longest-prefix lookup
  -> Scheduler compares GPU and CPU prefix lengths
     -> GPU wins ties (no transfer)
     -> CPU wins only for a strictly longer prefix
  -> BlockManager allocates fresh GPU physical blocks for a CPU hit
  -> ModelRunner synchronously loads every matched full block CPU -> GPU
  -> BlockManager registers restored token hashes only after all loads succeed
  -> ModelRunner prefills only the remaining suffix/tail
  -> completed full prompt blocks are synchronously persisted GPU -> CPU
```

`BlockManager` never reads or writes KV tensors. `ModelRunner` owns
`GPUBlockStore`, `CPUBlockStore`, and the copy operations. The scheduler only
handles logical state, physical block IDs, and versioned CPU handles.

At least one prompt token must execute to produce first-token logits. Therefore
CPU lookup may find every complete block, but the engine caps reuse at
`floor((prompt_tokens - 1) / block_size)`. A non-aligned tail is always executed.

## Failure transaction

CPU restore is confirmed only after every H2D copy succeeds. On any exception:

1. all freshly allocated GPU blocks are deallocated;
2. the sequence's cached-token and restore metadata are cleared;
3. the request is marked ineligible for another CPU attempt;
4. metrics are changed from a CPU hit to a recompute miss;
5. the complete prompt is rescheduled.

The integration test injects a load exception and proves the fallback generates
the cold-baseline token IDs, executes all 600 prompt tokens, records one failure,
and leaves zero GPU ref-counts after completion.

## Feature flags and API

```python
llm = LLM(
    model,
    enforce_eager=True,
    tensor_parallel_size=1,
    enable_cache_metrics=True,
    enable_cpu_cache=True,
    cpu_cache_capacity_bytes=...,  # rounded down to complete blocks
    cpu_cache_pinned=True,
)

metrics = llm.get_cache_metrics()
cpu_stats = llm.get_cpu_cache_stats()
evicted_gpu_entries = llm.clear_gpu_prefix_cache()  # engine must be idle
```

Enabling CPU cache automatically enables the reusable GPU adapter and limits
the scheduler to one sequence. Non-eager mode and TP > 1 are rejected rather
than silently using an unvalidated path. All flags default to disabled.

## Correctness evidence

Run the independent feature-disabled baseline followed by the CPU phase:

```bash
python benchmarks/validate_cpu_prefix_reuse.py \
  --model /opt/models/Qwen3-0.6B \
  --phase baseline \
  --output benchmarks/results/cpu_prefix_baseline.json

python benchmarks/validate_cpu_prefix_reuse.py \
  --model /opt/models/Qwen3-0.6B \
  --phase cpu \
  --baseline benchmarks/results/cpu_prefix_baseline.json \
  --pinned \
  --output benchmarks/results/cpu_prefix_integration.json
```

Every case compares generated token IDs with its independently cold prompt.
`prefill_executed_tokens` is counted from the actual `ModelRunner` input tensor,
not inferred from cache lookup metrics.

| Case | Tier/blocks | Reused | Actual prefill | Result |
| --- | --- | ---: | ---: | --- |
| A cold | miss | 0 | 600 | token IDs equal |
| B exact repeat | GPU / 2 | 512 | 88 | token IDs equal |
| C shared prefix | GPU / 1 | 256 | 344 | token IDs equal |
| D GPU identity cleared | CPU / 2 | 512 | 88 | token IDs equal |
| E CPU partial | CPU / 1 | 256 | 344 | token IDs equal |
| F 300-token non-aligned | CPU / 1 | 256 | 344 | token IDs equal |
| G first token differs | miss | 0 | 600 | token IDs equal |
| H differs after 2 blocks | CPU / 2 | 512 | 388 | token IDs equal |
| I CPU LRU eviction | miss | 0 | 600 | token IDs equal |
| J feature disabled | cold upstream path | 0 | full prompt | baseline saved |

Case D records exactly 58,720,256 H2D bytes (two 28 MiB blocks) after three GPU
cache identities were explicitly removed. Final allocator evidence is zero used
blocks, every physical block free, and zero nonzero ref-counts. Raw timings and
counters are stored in the JSON files above.

## Current limitations

- synchronous copies only;
- process-local CPU memory only;
- no compression or quantization;
- batch size / scheduled sequence count one;
- tensor parallel size one;
- eager mode only;
- no sparse attention, ANN, SSD, or remote storage.
