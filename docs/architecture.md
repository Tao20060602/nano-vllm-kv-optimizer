# nano-vLLM baseline architecture

This document describes the imported commit, not a hypothetical vLLM design. It is the reference that later KV-reuse changes must preserve when the feature flag is disabled.

## Request call chain

`LLM.generate()` in `nanovllm/engine/llm_engine.py` tokenizes prompts and calls `add_request()`. Each prompt becomes a `Sequence` and enters `Scheduler.waiting`.

Each `LLMEngine.step()` asks `Scheduler.schedule()` for work:

1. During prefill, `BlockManager.can_allocate()` walks complete token blocks using a chained `xxhash` value. Existing matching GPU blocks are counted as cached blocks; `BlockManager.allocate()` puts those physical GPU block IDs in `Sequence.block_table` and allocates the remaining blocks.
2. `ModelRunner.run(seqs, is_prefill=True)` builds input IDs, positions, cumulative query/key lengths, slot mappings, and (when a prefix is present) a GPU block table in `prepare_prefill()`.
3. The model runs layer by layer. `Attention.forward()` writes newly computed K/V into each layer's cache using the Triton `store_kvcache` kernel, then FlashAttention reads either the current K/V or the GPU block table.
4. `Scheduler.postprocess()` hashes newly completed blocks, updates cached-token counts, appends the sampled token, and eventually releases physical blocks.
5. Decode uses `prepare_decode()`, where each sequence contributes its last token, one cache slot, context length, and the GPU block table. FlashAttention's paged KV path reads the physical blocks.

## KV layout and ownership

`ModelRunner.allocate_kv_cache()` computes one complete-block byte size and allocates:

```text
kv_cache.shape = [2, num_hidden_layers, num_gpu_blocks,
                  block_size, local_num_kv_heads, head_dim]
```

Index `0` is K and index `1` is V. Each attention layer receives a view of one layer's physical GPU blocks through `module.k_cache` and `module.v_cache`. The cache uses the model dtype. With tensor parallel size one, `local_num_kv_heads == hf_config.num_key_value_heads`.

The logical/physical distinction is important:

- a `Sequence` owns a logical ordered `block_table`;
- each entry currently names a reusable GPU physical block ID;
- `BlockManager` owns reference counts, free IDs, used IDs, token IDs, and the hash index;
- the model/attention layer owns the actual K/V tensors;
- the scheduler must not directly copy hidden model tensors it does not own.

## Existing prefix-cache semantics

`BlockManager.compute_hash(token_ids, prefix)` chains the previous hash with the current token block. Only complete blocks are considered by `can_allocate()`, so a non-aligned tail is recomputed. A hash hit is additionally checked against the stored token IDs. `hash_blocks()` registers blocks after execution.

This is GPU-resident prefix caching only. A physical GPU block can disappear when its reference count reaches zero and it is returned to the free deque. There is no persistent CPU slot, CPU eviction policy, transfer metric, model/layout fingerprint, or cross-GPU-eviction restore path in the upstream snapshot. Those are the NanoKV contribution and must be implemented without treating a GPU block ID as a durable identity.

## Baseline invariants for NanoKV

- `tensor_parallel_size=1`, eager mode, and batch size one are the first supported integration target.
- Cached token counts must describe the prefix actually present in the GPU cache before model execution.
- The model consumes a block table only after all referenced K/V data is ready.
- A failed CPU restore must leave the request safe to recompute from the uncached boundary.
- Feature-disabled execution must retain this upstream call chain and output behavior.

## Milestone 1 instrumentation

Set `enable_cache_metrics=True` when constructing `LLM` to enable the structured
`CacheMetrics` collector. The engine exposes `get_cache_metrics()` and
`reset_cache_metrics()`; the returned mapping is JSON-serializable and includes
lookup hit/miss counts, reused/recomputed tokens, prefill/decode timing, TTFT,
and future CPU-transfer counters. With the flag left at its default `False`, no
lookup timer or model-stage timer is created and the upstream path is unchanged.

Lookup timing is measured around the chained token-block lookup. Model execution
uses synchronised CUDA events when CUDA is available (and a wall-clock fallback
for CPU probes). TTFT is measured from the beginning of the first prefill step
through completion of that model step, so it includes lookup orchestration rather
than only the kernel interval.
