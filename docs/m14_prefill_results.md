# M14: chunked-prefill coverage, multi-query routing, and FA2

Commit: `77219bf` (implementation); benchmark data was generated from that
commit on the authoritative WSL environment.

## What changed

1. **Correctness repair.**  A later chunk now packs
   `[selected distant blocks | sink | previous recent | current chunk]` before
   attention, and stores the current K/V only afterwards.  The previous M12
   implementation protected the previous recent blocks from retrieval but did
   not restore them in the packed K/V, leaving a context-coverage gap.
2. **Multi-query router.**  `sparse_prefill_query_segments=4` splits a 4096
   token later chunk into four contiguous segments, averages Q inside each,
   then max-reduces their representative scores.  The selected block count and
   H2D budget remain top-32 / 2048 historical tokens per layer.  `=1` retains
   the old whole-chunk mean route.
3. **FA2 packed prefill.**  Later chunk attention defaults to
   `sparse_prefill_attention_backend="flash"`; the Torch implementation is
   retained as an exact reference backend.  FA2's bottom-right causal mask
   gives each query the whole packed history plus current keys up to its own
   position.

## Validation

`python -m pytest -q tests/test_m14_prefill_layout.py` -> **6 passed**.
Coverage tests include a CUDA toy layout in which selected middle blocks,
sink, and previous recent together equal the entire history.  FA2 output is
also compared against the Torch reference on that same packed causal layout.

## Controlled 8K comparison

Qwen3-4B, BF16, YaRN, 8192-token deterministic prompt, chunk=4096,
query-segments=4, r=4, block=64, top-k=32, sink=64, recent=512.  Each run
generates one token so the recorded number is explicitly **synchronized
generate wall time (prefill plus one decode probe)**, not pure prefill time or
TPOT.

| Later-chunk backend | Wall time | Change vs Torch |
| --- | ---: | ---: |
| Torch reference | 8381.19 ms | baseline |
| FlashAttention-2 | 6673.01 ms | **-20.4%** (1.26x) |

Raw records:

- `benchmarks/results/m14_prefill_8k_q4_torch.json`
- `benchmarks/results/m14_prefill_8k_q4_flash.json`

This is one matched A/B pair, not a multi-run stability claim; it establishes
that the FA2 backend is faster under this controlled setup while preserving the
tested packed-attention semantics.

## Deliberate non-claims and next check

This comparison does **not** establish that four query summaries improve
long-context answer quality versus one global mean.  The next quality gate is
to run the same needle/distractor prompt with segments=1 and segments=4, record
the final-prefill target-block selection across layers, and then compare answer
hits.  Do not change the r=4 representative construction before that test
separates query-compression loss from key-representative loss.
