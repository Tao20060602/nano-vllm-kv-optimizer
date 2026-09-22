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

The routing quality gate was run on a separate 16K needle prompt, with the
needle at block 128 and the question in the final chunk.  Both configurations
generated `M14ORBIT42`, but the target block was selected by 30/36 layers
(83.3%) with one global query summary and 29/36 layers (80.6%) with four.
Therefore **four query summaries remain an opt-in experiment and the default
stays at one**.  There is no evidence from this gate to change the r=4
representative construction; the next quality experiment should be a harder
needle/distractor suite, not a speculative representative rewrite.

Raw routing records:

- `benchmarks/results/m14_routing_16k_q1.json`
- `benchmarks/results/m14_routing_16k_q4.json`
