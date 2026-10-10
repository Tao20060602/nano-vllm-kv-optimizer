# Line A: int8 CPU-KV quantization design

Status: implemented opt-in (`sparse_quant_history`, default off), 2026-10-10.
Motivation: profiling showed the decode H2D device transfer (~49 ms/token at
Top-32, 288 MiB) is on the critical path, while the selector's CPU time is
not. Halving H2D bytes should therefore show up in TPOT.

## Why H2D bytes, not selector CPU

- `top-k 32 -> 16` saved 34 ms/token; the saving tracked reduced H2D bytes,
  not the reduced selector CPU.
- CUDA graph, static mask and a fused Triton selector each cut selector CPU
  by 5-7 ms but moved TPOT by ~0.

So the lever is the transferred bytes, and the selected-block set must stay
unchanged for quality. int8 quantization halves bytes without touching which
blocks are chosen.

## Scheme

Store each layer's CPU history as int8; dequantize on GPU before attention
(low-precision store, high-precision accumulate). Sink/recent live in separate
GPU bf16 buffers and are never quantized; only the remote-history gather is.

- **K**: symmetric int8, **per-(kv_head, head_dim)** scale `[Hkv, D]`,
  estimated once from the first stored chunk and held fixed (K outlier
  channels are stable - KIVI observation).
- **V**: symmetric int8, **per-(block, kv_head)** scale `[blocks, Hkv]`,
  computed per 64-token block.
- Dequant on GPU: `x = q.float() * scale` (independent elementwise step).
- Selection representatives are still built from the original bf16 K, so the
  routing/fused selector is unaffected.

Modules: `nanovllm/sparse/quant_kv.py` (helpers),
`M12LayerRuntime` (`_store_kv`, gather, pack), config
`sparse_quant_history`.

## Numerical prototype (synthetic, CPU)

`bench_logs/lineA/test_quant_kv.py`, K with 30x/15x outlier channels:

| quantity | value |
|---|---|
| K rel err, per-channel | 0.0087 |
| K rel err, per-tensor | 0.0930 |
| V rel err, per-block | 0.0091 |
| bf16 K/V rel err (baseline) | 0.0017 |
| attention-output rel err, bf16 | 0.0075 |
| attention-output rel err, int8 | 0.0239 |
| block-score rel err, int8 | 0.0033 |
| Top-32 overlap, int8 vs fp32 | 32/32 |

Per-channel is ~10x better than per-tensor for K; routing (Top-32) is exactly
unchanged; attention output error grows from 0.75% (bf16) to 2.4% (int8).
This is synthetic data: only the engine run + 20-row quality screen decides
whether int8 is acceptable.

## Engine A/B (2026-10-10, 32K, Top-32, 3 fresh-process pairs)

| | steady TPOT | H2D device span | gather | selector |
|---|---:|---:|---:|---:|
| baseline | 113.5 | 47.9 | 26.1 | 32.9 |
| int8 | 92.5 | 35.4 | 16.5 | 16.6 |
| delta | **-21.0 (-18.5%)** | -12.5 | -9.6 | -16.3 |

Paired per-run deltas: 21.7 / 22.4 / 18.9 ms. Generated token IDs are
identical to baseline. The selector drop is a second-order effect: its
per-layer `.cpu()` sync waits for the prior layer's H2D, so smaller H2D
shortens the measured selector too. The H2D device span does not halve
because the event span includes the (new) dequantize kernels.

This recovers the Top-16 speed (~88 ms) **without reducing the block count**,
i.e. without the quality loss that lowering Top-K causes.

## Quality screen (20 regenerated 8K multikey rows)

| arm | accuracy | rows |
|---|---:|---|
| baseline | 12/20 | - |
| int8 | **12/20** | identical pass/fail on all 20 |

Per-row answers match exactly (same 8 misses: 6258, 8786, 10273, 18422,
9476, 9546, 7980, 5200). Within this screen, int8 does not change quality.

## Net result

int8 CPU-KV quantization: **-18.5% decode TPOT (113.5 -> 92.5 ms/token at
32K, Top-32) with no observed quality change on the 20-row screen**. It
recovers the Top-16 speed without the Top-16 quality loss. Bounded evidence:
20 rows, one prompt distribution, 32-token greedy, repeated-text timing;
not a general quality claim.

Next: fuse the dequantize step into the attention kernel (remove the extra
launches the event span exposes) and re-measure; then consider re-estimating
`k_scale` over the full history.

## Fused dequantize (2026-10-10)

`nanovllm/sparse/fused_dequant.py` replaces the eager
`hist.float() * scale -> to(bf16) -> packed` (several ops + a
`repeat_interleave` per layer) with one Triton kernel writing both packed K
and V (bit-exact vs the torch path; `bench_logs/lineA/test_fused_dequant.py`).
Gated by `sparse_fused_dequant` (requires quant history).

32K, Top-32, 3 pairs:

| | steady TPOT | H2D device span |
|---|---:|---:|
| torch dequant | 91.8 | 34.8 |
| fused dequant | 85.0 | 26.8 |

H2D span -8.0 ms; paired TPOT deltas 9.9 / -2.0 / 12.5 (mean -6.8). Run-to-run
variance is high (80.8-93.2), so the win is real but noise-level. Generated
token IDs identical across all arms.

## Cumulative decode result at 32K, Top-32

| config | TPOT | vs baseline |
|---|---:|---:|
| baseline (bf16, torch dequant) | 113.5 | - |
| int8 history | 92.5 | -18.5% |
| int8 + fused dequant | ~85 | ~-25% |

## Why cross-layer gather prefetch is infeasible

The idea (overlap layer L's CPU gather with layer L-1's H2D/attention) is
architecturally blocked, not merely hard. The per-layer dependency is:

```
h_{L-1} (layer L-1 output, incl. its MLP)
  -> q_L = qkv_proj(h_{L-1})            (nanovllm/models/qwen3.py:85-98)
     -> selector(q_L) -> block ids_L
        -> gather_L -> H2D_L -> attention_L   (attention.py:97)
```

Layer L's gather target is decided by q_L, which is produced from layer L-1's
full output. So while layer L-1 computes, layer L's rows are not yet known and
cannot be prefetched. There is no independent work to overlap across layers.

The only overlappable units are **within a layer**: K-gather vs V-gather
(attempted in M20 as the K/V copy pipeline; no end-to-end gain), or the tiny
sink/recent packing. H2D itself is a dead end within the layer: attention_L
waits on H2D_L, which waits on gather_L, which waits on ids_L.

The only "prefetch" that is even possible uses stale information (previous
layer's q, or the previous token's blocks) and then corrects. M13 measured
adjacent-layer/token block reuse at **32.2%**, so ~68% of prefetched rows
would be wasted plus an extra H2D: expected negative. Therefore the way to
reduce H2D is fewer bytes (int8, already done; int4 next), not overlap.


## Known limitations / open items

- `k_scale` is fixed from the first chunk; later K values outside its range
  clip. A full-history re-estimate or per-block K scales are future options.
- Dequantization is a separate step (several torch ops per layer); fusing it
  into the attention kernel is the next step and removes those launches.
- int8 only; int4 is not attempted.
- Incompatible with the segmented prefill adapter (asserted).
