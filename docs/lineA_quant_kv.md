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

## Engine A/B (pending at time of writing)

32K, Top-32, fresh-process ABBA baseline vs `--quant-history`, with CUDA
stage events. Report: steady TPOT, H2D device span, generated-token equality,
and the 20-row 8K multikey quality screen.

## Known limitations / open items

- `k_scale` is fixed from the first chunk; later K values outside its range
  clip. A full-history re-estimate or per-block K scales are future options.
- Dequantization is a separate step (several torch ops per layer); fusing it
  into the attention kernel is the next step and removes those launches.
- int8 only; int4 is not attempted.
- Incompatible with the segmented prefill adapter (asserted).
