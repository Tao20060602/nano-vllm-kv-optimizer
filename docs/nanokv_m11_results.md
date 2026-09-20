# NanoKV M11 Results — Sparse Generation Engine Integration

Hardware/software: RTX 3080 Laptop (sm86, 16 GB), CUDA 12.8, torch 2.7.1+cu128,
Qwen3-0.6B (28 layers, 16 q-head / 8 KV-head, head_dim=128, bf16), WSL2 Ubuntu.
Single sequence, eager, TP=1, greedy, **exactly 2048-token prompt** (tokenizer-verified),
16 completion tokens, max_model_len=2068.

## Integration proof

- Sparse prefill runs dense FlashAttention, then each layer copies post-RoPE K/V
  to CPU and builds its frozen index.
- Sparse decode appends the current token's post-RoPE K/V to CPU, selects blocks,
  gathers only the packed selection to pinned staging, H2D-copies it, and runs
  packed PyTorch attention whose output flows through the shared Attention class
  into o_proj and the sampler.
- Counters: 28 layer prefill inits, 15 decode steps, 420 decode layer calls,
  `dense_decode_fallbacks=0`.
- `flash_attn_with_kvcache` is never called in enabled sparse decode; the paged
  KV cache tensor is exactly **0 bytes** in sparse mode.

## Config / beta

- beta_raw = 48.0 (authoritative raw inner product);
  beta_scaled_logit = 48/sqrt(128) = 4.24.
- retrieval block size = 64, recent window = 128, r = 4 representatives,
  query_samples per layer.

## Selector comparison (layer 14 sample, 2048-token history)

| selector | selected ratio | pre-window recall vs exact oracle | token agree vs dense | p50 TPOT (ms) |
|---|---|---|---|---|
| dense baseline (paged FlashAttention) | 1.0 | n/a | 1.00 | ~35 |
| full (CPU offload) | 1.00 | 1.00* | 1.00 | 508 |
| exact Block-DIPR | 1.00 | 1.00 | 1.00 | 643 |
| top_k (k=8) | 0.91 | 0.91 | 1.00 | 747 |
| mean representative | 0.97 | ~1.0 | 1.00 | 938 |
| r=4 real representative | 0.41 | ~1.0 | 1.00 | 529 |
| KNN graph | 0.38 | ~1.0 | 1.00 | 1110 |
| query-guided graph | 0.38 | ~1.0 | 1.00 | 1299 |

\* full selects all blocks. Recall vs exact oracle is 1.0 for exact/top_k by
construction; representative/graph rows retain all oracle blocks here.

All sparse runs produced token-for-token agreement with the dense greedy output.
TPOT is **higher** than the paged FlashAttention baseline because this is a
synchronous, Python-level CPU gather + selector traversal over a 2048-token
history — no asynchronous overlap, no custom kernel. This is a functional,
honest integration, not a speed claim.

## Memory (architectural)

- dense paged GPU KV cache: 12.86 GB (engine reserves a large paged pool).
- sparse paged GPU KV cache: **0 bytes** (no physical blocks allocated).
- sparse CUDA allocated after init: 1.22 GB (model weights only).
- allocated CPU history buffer: **~226 MiB across 28 layers** at capacity 2068
  tokens (pageable, not pinned). Note: at an 8192-token capacity the linear
  buffer would be ~896 MiB; the number reported here is the actual 2068-token
  run, not the 8192-token projection.
- packed active K/V per decode is the only transient GPU transfer.

## Block-size offline replay (real layer-14 K/V, mean selector)

| rbs | blocks | selected ratio | recall vs exact | refined pairs |
|---|---|---|---|---|
| 32 | 26 | 1.00 | 1.00 | 307 |
| 64 | 13 | 1.00 | 1.00 | 172 |
| 128 | 7 | 1.00 | 1.00 | 94 |
| 256 | 4 | 1.00 | 1.00 | 56 |

Larger blocks reduce edges/refinement work but coarsen selection.

## Baseline wording

The M11 dense reference is the **feature-off, real paged FlashAttention engine**
(upstream kernel), not the M9/M10 `dense_decode_attention` PyTorch replay. The
packed GPU attention used in sparse decode is a PyTorch implementation; the
FlashAttention engine output is the traced production baseline.

## Limitations

Dense prefill only (must fit GPU), single sequence, eager TP=1, frozen graph at
prefill (new tokens via recent window), synchronous transfers, no arbitrary-long
context claim, simplified query-guided graph (not full RoarGraph).
