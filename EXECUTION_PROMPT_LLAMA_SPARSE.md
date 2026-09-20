# EXECUTION_PROMPT: Llama adaptation + block-level sparse attention

This document is the handoff for the next phase of NanoKV after Milestone 7
(the CPU-backed prefix reuse project). It records every design decision made
while planning, so a fresh agent can pick up without re-litigating them.

## 0. Where things stand right now

- Repo: /opt/nano-vllm in WSL distribution NanoVLLM-Ubuntu
- Virtualenv: /opt/nano-vllm/.venv (use .venv/bin/python directly; do NOT
  `source .venv/bin/activate` through `bash -lc` -- that path has broken
  stdout capture in this environment)
- Env exports: CUDA_HOME=/usr/local/cuda-12.8,
  HF_HOME=/opt/models/.cache/huggingface
- GPU: NVIDIA GeForce RTX 3080 Laptop, 16 GiB. Host RAM: 32 GiB.
- Last commit: 39c5d7e "milestone 7: complete NanoKV project delivery"
- Main is 9 commits ahead of origin (origin = upstream GeeeekExplorer/nano-vllm,
  NO push permission there). Nothing is pushed to a personal remote yet.
- `python -m pytest -q` = 28 passed. `git diff --check` clean.
- Current model tested: /opt/models/Qwen3-0.6B.

NanoKV as it stands = CPU KV offload (pageable + pinned, LRU), longest-prefix
lookup across GPU and CPU tiers, hash-collision-safe matching, non-aligned tail
never reused, transactional CPU->GPU restore with full-prefill fallback,
per-request metrics (lookup/H2D/prefill/TTFT), and a 5-mode benchmark.
It is in the "KV cache disaggregation" category (like Mooncake/LMCache), NOT
sparse attention.

## 1. Phase goal

Two sub-goals, in order:

1.1 Add a Llama model adapter so the engine can run Llama weights.
1.2 Add block-level sparse attention with a DIPR-style dynamic selection rule,
    targeting long-context (>=128k) workloads.

Do NOT do any of these now:
- No int4/int8 quantization (dropped; we will run bf16 and rely on sparse
  attention to fit long context, not on weight quantization).
- No DeepSeek (it needs MLA attention + MoE FFN; both are architecturally
  foreign to nano-vllm. Out of scope for this phase).
- No token-level graph index / RoarGraph on individual key vectors. We do
  block-level representatives first.
- No pushing to any remote until the user explicitly says which remote.

## 2. Llama adaptation -- decisions locked

- Target model for first bring-up: Llama-3.2-3B (small, fast to download and
  iterate). After it runs clean, move to Llama-3.1-8B as the real sparse-
  attention target.
- Reason: Llama and Qwen3 are the same family (RMSNorm + SwiGLU + RoPE
  decoder-only, GQA). The adaptation is mechanical, not architectural.
- Differences to handle vs models/qwen3.py:
  * use transformers.LlamaConfig instead of Qwen3Config;
  * no q_norm / k_norm (standard Llama does not have them; Qwen3 adds them
    when there is no attention bias);
  * attention has no bias;
  * tie_word_embeddings is read from config (usually false for Llama);
  * RoPE is standard full head_dim rotation;
  * packed_modules_mapping maps HF q/k/v_proj -> fused qkv_proj, gate/up_proj
    -> fused gate_up_proj, same as Qwen3.
- block_size stays 256. Do NOT change kvcache_block_size in this phase. The
  existing GPU block manager, Triton store kernel, and FlashAttention paged
  path all assume it.
- Estimated effort: one working day. Main risk area is the attention/rope
  wiring and the safetensors weight-name mapping.

## 3. Sparse attention -- design locked

Context length target: 128k as the first runnable milestone (Llama-3.1-8B is
natively 128k, max_position_embeddings=131072). Architecture must NOT hardcode
128k -- leave the door open to 256k later (which would require RoPE
extrapolation / YaRN; that is a separate follow-up, not now).

KV size math for Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, bf16):
- one 256-token block = 2 * 32 * 256 * 8 * 128 * 2 bytes = 32 MiB
- 128k context = 512 blocks = 16 GiB of KV (too big for GPU, fits in CPU RAM)
- 256k context = 1024 blocks = 32 GiB (already tight on 32 GiB host RAM)

Design (block-level, DIPR-inspired, NOT token-level):

1. Keep GPU block_size = 256. Do not introduce smaller blocks on the GPU side.
2. When a block of KV is evicted to CPU (existing CPUBlockStore path), also
   store a per-block representative key vector. Start with the simple
   "mean of the block's key vectors" as the representative; revisit later if
   quality demands max-pooling or a learned representative.
3. Build a graph index (start with hnswlib or a flat brute-force search for
   bring-up; do not write a custom HNSW) over these per-block representative
   vectors. Index per attention head / per layer as needed.
4. On each decode step, for each layer/head, use the current query to find
   the relevant blocks via the index. Selection rule is DIPR-style, NOT fixed
   top-k:
   - find the representative with max inner product for the current query;
   - select every block whose representative inner product is within beta of
     that max (inner-product range query);
   - beta is a tunable constant; start with a sweep (paper used beta=50 to 110
     on 128-dim single-token keys, but block-representative inner products
     are smaller/distribution-shifted, so DO NOT copy 50 blindly -- sweep it).
5. Window cache: the first 32 and last 32 tokens stay in GPU KV unconditionally
   (paper observation: ~98% of max-inner-product keys fall in these windows).
6. Layer policy: the first layer needs far more tokens than later layers
   (paper: layer 0 ~43k tokens vs layer 31 ~53 tokens at 90% recovery). Start
   by making layer 0 select all blocks (or a much larger beta) and use the
   DIPR rule for the rest.
7. Only the selected blocks' full K/V are streamed CPU->GPU for the attention
   math; the unselected blocks stay on CPU. The attention kernel must consume
   a scattered/paged subset of blocks. This is the hardest part -- do it last
   and benchmark a fixed-block version before the DIPR version.

Out of scope for this phase:
- token-level index (block representative first);
- MLA, MoE, DeepSeek;
- weight quantization;
- async/overlapped CPU-GPU decode pipeline (that is "phase 03" in the
  AlayaJet-style roadmap, after 04 works).

## 4. Reading the AlayaDB paper

Paper: https://arxiv.org/html/2504.10326v1 (AlayaDB).

What we actually take from it:
- The DIPR definition (Definition 6.2): a key k_j is "critical" for query q iff
  q.k_j >= max_s(q.k_s) - beta. This is mathematically equivalent to
  softmax-proportion gating but computed in inner-product space, so no
  softmax pass is needed to decide selection.
- The DIPRS graph-search loop (Algorithm 1): candidate list grows until a
  capacity threshold l0, then prunes points that fall outside the beta range.
- Window caching (first/last 32 tokens resident).
- Layer-dependent selection policy (layer 0 needs many more tokens).

What we do NOT take:
- The full vector-database product (query optimizer, vector file system,
  distributed service). We are a single-process, embedded library.
- RoarGraph (we use hnswlib or flat search first).
- The product numbers in the AlayaJet slides (6.84x throughput etc.) -- those
  are marketing material, not our expected results.

## 5. Acceptance criteria for this phase

After Llama adaptation:
- `python -m pytest -q` still 28 passed.
- A Llama-3.2-3B greedy generation on a short prompt produces sensible text
  and matches a HuggingFace Transformers reference run for the same prompt.
- `git diff --check` clean; commit message "add Llama model adapter".

After sparse attention (separate later commit):
- On a 128k-context prompt, end-to-end runs without OOM.
- Generated text quality on a long-context task (e.g. needle-in-haystack)
  does not degrade materially vs full attention on a shorter context.
- A benchmark compares: full-attention (OOM at 128k, expected), block-DIPR,
  and a fixed-top-k-block baseline, reporting TTFT/TPOT and quality.

## 6. Operational notes for the next agent

- Use `wsl.exe -d NanoVLLM-Ubuntu -- bash -c "..."` from PowerShell.
- For Python snippets with quotes, base64-encode the script on the Windows
  side and `echo <b64> | base64 -d > /tmp/x.py` inside WSL, then run it.
  Avoid nested double quotes -- they break under PowerShell->WSL.
- The NCCL/tcp port 2333 may be in TIME_WAIT after a crash; wait a few
  seconds before retrying if you see EADDRINUSE.
- Do NOT reinstall PyTorch / CUDA / Triton / FlashAttention / models inside
  Windows. All work happens inside WSL.
- The current NanoKV version should get a git tag before the Llama work
  starts, so it is easy to roll back. Suggested tag: `nanokv-m7`.