# M18: Nsight diagnosis and default pinned gather

Status: local experiment on `codex/m18-nsight-profile`. The only runtime
behavior change is making the existing direct `index_select(..., out=pinned)`
CPU gather the M12 default. `sparse_gather_index_select=False` retains the
legacy advanced-indexing path for controlled comparisons. Prefill/decode
Top-K, retrieval, attention semantics, and model weights are unchanged.

## Why this optimization, not another prefill Top-K cut

M17 measured diminishing prefill gains below Top-24 and a matched multi-key
regression at Top-24/16. We therefore kept prefill and decode Top-K at 32.
The M18 Nsight Systems captures used Qwen3-4B snapshot
`1cfa9a7208912126459214e8b04321603b3df60c`, BF16, YaRN, RTX 3080
Laptop 16 GB, 4096-token prefill chunks, one query summary, and FA2 prefill.

One 16K later-chunk prefill capture recorded approximately 747 ms of the two
dominant GEMM kernel families, 313 ms in FA2 forward, 177 ms device-to-host
KV copies, 25 ms host-to-device copies, and 4 ms device-to-device copies.
These are device activity totals in an instrumented chunk; they are not
exclusive shares of normal wall time and may overlap. In particular, this
does not support prioritizing the small prefill pack-copy path over the model
compute, nor does it establish that prefill has reached a general limit.

One 32K decode-token capture with the legacy gather showed an NVTX step
range of 231.65 ms. Across 36 layers, CPU-side NVTX ranges summed to 46.06
ms gather, 46.36 ms selector, 27.95 ms packed attention, 17.90 ms H2D pack,
and 13.07 ms recent update. The trace recorded about 35.20 ms of H2D
device activity for the 288 MiB selected-KV payload. Packed attention
launched 288 kernels with 6.53 ms total device kernel time. NVTX host ranges
and device activities **must not be added** as a wall-time decomposition:
H2D is asynchronous and selector ID `.cpu()` can wait for earlier GPU work.
The single capture is diagnostic, not a speed comparison.

## Unprofiled, matched decode A/B

Six fresh-process 32K/32-generation runs alternated legacy/direct-gather
order as `0,1,1,0,0,1`. Both sides used fixed prefill and decode Top-32,
same repeated-text prompt, model snapshot, YaRN, and greedy decoding. The
benchmark synchronizes CUDA around each engine step; steady decode drops
the first four decode steps. Each pair's 32 generated token IDs were exactly
equal. This is a functional equivalence check on one prompt, not a general
quality score.

| Pair | Legacy steady median (ms/token) | Direct gather (ms/token) | Paired reduction | Legacy/direct last-step CPU gather (ms, 36 layers) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 162.76 | 140.59 | 13.6% | 34.74 / 28.04 |
| 2 | 191.41 | 155.50 | 18.8% | 45.83 / 26.46 |
| 3 | 175.84 | 155.10 | 11.8% | 44.40 / 33.35 |

The mean of the three per-run steady medians was 176.67 versus 150.40
ms/token; the mean paired relative reduction was 14.7%. All three matched
pairs favored direct gather, but run-to-run variation remains substantial.
Report these raw pairs rather than advertising the single best 18.8% as a
stable gain. This evidence supports promoting the already-tested gather
path to the M12 default. It does **not** establish an end-to-end prefill
improvement; M17's prefill gather-stage gain did not have a reliable whole
prefill signal.

The direct-gather trace, collected separately for diagnosis, had an NVTX
step range of 182.49 ms and 33.17 ms of gather NVTX host range across 36
layers. Do not compare 182.49 directly with the legacy trace's 231.65 as
an A/B result: Nsight instrumentation and WSL scheduling can perturb both.

One additional unprofiled 64K/16-generation pair used the same model and
fixed Top-32. Legacy/direct steady medians were 228.78/182.82 ms/token
(20.1% paired reduction), steady means 229.31/186.19 ms/token, and last-step
36-layer CPU gather 68.54/34.14 ms. All 16 generated token IDs matched.
Prefill wall was 30.29/28.22 s, but one pair is far too little evidence for
a prefill speedup claim. The 64K pair is a scaling/correctness guard, not a
stable effect estimate.

## Reproduction and next measurement

NVIDIA Nsight Systems CLI and GUI 2026.5.1 are installed in
`NanoVLLM-Ubuntu`; the GUI can display through WSLg. The GUI was launched
with `nsys-ui bench_logs/m18_decode_index_32k_step4.nsys-rep` and its process
remained running with that report path. `nsys status --environment` reported
CPU process-tree profiling OK. The benchmark scripts provide optional
`--nsys-trace-step` (decode) and
`--nsys-step` (prefill) markers; normal runs do not enable NVTX ranges or
cudaProfilerApi capture. To reproduce a single decode trace:

```bash
cd /opt/nano-vllm
source .venv/bin/activate
export HF_HOME=/opt/models/.cache/huggingface
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o bench_logs/m18_repeat_decode \
  python benchmarks/benchmark_m15_decode.py --seq-len 32768 \
    --gen-tokens 8 --index-select 1 --nsys-trace-step 4 \
    --output bench_logs/m18_repeat_decode.json
nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_kern_sum,nvtx_sum \
  bench_logs/m18_repeat_decode.nsys-rep
nsys-ui bench_logs/m18_repeat_decode.nsys-rep
```

The existing `--index-select 0/1` benchmark switch explicitly overrides
the new default for A/B runs. Raw ignored artifacts are
`bench_logs/m18_decode_ab_{1,2,3}_{0,1}.json`,
`m18_decode_64k_{0,1}.json`,
`m18_decode_32k_step4.nsys-rep`,
`m18_decode_index_32k_step4.nsys-rep`, and
`m18_prefill_16k_step1.nsys-rep` (plus associated JSON/log/SQLite files).

After this change, the next candidate is **not yet an optimization claim**:
measure whether decode selector synchronization / CPU scheduling remains the
dominant unoverlapped interval with direct gather enabled. A FlashAttention-2
decode backend is a possible isolated prototype, but the legacy trace gives
only 6.53 ms of attention device kernels per token, so its benefit must be
demonstrated with a separate unchanged-Top-K, unprofiled A/B and token/ID
correctness check. Do not mix that experiment with this default promotion.

Remaining evidence in priority order: (1) a repeated 64K pair and a
representative non-repeated prompt before claiming the same percentage
outside the 32K repeated-text workload; (2) a matched long-context quality
benchmark and 128K memory/correctness regression before making new quality
or 128K claims; (3) an unprofiled A/B for any future selector or overlap
prototype. Nsight Compute hardware counters are only warranted once a
specific GPU kernel, rather than the CPU/GPU pipeline, is the target.

The Windows-native GUI installer requires an administrator account on this
machine and was not installed. This is optional: the WSLg GUI and CLI are
both operational, and all traces were collected inside the authoritative
WSL environment.

Validation for the change: 19 focused M14/M16/M17 tests passed; all three
32-token A/B pairs and the 16-token 64K pair had identical generated token
IDs.
