#!/usr/bin/env bash
# One-command NanoKV M11 demo: dense baseline vs sparse generation.
set -e
cd /opt/nano-vllm
[ -f venv ] || true
VENV=/opt/nano-vllm/.venv/bin/python
[ -x "$VENV" ] || { echo "expected venv at /opt/nano-vllm/.venv"; exit 1; }
export HF_HOME=/opt/models/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-12.8
[ -d /opt/models/Qwen3-0.6B ] || { echo "missing local model"; exit 1; }

echo "=== NanoKV M11 demo: dense baseline vs query-guided sparse ==="
$VENV - <<'PY'
import os, time
import torch
from nanovllm import LLM, SamplingParams
SP = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
PROMPT = "The secret passkey is 74291. The Eiffel Tower is in Paris. What is the secret passkey?"

dense = LLM("/opt/models/Qwen3-0.6B", enforce_eager=True, max_num_seqs=1,
            max_model_len=2048)
t0=time.perf_counter(); o=dense.generate([PROMPT],[SP],use_tqdm=False)[0]; dense_t=(time.perf_counter()-t0)*1000
print(f"DENSE  ({dense_t:.0f}ms): {o['text']}")
dense.exit()

sparse = LLM("/opt/models/Qwen3-0.6B", enforce_eager=True, max_num_seqs=1,
             max_model_len=2048, enable_sparse_attention=True,
             sparse_selector="query_guided", sparse_beta_raw=48.0)
sparse.sparse_reset()
t0=time.perf_counter(); o=sparse.generate([PROMPT],[SP],use_tqdm=False)[0]; spt=(time.perf_counter()-t0)*1000
print(f"SPARSE ({spt:.0f}ms): {o['text']}")
sel = sparse.sparse_sample_selection(14)
print("selected_token_ratio:", round(sel.get('selected_token_ratio',0),3) if sel else None)
print("counters:", sparse.sparse_counters())
print("cpu history bytes:", sparse.sparse_history_bytes())
print("paged kv bytes: 0 (sparse mode), CUDA allocated:", torch.cuda.memory_allocated())
sparse.exit()
PY
echo "=== demo complete ==="
