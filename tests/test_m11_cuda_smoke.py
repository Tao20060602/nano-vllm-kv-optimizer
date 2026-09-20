"""CUDA integration smoke test (skipped without CUDA/model)."""
import os
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA")
MODEL = "/opt/models/Qwen3-0.6B"
pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL), reason="needs local model")


def test_cuda_smoke_sparse_generates_three_tokens():
    from nanovllm import LLM, SamplingParams
    os.environ.setdefault("HF_HOME", "/opt/models/.cache/huggingface")
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              max_num_seqs=1, max_model_len=1024,
              enable_sparse_attention=True, sparse_selector="query_guided")
    llm.sparse_reset()
    sp = SamplingParams(max_tokens=5, temperature=0.0, ignore_eos=True)
    out = llm.generate(["The sun rises in the"], [sp], use_tqdm=False)
    assert out[0]["token_ids"]
    c = llm.sparse_counters()
    # >=2 actual sparse decode iterations through the shared path
    assert c["sparse_generated_steps"] >= 2
    assert c["dense_decode_fallbacks"] == 0
    assert c["sparse_decode_layer_calls"] >= 2 * 1
    llm.exit()
