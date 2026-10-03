"""Exact staged KV and decode parity for same-layer K-DMA/V-gather overlap."""

import pytest
import torch

from nanovllm.sparse.m12_runtime import M12Config, M12LayerRuntime


def test_pipeline_requires_direct_pinned_gather():
    with pytest.raises(ValueError, match="requires use_index_select"):
        M12Config(use_index_select=False, decode_kv_pipeline=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU runtime")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pipeline_preserves_ids_packed_kv_and_outputs(dynamic, graph, dtype):
    torch.manual_seed(20)
    history_k = torch.randn(48, 1, 4, device="cuda", dtype=dtype)
    history_v = torch.randn_like(history_k)
    queries = torch.randn(4, 1, 2, 4, device="cuda", dtype=dtype)
    new_keys = torch.randn(4, 1, 1, 4, device="cuda", dtype=dtype)
    new_values = torch.randn_like(new_keys)
    runs = []
    for pipeline in (False, True):
        cfg = M12Config(
            block_size=2, r=1, recent_tokens=2, sink_tokens=2,
            top_k_blocks=8, decode_top_k_blocks=6, max_model_len=64,
            num_heads=2, num_kv_heads=1, head_dim=4, dtype=dtype,
            dynamic_top_k=dynamic, selector_cuda_graph=graph,
            decode_kv_pipeline=pipeline,
        )
        rt = M12LayerRuntime(0, cfg)
        rt.prefill_first(history_k, history_k, history_v)
        pointers = (rt.stage_k.data_ptr(), rt.stage_v.data_ptr())
        steps = []
        for q, k, v in zip(queries, new_keys, new_values):
            # Previous output's .cpu() below completes the current stream, so
            # these sentinel writes cannot race a previous pinned H2D read.
            rt.stage_k.fill_(-77)
            rt.stage_v.fill_(-88)
            out = rt.decode(q, k, v, q.device)
            total = rt.timings["total_attend_tokens"]
            tokens = rt.timings["selected_tokens"]
            steps.append((out.cpu(), rt.last_block_ids.clone(),
                          rt.packed_k[:total].cpu(), rt.packed_v[:total].cpu()))
            assert (rt.stage_k.data_ptr(), rt.stage_v.data_ptr()) == pointers
            assert rt.stage_k.is_pinned() and rt.stage_v.is_pinned()
            assert bool((rt.stage_k[tokens:] == -77).all())
            assert bool((rt.stage_v[tokens:] == -88).all())
            assert rt.timings["h2d_event_spans_cpu_gather"] == pipeline
            assert rt.timings["cpu_gather_ms"] >= 0
            assert rt.timings["h2d_pack_ms"] >= 0
        runs.append(steps)
    for serial_step, pipelined_step in zip(*runs):
        for reference, candidate in zip(serial_step, pipelined_step):
            assert torch.equal(reference, candidate)
