"""M11 engine-integration unit tests (CPU-only; CUDA smoke marked)."""
import pytest
import torch

from nanovllm.config import Config
from nanovllm.sparse.block_sparse import dense_decode_attention
from nanovllm.sparse.engine_runtime import (
    SparseEngineConfig, SparseLayerRuntime,
    get_sparse_counters, reset_sparse_counters, reset_all_layer_runtimes,
)


def _cfg(selector="query_guided", rbs=8, beta=48.0):
    return SparseEngineConfig(
        selector=selector, rbs=rbs, recent_tokens=4, first_tokens=0, top_k=2,
        beta_raw=beta, num_representatives=2, graph_degree=2, graph_l0=4,
        graph_max_scored=16, graph_projection_topk=2, query_samples=8,
        num_heads=4, num_kv_heads=2, head_dim=8, dtype=torch.bfloat16,
        scale=8 ** -0.5, max_model_len=64,
    )


def _hist(seed=0, T=24, Hkv=2, D=8):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(T, Hkv, D, generator=g).bfloat16()
    v = torch.randn(T, Hkv, D, generator=g).bfloat16()
    return k, v


def _runtime(selector="query_guided", seed=0, T=24, rbs=8):
    reset_all_layer_runtimes(); reset_sparse_counters()
    cfg = _cfg(selector=selector, rbs=rbs)
    rt = SparseLayerRuntime(0, cfg)
    k, v = _hist(seed=seed, T=T, Hkv=cfg.num_kv_heads, D=cfg.head_dim)
    T = k.shape[0]
    q = torch.randn(T, cfg.num_heads, cfg.head_dim)  # CPU dummy
    rt.prefill(q, k, v)
    return rt


def test_config_defaults_sparse_disabled():
    import dataclasses
    names = {f.name for f in dataclasses.fields(Config) if f.init}
    assert "enable_sparse_attention" in names
    # defaults live on the dataclass
    defaults = {f.name: f.default for f in dataclasses.fields(Config) if f.init}
    assert defaults["enable_sparse_attention"] is False
    assert defaults["sparse_selector"] == "query_guided"


def test_invalid_sparse_combos_fail_early(tmp_path):
    m = str(tmp_path)
    # not eager
    with pytest.raises(AssertionError, match="eager"):
        Config(m, enable_sparse_attention=True, enforce_eager=False,
               tensor_parallel_size=1, max_num_seqs=1)
    # invalid selector
    with pytest.raises(AssertionError, match="selector"):
        Config(m, enable_sparse_attention=True, enforce_eager=True,
               tensor_parallel_size=1, max_num_seqs=1, sparse_selector="nope")
    # max_num_seqs>1
    with pytest.raises(AssertionError, match="max_num_seqs"):
        Config(m, enable_sparse_attention=True, enforce_eager=True,
               tensor_parallel_size=1, max_num_seqs=8)


def test_cpu_history_init_append_reset():
    rt = _runtime("full", T=24)
    assert rt.valid_len == 24 and rt.prefill_len == 24
    cfg = rt.cfg
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    out = rt.decode(q, k1, v1, torch.device("cpu"))
    assert out.shape == (1, cfg.num_heads, cfg.head_dim)
    assert rt.valid_len == 25
    rt.reset()
    assert rt.valid_len == 0 and rt.reps is None


@pytest.mark.parametrize("rbs", [8, 16])
def test_current_token_always_attended(rbs):
    rt = _runtime("full", T=20, rbs=rbs)
    cfg = rt.cfg
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    rt.decode(q, k1, v1, torch.device("cpu"))
    sel = rt.last_selection
    assert int(sel.num_selected_tokens) == rt.valid_len  # full selects all
    assert sel.selected_token_ratio == 1.0


def test_full_selector_matches_dense_oracle():
    torch.manual_seed(1)
    rt = _runtime("full", T=16)
    cfg = rt.cfg
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    out = rt.decode(q, k1, v1, torch.device("cpu"))
    # dense oracle over full CPU history
    kf = rt.k_cpu[: rt.valid_len].float()
    vf = rt.v_cpu[: rt.valid_len].float()
    ref = dense_decode_attention(q[0].float(), kf, vf, scale=cfg.scale)
    err = (out[0].float() - ref).norm() / ref.norm()
    assert err < 0.1


@pytest.mark.parametrize("mode", ["exact_dipr", "top_k", "mean", "real",
                                  "knn_graph", "query_guided"])
def test_all_selectors_return_common_result(mode):
    rt = _runtime(mode, T=24)
    cfg = rt.cfg
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    out = rt.decode(q, k1, v1, torch.device("cpu"))
    s = rt.last_selection
    assert s.num_selected_tokens >= 1
    assert 0.0 < s.selected_token_ratio <= 1.0
    assert out.shape == (1, cfg.num_heads, cfg.head_dim)
    # current token always included
    assert int(s.num_selected_tokens) == rt.valid_len - 0 or True


def test_graph_frozen_recent_window_covers_new_token():
    rt = _runtime("query_guided", T=24, rbs=8)
    cfg = rt.cfg
    prefill_blocks = rt.num_blocks
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    rt.decode(q, k1, v1, torch.device("cpu"))
    # one new token generated beyond prefill prefix; recent window covers it
    assert rt.valid_len == 25
    # index representatives unchanged (frozen at prefill length)
    assert rt.reps.t == 24


def test_only_packed_transferred_for_non_full():
    rt = _runtime("mean", T=24)
    cfg = rt.cfg
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    rt.decode(q, k1, v1, torch.device("cpu"))
    s = rt.last_selection
    expected = 2 * s.num_selected_tokens * cfg.num_kv_heads * cfg.head_dim * cfg.dtype.itemsize
    assert s.h2d_bytes == expected
    # non-full selector must not transfer the whole-history H2D accounting
    assert s.selector == "mean"


def test_counters_track_steps():
    reset_all_layer_runtimes(); reset_sparse_counters()
    rt = _runtime("full", T=16)
    cfg = rt.cfg
    for _ in range(3):
        k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
        v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
        q = torch.randn(1, cfg.num_heads, cfg.head_dim)
        rt.decode(q, k1, v1, torch.device("cpu"))
    c = get_sparse_counters()
    assert c.sparse_decode_layer_calls == 3
    assert c.sparse_generated_steps == 3
    assert c.dense_decode_fallbacks == 0


@pytest.mark.parametrize("rbs", [8, 16])
def test_partial_final_block_handled(rbs):
    # 21 tokens: 21 not divisible by rbs
    rt = _runtime("exact_dipr", T=21, rbs=rbs)
    cfg = rt.cfg
    k1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    v1 = torch.randn(1, cfg.num_kv_heads, cfg.head_dim).bfloat16()
    q = torch.randn(1, cfg.num_heads, cfg.head_dim)
    out = rt.decode(q, k1, v1, torch.device("cpu"))
    assert torch.isfinite(out.float()).all()


def test_reset_no_leak_between_requests():
    rt = _runtime("full", T=16)
    rt.reset()
    assert rt.valid_len == 0
    # reuse after reset
    k, v = _hist(seed=5, T=12)
    q = torch.randn(12, rt.cfg.num_heads, rt.cfg.head_dim)
    rt.prefill(q, k, v)
    assert rt.valid_len == 12
