import pytest
import torch

from nanovllm.kvdb.store.gpu import GPUBlockStore
from nanovllm.kvdb.types import CacheFingerprint, KVBlockPayload


def _fingerprint(**overrides):
    values = {
        "model_id": "/model",
        "config_digest": "config",
        "dtype": "torch.float32",
        "num_layers": 2,
        "num_kv_heads": 1,
        "head_dim": 2,
        "block_size": 4,
        "tensor_parallel_size": 1,
        "rope_theta": 1_000_000.0,
        "rope_scaling": "null",
    }
    values.update(overrides)
    return CacheFingerprint(**values)


def test_gpu_store_wraps_physical_block_dimension_without_copying():
    kv_cache = torch.arange(2 * 2 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
        2, 2, 3, 4, 1, 2
    )
    store = GPUBlockStore(kv_cache, _fingerprint())

    payload = store.read_block(1)
    assert payload.tensor.data_ptr() == kv_cache[:, :, 1].data_ptr()
    assert payload.tensor.shape == (2, 2, 4, 1, 2)
    assert store.capacity_blocks == 3
    assert store.bytes_per_block == 2 * 2 * 4 * 1 * 2 * 4


def test_gpu_store_write_validates_fingerprint_and_layout():
    kv_cache = torch.zeros(2, 2, 3, 4, 1, 2)
    fingerprint = _fingerprint()
    store = GPUBlockStore(kv_cache, fingerprint)
    source = torch.ones(2, 2, 4, 1, 2)

    store.write_block(2, KVBlockPayload(fingerprint, source))
    assert torch.equal(kv_cache[:, :, 2], source)
    assert torch.count_nonzero(kv_cache[:, :, :2]) == 0

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        store.write_block(
            1,
            KVBlockPayload(_fingerprint(config_digest="other"), source),
        )
    with pytest.raises(ValueError, match="payload shape"):
        store.write_block(1, KVBlockPayload(fingerprint, source[:, :, :-1]))
    with pytest.raises(IndexError, match="out of range"):
        store.read_block(3)


def test_gpu_store_rejects_tensor_fingerprint_mismatch():
    kv_cache = torch.zeros(2, 2, 3, 4, 1, 2)

    with pytest.raises(ValueError, match="KV layout"):
        GPUBlockStore(kv_cache, _fingerprint(block_size=8))
    with pytest.raises(ValueError, match="KV dtype"):
        GPUBlockStore(kv_cache, _fingerprint(dtype="torch.bfloat16"))
