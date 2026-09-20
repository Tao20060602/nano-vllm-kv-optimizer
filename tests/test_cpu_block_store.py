import pytest
import torch

from nanovllm.kvdb.store.cpu import CPUBlockStore, StaleCPUBlockHandle
from nanovllm.kvdb.types import CacheFingerprint, CacheKey, KVBlockPayload


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


def _tensor(fingerprint, value=0):
    dtype = getattr(torch, fingerprint.dtype.removeprefix("torch."))
    shape = (
        2,
        fingerprint.num_layers,
        fingerprint.block_size,
        fingerprint.num_kv_heads,
        fingerprint.head_dim,
    )
    return torch.full(shape, value, dtype=dtype)


def _store(fingerprint=None, blocks=2, **kwargs):
    fingerprint = fingerprint or _fingerprint()
    bytes_per_block = _tensor(fingerprint).numel() * _tensor(fingerprint).element_size()
    return CPUBlockStore(
        fingerprint,
        capacity_bytes=blocks * bytes_per_block,
        **kwargs,
    )


@pytest.mark.parametrize(
    "fingerprint",
    [
        _fingerprint(dtype="torch.float32", num_layers=1),
        _fingerprint(dtype="torch.float16", num_layers=2, num_kv_heads=2),
        _fingerprint(dtype="torch.bfloat16", head_dim=4),
    ],
)
def test_pageable_round_trip_preserves_layers_kv_dtype_and_shape(fingerprint):
    store = _store(fingerprint, blocks=1)
    source = _tensor(fingerprint)
    source.copy_(torch.arange(source.numel(), dtype=source.dtype).reshape(source.shape))
    handle = store.store_block(KVBlockPayload(fingerprint, source))
    destination = torch.empty_like(source)

    store.load_into(handle, destination)

    assert torch.equal(destination, source)
    assert store.read_block(handle).tensor.shape == source.shape
    assert store.read_block(handle).tensor.dtype == source.dtype
    assert not store.read_block(handle).tensor.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned allocator needs CUDA")
def test_pinned_store_allocates_pinned_cpu_memory():
    fingerprint = _fingerprint()
    store = _store(fingerprint, blocks=1, pinned=True)
    handle = store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint, 7)))

    assert store.read_block(handle).tensor.is_pinned()


def test_lru_access_order_evicts_least_recently_used_and_calls_back():
    fingerprint = _fingerprint()
    invalidated = []
    store = _store(
        fingerprint,
        blocks=2,
        on_invalidate=lambda handle, key: invalidated.append((handle, key)),
    )
    key_a = CacheKey(fingerprint.digest, 1)
    key_b = CacheKey(fingerprint.digest, 2)
    handle_a = store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint, 1)), key=key_a)
    handle_b = store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint, 2)), key=key_b)
    store.read_block(handle_a)  # B is now least recently used.

    handle_c = store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint, 3)))

    assert handle_c.slot_id == handle_b.slot_id
    assert handle_c.generation > handle_b.generation
    assert invalidated == [(handle_b, key_b)]
    assert torch.all(store.read_block(handle_a).tensor == 1)
    with pytest.raises(StaleCPUBlockHandle):
        store.read_block(handle_b)
    assert store.stats()["evictions"] == 1
    assert store.resident_blocks == store.capacity_blocks == 2


def test_delete_and_reinsert_invalidates_old_generation():
    fingerprint = _fingerprint()
    invalidated = []
    store = _store(
        fingerprint,
        blocks=1,
        on_invalidate=lambda handle, key: invalidated.append(handle),
    )
    first = store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint, 1)))
    store.delete(first)
    second = store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint, 2)))

    assert first.slot_id == second.slot_id
    assert first.generation != second.generation
    assert invalidated == [first]
    with pytest.raises(StaleCPUBlockHandle):
        store.load_into(first, _tensor(fingerprint))
    assert torch.all(store.read_block(second).tensor == 2)


def test_rejects_capacity_fingerprint_shape_and_dtype_mismatches():
    fingerprint = _fingerprint()
    bytes_per_block = _tensor(fingerprint).numel() * _tensor(fingerprint).element_size()
    with pytest.raises(ValueError, match="cannot hold one"):
        CPUBlockStore(fingerprint, capacity_bytes=bytes_per_block - 1)

    store = _store(fingerprint, blocks=1)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        store.store_block(
            KVBlockPayload(
                _fingerprint(config_digest="other"), _tensor(fingerprint)
            )
        )
    with pytest.raises(ValueError, match="payload shape"):
        store.store_block(KVBlockPayload(fingerprint, _tensor(fingerprint)[:, :, :-1]))
    with pytest.raises(ValueError, match="payload dtype"):
        store.store_block(
            KVBlockPayload(fingerprint, _tensor(fingerprint).to(torch.float16))
        )


def test_one_hundred_store_load_evict_cycles_never_exceed_capacity():
    fingerprint = _fingerprint()
    store = _store(fingerprint, blocks=3)
    handles = []
    destination = _tensor(fingerprint)

    for value in range(100):
        handle = store.store_block(
            KVBlockPayload(fingerprint, _tensor(fingerprint, value))
        )
        handles.append(handle)
        store.load_into(handle, destination)
        assert torch.all(destination == value)
        assert store.resident_blocks <= store.capacity_blocks
        assert store.resident_bytes <= store.capacity_bytes

    assert store.resident_blocks == 3
    assert store.stats()["stores"] == 100
    assert store.stats()["loads"] == 100
    assert store.stats()["evictions"] == 97
    for stale in handles[:-3]:
        with pytest.raises(StaleCPUBlockHandle):
            store.read_block(stale)
