import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.kvdb.coordinator import ContextDB
from nanovllm.kvdb.prefix_index import PrefixIndex, compute_block_hash
from nanovllm.kvdb.store.cpu import CPUBlockStore
from nanovllm.kvdb.types import CacheFingerprint, CacheTier, KVBlockPayload


def _fingerprint(**overrides):
    values = {
        "model_id": "/model",
        "config_digest": "config",
        "dtype": "torch.float32",
        "num_layers": 1,
        "num_kv_heads": 1,
        "head_dim": 2,
        "block_size": 4,
        "tensor_parallel_size": 1,
        "rope_theta": 1_000_000.0,
        "rope_scaling": "null",
    }
    values.update(overrides)
    return CacheFingerprint(**values)


def _payload(fingerprint, value):
    shape = (
        2,
        fingerprint.num_layers,
        fingerprint.block_size,
        fingerprint.num_kv_heads,
        fingerprint.head_dim,
    )
    return KVBlockPayload(fingerprint, torch.full(shape, value, dtype=torch.float32))


def _db(blocks=8, *, fingerprint=None, index=None):
    fingerprint = fingerprint or _fingerprint()
    block_bytes = _payload(fingerprint, 0).tensor.numel() * 4
    store = CPUBlockStore(fingerprint, capacity_bytes=blocks * block_bytes)
    return ContextDB(store, index=index)


def _store_tokens(db, tokens):
    session = db.create_session(tokens)
    payloads = [
        _payload(db.fingerprint, block)
        for block in range(session.matched_blocks, len(tokens) // db.block_size)
    ]
    db.store_session(session, payloads)
    return session


def test_hash_helper_exactly_matches_upstream_block_manager():
    tokens = [1, 2, 3, 4]
    prefix = compute_block_hash([5, 6, 7, 8])

    assert compute_block_hash(tokens, prefix) == BlockManager.compute_hash(tokens, prefix)


def test_full_miss_full_block_hit_and_multi_block_hit():
    db = _db()
    miss = db.create_session(list(range(9)))
    assert miss.matched_blocks == 0
    assert miss.lookup.tier is CacheTier.MISS

    stored = _store_tokens(db, list(range(9)))
    assert stored.matched_blocks == 2
    hit = db.create_session(list(range(9)))
    assert hit.matched_blocks == 2
    assert hit.matched_tokens == 8
    assert hit.uncached_token_ids == (8,)
    assert hit.lookup.tier is CacheTier.CPU


def test_partial_hit_stops_at_first_different_block():
    db = _db()
    _store_tokens(db, list(range(13)))
    query = list(range(8)) + [100, 101, 102, 103, 999]

    session = db.create_session(query)

    assert session.matched_blocks == 2
    assert session.matched_tokens == 8
    assert session.uncached_token_ids == tuple(query[8:])


def test_non_full_tail_is_never_indexed_or_reused():
    db = _db()
    _store_tokens(db, [1, 2, 3, 4, 5, 6])

    session = db.create_session([1, 2, 3, 4, 5, 6])

    assert session.matched_blocks == 1
    assert session.matched_tokens == 4
    assert session.uncached_token_ids == (5, 6)
    assert len(db.index) == 1


def test_gap_in_index_prevents_skipping_to_later_block():
    db = _db(blocks=3)
    blocks = [tuple(range(i, i + 4)) for i in (0, 4, 8)]
    prefix_hash = -1
    hashes = []
    handles = []
    for index, block in enumerate(blocks):
        prefix_hash = compute_block_hash(block, prefix_hash)
        hashes.append(prefix_hash)
        handles.append(db.store.store_block(_payload(db.fingerprint, index)))
    db.index.register(db.fingerprint, hashes[0], blocks[0], handles[0])
    db.index.register(db.fingerprint, hashes[2], blocks[2], handles[2])

    lookup = db.index.lookup(list(range(12)), db.fingerprint)

    assert lookup.matched_blocks == 1
    assert lookup.cpu_handles == (handles[0],)


def test_hash_collision_is_rejected_by_token_comparison():
    constant_hash = lambda token_ids, prefix: 42
    index = PrefixIndex(4, hash_fn=constant_hash)
    db = _db(blocks=2, index=index)
    first = db.store.store_block(_payload(db.fingerprint, 1))
    second = db.store.store_block(_payload(db.fingerprint, 2))
    index.register(db.fingerprint, 42, [1, 2, 3, 4], first)
    index.register(db.fingerprint, 42, [9, 8, 7, 6], second)

    first_lookup = index.lookup([1, 2, 3, 4], db.fingerprint)
    second_lookup = index.lookup([9, 8, 7, 6], db.fingerprint)
    miss = index.lookup([4, 3, 2, 1], db.fingerprint)

    assert first_lookup.cpu_handles == (first,)
    assert second_lookup.cpu_handles == (second,)
    assert miss.matched_blocks == 0


def test_different_fingerprint_cannot_hit_same_hash():
    db = _db()
    _store_tokens(db, [1, 2, 3, 4, 5])

    lookup = db.index.lookup(
        [1, 2, 3, 4, 5], _fingerprint(config_digest="different")
    )

    assert lookup.matched_blocks == 0


def test_cpu_lru_eviction_removes_prefix_index_references():
    db = _db(blocks=1)
    first_tokens = [1, 2, 3, 4, 5]
    second_tokens = [9, 8, 7, 6, 5]
    first_session = _store_tokens(db, first_tokens)
    old_handle = first_session.lookup.cpu_handles[0]
    assert len(db.index) == 1

    _store_tokens(db, second_tokens)

    assert not db.store.contains(old_handle)
    assert db.create_session(first_tokens).matched_blocks == 0
    assert db.create_session(second_tokens).matched_blocks == 1
    assert len(db.index) == 1
