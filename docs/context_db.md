# Milestone 4: PrefixIndex and ContextDB

This milestone creates the token-identity layer over `CPUBlockStore`; it still
does not modify scheduler or model execution.

## Chained token-block identity

`compute_block_hash()` is shared by the upstream `BlockManager` adapter and the
new `PrefixIndex`, preserving the original xxHash byte/chaining semantics. Each
index key is:

```text
CacheKey(CacheFingerprint.digest, chained_token_block_hash)
```

The hash only selects candidates. Every hit compares the complete token-ID tuple
before returning a handle, so a simulated equal-hash/different-token collision is
a miss. Lookup starts at logical block zero and stops at the first absent or
different block; it never skips a gap to reuse a later block.

Only `len(token_ids) // block_size` complete blocks can be registered. A partial
tail remains in `ContextSession.uncached_token_ids`.

## API

```python
session = db.create_session(token_ids)
session.matched_blocks
session.matched_tokens
session.uncached_token_ids

db.store_session(session, uncached_full_block_payloads)
stats = db.stats()
```

Payloads begin at `session.matched_blocks`; already cached prefix blocks are not
stored again. The store re-runs lookup after materialization because a context
larger than CPU capacity may evict one of its own earlier blocks. The returned
session therefore never claims stale or non-contiguous residency.

## Lifetime coupling

`ContextDB` registers `PrefixIndex.remove_handle()` as a CPU-store invalidation
callback. Both explicit delete and LRU eviction remove every index entry that
references the invalidated `(slot_id, generation)`. Lookup also validates handle
liveness and removes any stale reference defensively.

This is why CPU context identity survives GPU physical-block reuse but does not
survive CPU slot eviction: the index points only to versioned CPU handles, never
to GPU block IDs.

## Validation

```bash
python -m pytest -q tests/test_prefix_index.py
```

Coverage includes full miss, one/multiple complete-block hits, partial hit,
non-aligned tail, an explicit middle-index gap, forced hash collision, fingerprint
mismatch, and CPU LRU eviction cleanup. These tests use small CPU tensors and do
not start a model.
