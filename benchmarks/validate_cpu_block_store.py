"""Exercise pageable and pinned synchronous GPU/CPU KV block round trips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from nanovllm.kvdb.fingerprint import build_cache_fingerprint
from nanovllm.kvdb.store.cpu import CPUBlockStore
from nanovllm.kvdb.types import KVBlockPayload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/cpu_block_store_validation.json"),
    )
    args = parser.parse_args()

    hf_config = AutoConfig.from_pretrained(args.model)
    fingerprint = build_cache_fingerprint(
        str(args.model), hf_config, block_size=256, tensor_parallel_size=1
    )
    shape = (
        2,
        fingerprint.num_layers,
        fingerprint.block_size,
        fingerprint.num_kv_heads,
        fingerprint.head_dim,
    )
    dtype = getattr(torch, fingerprint.dtype.removeprefix("torch."))
    torch.manual_seed(0)
    source = torch.randn(shape, dtype=dtype, device="cuda")
    bytes_per_block = source.numel() * source.element_size()
    results = []

    for pinned in (False, True):
        store = CPUBlockStore(
            fingerprint,
            capacity_bytes=2 * bytes_per_block,
            pinned=pinned,
        )
        handle = store.store_block(KVBlockPayload(fingerprint, source))
        destination = torch.empty_like(source)
        destination.zero_()
        store.load_into(handle, destination)
        equal = torch.equal(source, destination)
        assert equal
        results.append(
            {
                "mode": "pinned" if pinned else "pageable",
                "round_trip_equal": equal,
                "handle": {
                    "slot_id": handle.slot_id,
                    "generation": handle.generation,
                },
                "stats": store.stats(),
            }
        )

    result = {
        "model": str(args.model),
        "device": torch.cuda.get_device_name(),
        "fingerprint_digest": fingerprint.digest,
        "block_shape": list(shape),
        "dtype": str(dtype),
        "bytes_per_block": bytes_per_block,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
