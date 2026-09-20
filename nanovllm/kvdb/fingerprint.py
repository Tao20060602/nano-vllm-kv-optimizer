from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from nanovllm.kvdb.types import CacheFingerprint


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def build_cache_fingerprint(
    model: str,
    hf_config: Any,
    *,
    block_size: int,
    tensor_parallel_size: int,
) -> CacheFingerprint:
    """Build a stable identity without importing or touching CUDA tensors."""

    config_dict = hf_config.to_dict()
    config_digest = sha256(_canonical_json(config_dict).encode("utf-8")).hexdigest()
    num_kv_heads = hf_config.num_key_value_heads // tensor_parallel_size
    head_dim = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
    )
    return CacheFingerprint(
        model_id=str(Path(model).resolve()),
        config_digest=config_digest,
        dtype=str(hf_config.dtype),
        num_layers=hf_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        tensor_parallel_size=tensor_parallel_size,
        rope_theta=float(getattr(hf_config, "rope_theta", 10000.0)),
        rope_scaling=_canonical_json(getattr(hf_config, "rope_scaling", None)),
    )
