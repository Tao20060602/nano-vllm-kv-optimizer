from copy import deepcopy
from types import SimpleNamespace

from nanovllm.kvdb.fingerprint import build_cache_fingerprint


class FakeConfig(SimpleNamespace):
    def to_dict(self):
        return deepcopy(self.serialized)


def _config(**overrides):
    values = {
        "dtype": "torch.bfloat16",
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "rope_theta": 1_000_000,
        "rope_scaling": None,
        "serialized": {"model_type": "test", "revision": 1},
    }
    values.update(overrides)
    return FakeConfig(**values)


def test_fingerprint_is_stable_for_same_model_and_layout(tmp_path):
    first = build_cache_fingerprint(
        str(tmp_path), _config(), block_size=16, tensor_parallel_size=1
    )
    second = build_cache_fingerprint(
        str(tmp_path), _config(), block_size=16, tensor_parallel_size=1
    )

    assert first == second
    assert first.digest == second.digest


def test_fingerprint_changes_with_model_config_or_kv_layout(tmp_path):
    baseline = build_cache_fingerprint(
        str(tmp_path), _config(), block_size=16, tensor_parallel_size=1
    )
    changed_config = build_cache_fingerprint(
        str(tmp_path),
        _config(serialized={"model_type": "test", "revision": 2}),
        block_size=16,
        tensor_parallel_size=1,
    )
    changed_layout = build_cache_fingerprint(
        str(tmp_path), _config(), block_size=32, tensor_parallel_size=1
    )

    assert baseline.digest != changed_config.digest
    assert baseline.digest != changed_layout.digest
