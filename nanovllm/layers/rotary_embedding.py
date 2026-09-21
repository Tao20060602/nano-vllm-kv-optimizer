from functools import lru_cache
import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Standard RoPE (no scaling)."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.attention_scaling = 1.0

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNRotaryEmbedding(nn.Module):
    """YaRN rotary embedding, numerically aligned with HuggingFace Transformers.

    Mirrors `transformers.modeling_rope_utils._compute_yarn_parameters` and
    `Qwen3RotaryEmbedding.forward`: the modified ``inv_freq`` is combined with
    a linear ramp between interpolation and extrapolation, and the resulting
    cos/sin are scaled by ``attention_scaling`` (mscale).
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attention_factor: float | None = None,
        truncate: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        dim = rotary_dim

        # -- compute modified inv_freq (identical to HF Transformers) ------
        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, dim, base,
            original_max_position_embeddings, truncate,
        )

        ramp = torch.arange(dim // 2, dtype=torch.float)
        if low == high:
            high += 0.001
        linear_func = (ramp - low) / (high - low)
        ramp_clamped = torch.clamp(linear_func, 0.0, 1.0)
        extrapolation_factor = 1.0 - ramp_clamped

        inv_freq = (
            inv_freq_interpolation * (1.0 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor
        )

        # -- attention scaling (mscale) ----------------------------------
        if attention_factor is None:
            attention_factor = _yarn_get_mscale(factor)
        self.attention_scaling = float(attention_factor)

        # -- build cos/sin cache for all positions up to max_position -----
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [T, dim//2]
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # [T, 1, dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


_ROPE_CACHE: dict = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    cache_key = (head_size, rotary_dim, max_position, round(base, 6),
                 tuple(sorted(rope_scaling.items())) if rope_scaling else None)
    if cache_key in _ROPE_CACHE:
        return _ROPE_CACHE[cache_key]

    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type == "yarn":
            factor = float(rope_scaling.get("factor", 1.0))
            orig = int(rope_scaling.get("original_max_position_embeddings", max_position))
            result = YaRNRotaryEmbedding(
                head_size, rotary_dim, max_position, base,
                factor=factor,
                original_max_position_embeddings=orig,
                beta_fast=float(rope_scaling.get("beta_fast", 32)),
                beta_slow=float(rope_scaling.get("beta_slow", 1)),
                attention_factor=rope_scaling.get("attention_factor"),
                truncate=bool(rope_scaling.get("truncate", True)),
            )
            _ROPE_CACHE[cache_key] = result
            return result
        # unknown rope_type -> fall through to standard RoPE
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    _ROPE_CACHE[cache_key] = rotary_emb
    return rotary_emb
