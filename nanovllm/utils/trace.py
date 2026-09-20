"""Opt-in, one-shot attention trace for the M9 real-model laboratory.

The tracer is disabled by default and does essentially no work in
:meth:`Attention.forward` unless explicitly armed *after* model construction
(so warmup and CUDA-graph capture can never populate it).  On the next cold,
single-sequence prefill of one chosen layer it copies the post-RoPE
``q_last / k / v`` and the FlashAttention output ``o_last`` to CPU, detaches
them, and immediately disarms itself.

Only the shared :class:`nanovllm.layers.attention.Attention` path is involved;
model-specific attention code is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionTrace:
    """CPU-resident snapshot of one layer's cold-prefill attention."""

    layer_id: int
    q_last: torch.Tensor   # [num_query_heads, head_dim], CPU
    k: torch.Tensor        # [prompt_tokens, num_kv_heads, head_dim], CPU
    v: torch.Tensor        # [prompt_tokens, num_kv_heads, head_dim], CPU
    o_last: torch.Tensor   # [num_query_heads, head_dim], CPU
    dtype: str

    @property
    def num_tokens(self) -> int:
        return self.k.shape[0]

    @property
    def num_query_heads(self) -> int:
        return self.q_last.shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self.k.shape[1]

    @property
    def head_dim(self) -> int:
        return self.q_last.shape[1]


class AttentionTracer:
    """Process-global, one-shot tracer guarded by an explicit arm flag."""

    def __init__(self) -> None:
        self._armed: bool = False
        self._layer_id: int | None = None
        self._trace: AttentionTrace | None = None

    def arm(self, layer_id: int) -> None:
        """Arm the tracer for one specific layer's next cold prefill."""
        self._trace = None
        self._layer_id = int(layer_id)
        self._armed = True

    def disarm(self) -> None:
        self._armed = False
        self._layer_id = None

    def clear(self) -> None:
        self._trace = None
        self._armed = False
        self._layer_id = None

    @property
    def armed(self) -> bool:
        return self._armed

    def maybe_capture(
        self,
        module_layer_id,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        context,
    ) -> None:
        """Capture once if armed for this layer on a valid cold prefill.

        Conditions: explicitly armed; prefill; no paged block table (cold path,
        so ``k/v`` are the raw post-RoPE tensors); exactly one packed sequence.
        """
        if not self._armed:
            return
        if module_layer_id is None or module_layer_id != self._layer_id:
            return
        if not getattr(context, "is_prefill", False):
            return
        if context.block_tables is not None:
            return
        cu_q = context.cu_seqlens_q
        # A single packed sequence has exactly two cumulative boundaries [0, T].
        if cu_q is None or cu_q.numel() != 2:
            return

        trace = AttentionTrace(
            layer_id=int(module_layer_id),
            q_last=q[-1].detach().to("cpu", copy=True).contiguous(),
            k=k.detach().to("cpu", copy=True).contiguous(),
            v=v.detach().to("cpu", copy=True).contiguous(),
            o_last=o[-1].detach().to("cpu", copy=True).contiguous(),
            dtype=str(q.dtype),
        )
        self._trace = trace
        # One-shot: disarm immediately after a successful capture.
        self._armed = False
        self._layer_id = None

    def retrieve(self) -> AttentionTrace | None:
        """Return the captured trace (CPU tensors only), or None."""
        return self._trace


_TRACER = AttentionTracer()


def get_tracer() -> AttentionTracer:
    return _TRACER
