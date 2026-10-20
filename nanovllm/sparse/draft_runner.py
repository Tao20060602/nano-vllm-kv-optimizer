"""Bounded-window draft runner for speculative decoding (Qwen3-0.6B).

The draft keeps a KV cache of at most ``window`` tokens, so its GPU memory is
constant in context length (~112 KB/token for Qwen3-0.6B; 4096 -> ~448 MB).
Proposals are greedy. The cache is rolled back by ``commit(n)`` to keep only
the accepted prefix, and trimmed from the front when it exceeds the window.

Absolute positions are passed via ``cache_position`` so RoPE stays correct
after front-trimming.  Greedy proposals over a short (untrimmed) context are
bit-identical to a full-context reference; after trimming the draft simply
sees less history, which only lowers acceptance (the target still verifies).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, DynamicCache


class WindowedDraftRunner:
    def __init__(self, model_path: str, window: int = 4096,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa").to(device)
        self.model.eval()
        self.window = int(window)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.cache = DynamicCache()
        self.n_seen = 0          # absolute tokens fed
        self._last_k = 0         # proposals produced by the last propose()
        self.profile_steps = False
        self.step_ms: list[float] = []

    # -- internals ------------------------------------------------------
    def _cache_len(self) -> int:
        return int(self.cache.get_seq_length())

    def _trim(self) -> None:
        # Amortize the copy: trim only after exceeding the window by a margin.
        if self._cache_len() <= self.window + 256:
            return
        for layer in self.cache.layers:
            k, v = layer.keys, layer.values
            if k is not None and k.shape[2] > self.window:
                layer.keys = k[:, :, -self.window:, :].contiguous()
                layer.values = v[:, :, -self.window:, :].contiguous()

    @torch.no_grad()
    def _step_tensor(self, nxt: torch.Tensor) -> torch.Tensor:
        import time as _time
        t0 = _time.perf_counter()
        pos = torch.tensor([self.n_seen], device=self.device)
        out = self.model(nxt, past_key_values=self.cache, use_cache=True,
                         cache_position=pos)
        self.cache = out.past_key_values
        self.n_seen += 1
        self._trim()
        logits = out.logits[:, -1, :]
        if self.profile_steps:
            self.step_ms.append((_time.perf_counter() - t0) * 1000.0)
        return logits

    # -- public ---------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Feed a prompt [1, T]; returns the last-position logits."""
        T = int(input_ids.shape[1])
        start = max(0, T - self.window)
        ids = input_ids[:, start:].to(self.device)
        pos = torch.arange(start, T, device=self.device)
        out = self.model(ids, past_key_values=self.cache, use_cache=True,
                         cache_position=pos)
        self.cache = out.past_key_values
        self.n_seen = T
        self._trim()
        return out.logits[:, -1, :]

    @torch.no_grad()
    def propose(self, token_id: int, k: int) -> list[int]:
        """Feed ``token_id`` (the current prefix's last token) and propose k."""
        nxt = torch.tensor([[int(token_id)]], device=self.device)
        logits = self._step_tensor(nxt)
        collected = []
        for _ in range(k):
            nxt = logits.argmax(-1, keepdim=True)
            collected.append(nxt)
            logits = self._step_tensor(nxt)
        self._last_k = k
        return torch.cat(collected, dim=1).tolist()[0]

    def commit(self, n_accepted: int) -> None:
        """Keep the accepted prefix; drop the unaccepted proposals."""
        drop = self._last_k - int(n_accepted)
        if drop > 0:
            keep_abs = self.n_seen - drop
            start = self.n_seen - self._cache_len()
            self.cache.crop(keep_abs - start)
            self.n_seen = keep_abs
        self._last_k = 0
