"""Pluggable draft proposers for speculative decoding.

The speculative loop only needs three things from a proposer:

    prefill(input_ids)            # build state for a new prompt
    propose(last_token, k, pos)   # return up to k candidate token ids
    note_accepted(tokens)         # advance state after verification

Implementations:
  * DraftModelProposer - the static-graph Qwen3-0.6B draft (current default).
  * NgramProposer      - no model at all; copies continuations from the
                         existing context (prompt lookup). Zero GPU cost,
                         strong on repetitive / RAG / editing workloads.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import torch

from nanovllm.sparse.static_draft import StaticDraftRunner


class DraftModelProposer:
    def __init__(self, model_path: str, window: int = 8192):
        self.runner = StaticDraftRunner(model_path, window=window)

    def prefill(self, input_ids: torch.Tensor) -> None:
        self.runner.prefill(input_ids)

    def note_accepted(self, tokens: list[int]) -> None:
        pass  # the static engine is position-driven

    def propose(self, last_token: int, k: int, pos: int) -> list[int]:
        return self.runner.propose(last_token, k, pos)

    def reset(self) -> None:
        self.runner.reset()


class NgramProposer:
    """Prompt-lookup drafter: match the last n tokens in the context."""

    def __init__(self, n: int = 4, max_k: int = 8):
        self.n = int(n)
        self.max_k = int(max_k)
        self.tokens: list[int] = []
        self.index: dict[tuple, list[int]] = {}

    def reset(self) -> None:
        self.tokens = []
        self.index = {}

    def _index_from(self, start: int) -> None:
        n = self.n
        for i in range(max(0, start), len(self.tokens) - n):
            key = tuple(self.tokens[i:i + n])
            self.index.setdefault(key, []).append(self.tokens[i + n])

    def prefill(self, input_ids) -> None:
        self.tokens = list(input_ids)
        self.index = {}
        self._index_from(0)

    def note_accepted(self, tokens: list[int]) -> None:
        start = max(0, len(self.tokens) - self.n)
        self.tokens.extend(tokens)
        self._index_from(start)

    def propose(self, last_token: int, k: int, pos: int) -> list[int]:
        k = min(int(k), self.max_k)
        if len(self.tokens) < self.n:
            return []
        if not self.tokens or self.tokens[-1] != last_token:
            # keep the context aligned even if the caller did not note it
            self.note_accepted([last_token])
        key = tuple(self.tokens[-self.n:])
        opts = self.index.get(key)
        if not opts:
            return []
        props: list[int] = []
        for _ in range(k):
            key = tuple((self.tokens + props)[-self.n:])
            opts = self.index.get(key)
            if not opts:
                break
            props.append(Counter(opts).most_common(1)[0][0])
        return props
