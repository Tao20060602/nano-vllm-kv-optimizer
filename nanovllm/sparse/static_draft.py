"""Minimal static-shape Qwen3-0.6B draft engine with a CUDA-graph decode step.

Why: the transformers draft costs ~25 ms/step, ~5x the memory-bandwidth floor,
because every step launches ~150 small kernels over 28 layers.  This module
implements the forward with a **preallocated static KV cache**, so the whole
28-layer step can be captured once with `torch.cuda.CUDAGraph` and replayed.

Design:
  * KV cache: [layers, window, kv_heads, head_dim], written at index ``pos``.
  * Attention always runs over the full window with a mask ``arange(W) > pos``
    (pos is a static tensor whose value changes between replays).
  * All inputs/outputs live in preallocated static buffers.
  * Prefill is done once with the HF model and its K/V are copied into the
    static cache (one-time cost); the HF model is then freed.

v1 assumes prompt length <= window (no cache shifting during generation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

DTYPE = torch.bfloat16


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y.to(x.dtype) * w)


def _rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # q: [N, heads, D]; cos/sin: [N, 1, D] (already repeated halves)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    return q * cos + rotate_half(q) * sin


class StaticDraft:
    def __init__(self, model_path: str, window: int = 8192,
                 device: str = "cuda", max_position: int = 131072):
        self.device = torch.device(device)
        self.window = int(window)
        cfg = __import__("json").load(open(f"{model_path}/config.json"))
        self.L = cfg["num_hidden_layers"]
        self.Hq = cfg["num_attention_heads"]
        self.Hkv = cfg["num_key_value_heads"]
        self.D = cfg["head_dim"]
        self.hidden = cfg["hidden_size"]
        self.G = self.Hq // self.Hkv
        self.eps = 1e-6

        sd = load_file(f"{model_path}/model.safetensors")
        dev = self.device

        def w(name, transpose=False):
            t = sd[name].to(dev)
            return t.T.contiguous() if transpose else t.contiguous()

        self.embed = w("model.embed_tokens.weight")          # [vocab, hidden]
        self.lm_head = w("lm_head.weight")                   # [vocab, hidden]
        self.ln1 = [w(f"model.layers.{l}.input_layernorm.weight") for l in range(self.L)]
        self.ln2 = [w(f"model.layers.{l}.post_attention_layernorm.weight") for l in range(self.L)]
        self.qn = [w(f"model.layers.{l}.self_attn.q_norm.weight") for l in range(self.L)]
        self.kn = [w(f"model.layers.{l}.self_attn.k_norm.weight") for l in range(self.L)]
        self.Wq = [w(f"model.layers.{l}.self_attn.q_proj.weight", True) for l in range(self.L)]
        self.Wk = [w(f"model.layers.{l}.self_attn.k_proj.weight", True) for l in range(self.L)]
        self.Wv = [w(f"model.layers.{l}.self_attn.v_proj.weight", True) for l in range(self.L)]
        self.Wo = [w(f"model.layers.{l}.self_attn.o_proj.weight", True) for l in range(self.L)]
        self.Wg = [w(f"model.layers.{l}.mlp.gate_proj.weight", True) for l in range(self.L)]
        self.Wu = [w(f"model.layers.{l}.mlp.up_proj.weight", True) for l in range(self.L)]
        self.Wd = [w(f"model.layers.{l}.mlp.down_proj.weight", True) for l in range(self.L)]
        self.norm_w = w("model.norm.weight")

        # RoPE tables (absolute positions, independent of the KV window)
        theta = float(cfg.get("rope_theta", 1e6))
        inv = 1.0 / (theta ** (torch.arange(0, self.D, 2, device=dev).float() / self.D))
        t = torch.arange(max_position + 1, device=dev).float()
        freqs = torch.outer(t, inv)                       # [max_pos+1, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)           # [max_pos+1, D]
        self.cos_tab = emb.cos().to(DTYPE)
        self.sin_tab = emb.sin().to(DTYPE)

        # static KV cache and buffers (ring buffer over absolute positions)
        self.kcache = torch.zeros(self.L, self.window, self.Hkv, self.D,
                                  dtype=DTYPE, device=dev)
        self.vcache = torch.zeros(self.L, self.window, self.Hkv, self.D,
                                  dtype=DTYPE, device=dev)
        self.slot_pos = torch.full((self.window,), -1, dtype=torch.long, device=dev)
        self.in_id = torch.zeros(1, dtype=torch.long, device=dev)
        self.in_pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.in_slot = torch.zeros(1, dtype=torch.long, device=dev)
        self.out_logits = torch.empty(1, self.embed.shape[0], dtype=DTYPE, device=dev)
        self.graph = None
        self._hf = None

    def _hf_model(self):
        if self._hf is None:
            from transformers import AutoModelForCausalLM
            self._hf = AutoModelForCausalLM.from_pretrained(
                "/home/tmz/models/Qwen3-0.6B", dtype=DTYPE,
                attn_implementation="sdpa").to(self.device).eval()
        return self._hf

    # -- one token forward (capturable: static shapes/addresses) ----------
    def _forward(self) -> torch.Tensor:
        pos = self.in_pos                                   # [1] static tensor
        slot = self.in_slot                                 # [1] absolute pos % W
        cos = self.cos_tab[pos].unsqueeze(1)                # [1,1,D]
        sin = self.sin_tab[pos].unsqueeze(1)
        x = self.embed[self.in_id]                          # [1, hidden]
        # mark the current slot before computing the mask (self-visible)
        self.slot_pos.index_copy_(0, slot, pos)
        # a slot is visible iff its absolute position is in [pos-W+1, pos]
        valid = ((self.slot_pos >= 0) & (self.slot_pos <= pos)
                 & (self.slot_pos > (pos - self.window)))
        mask = (~valid).view(1, 1, self.window)              # True = masked
        for l in range(self.L):
            h = _rmsnorm(x, self.ln1[l], self.eps)
            q = (h @ self.Wq[l]).view(1, self.Hq, self.D)
            k = (h @ self.Wk[l]).view(1, self.Hkv, self.D)
            v = (h @ self.Wv[l]).view(1, self.Hkv, self.D)
            q = _rmsnorm(q, self.qn[l], self.eps)
            k = _rmsnorm(k, self.kn[l], self.eps)
            qr = _rope(q, cos, sin)                          # [1,Hq,D]
            kr = _rope(k, cos, sin)                          # [1,Hkv,D]
            self.kcache[l].index_copy_(0, slot, kr)
            self.vcache[l].index_copy_(0, slot, v)
            qg = qr[0].view(self.Hkv, self.G, self.D)        # [Hkv,G,D]
            scores = torch.einsum("hgd,whd->hgw", qg, self.kcache[l]) * (self.D ** -0.5)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores.float(), dim=-1).to(DTYPE)
            o = torch.einsum("hgw,whd->hgd", attn, self.vcache[l])
            o = o.reshape(1, self.Hq * self.D).to(DTYPE)
            x = x + o @ self.Wo[l]
            h = _rmsnorm(x, self.ln2[l], self.eps)
            mlp = F.silu(h @ self.Wg[l]) * (h @ self.Wu[l])
            x = x + mlp @ self.Wd[l]
        x = _rmsnorm(x, self.norm_w, self.eps)
        self.out_logits.copy_(x @ self.lm_head.T)
        return self.out_logits

    # -- prefill via HF, then copy K/V into the static cache --------------
    # -- batched prefill (no HF model; chunk-causal over the ring) --------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = int(input_ids.shape[1])
        if T > self.cos_tab.shape[0]:
            raise ValueError(f"prompt {T} exceeds RoPE table "
                             f"({self.cos_tab.shape[0]})")
        ids = input_ids.to(self.device)
        self.kcache.zero_()
        self.vcache.zero_()
        self.slot_pos.fill_(-1)
        C = 256
        scale = self.D ** -0.5
        for start in range(0, T, C):
            end = min(T, start + C)
            chunk = ids[:, start:end]
            chunk_ids = ids[0, start:end]                          # [C]
            pos = torch.arange(start, end, device=self.device)      # [C]
            slots = pos % self.window
            cos = self.cos_tab[start:end].unsqueeze(1)              # [C,1,D]
            sin = self.sin_tab[start:end].unsqueeze(1)
            self.slot_pos[slots] = pos
            sp = self.slot_pos                                     # [W] abs positions
            valid = ((sp >= 0) & (sp <= pos[:, None])
                     & (sp > (pos[:, None] - self.window)))         # [C, W]
            x = self.embed[chunk_ids]                              # [C, hidden]
            Cc = x.shape[0]
            for l in range(self.L):
                h = _rmsnorm(x, self.ln1[l], self.eps)
                q = (h @ self.Wq[l]).view(Cc, self.Hq, self.D)
                k = (h @ self.Wk[l]).view(Cc, self.Hkv, self.D)
                v = (h @ self.Wv[l]).view(Cc, self.Hkv, self.D)
                q = _rmsnorm(q, self.qn[l], self.eps)
                k = _rmsnorm(k, self.kn[l], self.eps)
                qr = _rope(q, cos, sin)                            # [C,Hq,D]
                kr = _rope(k, cos, sin)                            # [C,Hkv,D]
                self.kcache[l].index_copy_(0, slots, kr)
                self.vcache[l].index_copy_(0, slots, v)
                o = torch.empty(Cc, self.Hq, self.D,
                                dtype=DTYPE, device=self.device)
                for hh in range(self.Hkv):
                    qh = qr[:, hh * self.G:(hh + 1) * self.G, :]   # [C,G,D]
                    Kh = self.kcache[l][:, hh, :]                  # [W,D]
                    Vh = self.vcache[l][:, hh, :]
                    sc = torch.einsum("cgd,wd->cgw", qh, Kh) * scale
                    sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
                    at = torch.softmax(sc.float(), dim=-1).to(DTYPE)
                    o[:, hh * self.G:(hh + 1) * self.G, :] = torch.einsum(
                        "cgw,wd->cgd", at, Vh)
                o = o.reshape(Cc, self.Hq * self.D)
                x = x + o @ self.Wo[l]
                h = _rmsnorm(x, self.ln2[l], self.eps)
                mlp = F.silu(h @ self.Wg[l]) * (h @ self.Wu[l])
                x = x + mlp @ self.Wd[l]
        self.in_pos.fill_(T - 1)
        self.in_slot.fill_((T - 1) % self.window)
        self.in_id.fill_(int(input_ids[0, -1]))
        return None

    def _hf_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Legacy HF-based prefill (kept for reference/parity tests)."""
        T = int(input_ids.shape[1])
        ids = input_ids.to(self.device)
        hf = self._hf_model()
        from transformers import DynamicCache
        cache = DynamicCache()
        CH = 2048
        with torch.no_grad():
            for start in range(0, T, CH):
                end = min(T, start + CH)
                pos = torch.arange(start, end, device=self.device)
                out = hf(ids[:, start:end], past_key_values=cache, use_cache=True,
                         cache_position=pos, logits_to_keep=1)
                cache = out.past_key_values
                excess = cache.get_seq_length() - self.window
                if excess > 0:
                    for layer in cache.layers:
                        layer.keys = layer.keys[:, :, excess:, :].contiguous()
                        layer.values = layer.values[:, :, excess:, :].contiguous()
                del out
        Lc = cache.get_seq_length()
        base = T - Lc
        self.kcache.zero_()
        self.vcache.zero_()
        self.slot_pos.fill_(-1)
        slots = (torch.arange(base, T, device=self.device) % self.window)
        for l in range(self.L):
            self.kcache[l][slots] = cache.layers[l].keys[0].transpose(0, 1)
            self.vcache[l][slots] = cache.layers[l].values[0].transpose(0, 1)
        del cache
        self.slot_pos[slots] = torch.arange(base, T, device=self.device)
        self.in_pos.fill_(T - 1)
        self.in_slot.fill_((T - 1) % self.window)
        self.in_id.fill_(int(input_ids[0, -1]))
        return None

    # -- graph capture and replay ----------------------------------------
    def capture(self, warmup: int = 3) -> None:
        for _ in range(warmup):
            self._forward()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._forward()

    @torch.no_grad()
    def step(self, token_id: int, pos: int) -> torch.Tensor:
        if not 0 <= int(pos) < self.cos_tab.shape[0]:
            raise ValueError(f"position {pos} outside RoPE table "
                             f"(max {self.cos_tab.shape[0] - 1})")
        self.in_id.fill_(int(token_id))
        self.in_pos.fill_(int(pos))
        self.in_slot.fill_(int(pos) % self.window)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._forward()
        return self.out_logits


class StaticDraftRunner:
    """Drop-in replacement for WindowedDraftRunner using the static engine.

    ``propose(token, k, pos)`` feeds ``token`` at absolute ``pos`` and returns
    k greedy proposals.  No rollback is needed: the next round passes its own
    ``pos`` and positions beyond it are masked.
    """

    def __init__(self, model_path: str, window: int = 8192):
        self.engine = StaticDraft(model_path, window)
        self._captured = False

    def prefill(self, input_ids: torch.Tensor, drop_hf: bool = False) -> None:
        self.engine.prefill(input_ids)
        if not self._captured:
            self.engine.capture()
            self._captured = True
        if drop_hf and self.engine._hf is not None:
            del self.engine._hf
            self.engine._hf = None
            torch.cuda.empty_cache()

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def propose(self, token_id: int, k: int, pos: int) -> list[int]:
        logits = self.engine.step(token_id, pos)
        props: list[int] = []
        for i in range(k):
            nxt = int(logits.argmax(-1).item())
            props.append(nxt)
            logits = self.engine.step(nxt, pos + 1 + i)
        return props
