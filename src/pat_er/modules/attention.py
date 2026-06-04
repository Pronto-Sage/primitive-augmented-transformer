from __future__ import annotations

import math

import torch
from torch import nn

from .utils import RMSNorm, causal_mask, expand_attention_mask


class RotaryEmbedding(nn.Module):
    """Minimal RoPE cache for smoke-scale decoder attention."""

    def __init__(self, dim: int, max_position_embeddings: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQACausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int | None = None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        # head_dim may be decoupled from hidden//heads (warm-start path: Qwen3
        # uses head_dim 128 with hidden 1024 / 16 heads). None -> from-scratch.
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        # Qwen3-style per-head q/k RMSNorm (applied on head_dim before RoPE).
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rope(seq_len, x.device, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.float()
        scores = scores.masked_fill(causal_mask(seq_len, x.device)[None, None, :, :], -1.0e9)
        expanded_mask = expand_attention_mask(attention_mask, scores.dtype)
        if expanded_mask is not None:
            scores = scores + expanded_mask

        probs = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Small multi-head cross-attention used for register reads and token injection."""

    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        context_len = context.shape[1]

        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.float()
        expanded_mask = expand_attention_mask(context_mask, scores.dtype)
        if expanded_mask is not None:
            scores = scores + expanded_mask
        probs = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_size)
        return self.o_proj(out)

