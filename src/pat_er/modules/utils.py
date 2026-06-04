from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Iterator

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm used by modern decoder-only LLMs."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (x_norm.to(orig_dtype) * self.weight).to(orig_dtype)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def expand_attention_mask(attention_mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Convert a [B, T] mask with 1 for valid tokens to additive attention bias."""

    if attention_mask is None:
        return None
    min_value = -1.0e4 if dtype in (torch.float16, torch.bfloat16) else -1.0e9
    mask = (1.0 - attention_mask.to(dtype=dtype)) * min_value
    return mask[:, None, None, :]


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def estimate_decoder_params(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    tie_embeddings: bool = True,
) -> int:
    """Rough dense decoder parameter estimate, excluding tiny heads/register params."""

    head_dim = hidden_size // num_heads
    q = hidden_size * (num_heads * head_dim)
    k = hidden_size * (num_kv_heads * head_dim)
    v = hidden_size * (num_kv_heads * head_dim)
    o = hidden_size * hidden_size
    # SwiGLU has gate, up, and down projections.
    ffn = (2 * hidden_size * intermediate_size) + (intermediate_size * hidden_size)
    norms = 2 * hidden_size
    per_layer = q + k + v + o + ffn + norms
    embeddings = vocab_size * hidden_size
    lm_head = 0 if tie_embeddings else vocab_size * hidden_size
    return embeddings + lm_head + num_layers * per_layer


def maybe_autocast(device: str, dtype_name: str):
    """Return a best-effort autocast context for smoke scripts."""

    if dtype_name == "fp32":
        return nullcontext()
    if dtype_name != "bf16":
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def iter_module_summary(module: nn.Module, max_depth: int = 2) -> Iterator[str]:
    for name, child in module.named_modules():
        if not name:
            continue
        depth = name.count(".") + 1
        if depth <= max_depth:
            params = sum(p.numel() for p in child.parameters(recurse=False))
            yield f"{name}: {child.__class__.__name__} direct_params={params}"


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)

