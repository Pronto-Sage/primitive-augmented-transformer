from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN used as the base decoder feed-forward path."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LowRankAdapter(nn.Module):
    """Tiny CPU-safe low-rank adapter."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(x)))


class RolePrimitiveConditionedFFN(nn.Module):
    """Base SwiGLU plus small adapters gated by event/primitive pools."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        adapter_rank: int,
        num_adapters: int = 6,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = enabled and adapter_rank > 0 and num_adapters > 0
        self.base = SwiGLUFFN(hidden_size, intermediate_size)
        self.num_adapters = num_adapters

        if self.enabled:
            self.gate = nn.Linear(hidden_size * 2, num_adapters)
            self.adapters = nn.ModuleList([LowRankAdapter(hidden_size, adapter_rank) for _ in range(num_adapters)])
        else:
            self.gate = None
            self.adapters = nn.ModuleList()

    def forward(
        self,
        x: torch.Tensor,
        event_registers: torch.Tensor | None,
        primitive_registers: torch.Tensor | None,
    ) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out

        batch_size = x.shape[0]
        if event_registers is None or event_registers.numel() == 0:
            event_pool = x.new_zeros(batch_size, x.shape[-1])
        else:
            event_pool = event_registers.mean(dim=1)
        if primitive_registers is None or primitive_registers.numel() == 0:
            primitive_pool = x.new_zeros(batch_size, x.shape[-1])
        else:
            primitive_pool = primitive_registers.mean(dim=1)

        gates = torch.softmax(self.gate(torch.cat([event_pool, primitive_pool], dim=-1)), dim=-1)
        adapter_out = x.new_zeros(x.shape)
        for idx, adapter in enumerate(self.adapters):
            adapter_out = adapter_out + gates[:, idx].view(batch_size, 1, 1) * adapter(x)
        return out + adapter_out
