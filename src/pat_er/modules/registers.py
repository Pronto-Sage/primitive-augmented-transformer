from __future__ import annotations

import torch
from torch import nn


class RegisterBank(nn.Module):
    """Learned side-state registers expanded per batch.

    The first prototype uses learned tensors as the initial event-role and
    primitive streams. Blocks then update them through cross-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_event_registers: int,
        num_argument_registers: int,
        num_primitive_registers: int,
    ) -> None:
        super().__init__()
        self.num_event_registers = num_event_registers
        self.num_argument_registers = num_argument_registers
        self.num_event_role_registers = num_event_registers + num_argument_registers
        self.num_primitive_registers = num_primitive_registers

        self.event_role_registers = nn.Parameter(torch.empty(self.num_event_role_registers, hidden_size))
        self.primitive_registers = nn.Parameter(torch.empty(num_primitive_registers, hidden_size))
        nn.init.normal_(self.event_role_registers, mean=0.0, std=0.02)
        nn.init.normal_(self.primitive_registers, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        event_role = self.event_role_registers.unsqueeze(0).expand(batch_size, -1, -1)
        primitive = self.primitive_registers.unsqueeze(0).expand(batch_size, -1, -1)
        return event_role.contiguous(), primitive.contiguous()

    def split_event_argument(self, event_role_registers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        events = event_role_registers[:, : self.num_event_registers, :]
        arguments = event_role_registers[:, self.num_event_registers :, :]
        return events, arguments


def pooled_register_state(
    event_registers: torch.Tensor | None,
    primitive_registers: torch.Tensor | None,
    hidden_size: int,
) -> torch.Tensor:
    """Pool available register streams for gating."""

    pools: list[torch.Tensor] = []
    if event_registers is not None and event_registers.numel() > 0:
        pools.append(event_registers.mean(dim=1))
    if primitive_registers is not None and primitive_registers.numel() > 0:
        pools.append(primitive_registers.mean(dim=1))
    if pools:
        return torch.cat(pools, dim=-1)
    raise ValueError("At least one register stream is required for pooled_register_state")

