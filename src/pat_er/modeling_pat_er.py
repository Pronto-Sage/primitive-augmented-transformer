from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .configuration_pat_er import PATERConfig
from .modules.attention import CrossAttention, GQACausalSelfAttention
from .modules.ffn import RolePrimitiveConditionedFFN
from .modules.heads import PATERAuxHeads
from .modules.registers import RegisterBank
from .modules.utils import RMSNorm, init_weights


@dataclass
class PATEROutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    event_registers: torch.Tensor
    primitive_registers: torch.Tensor
    aux_outputs: dict[str, torch.Tensor]

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "logits": self.logits,
            "event_registers": self.event_registers,
            "primitive_registers": self.primitive_registers,
            "aux_outputs": self.aux_outputs,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class VocabPressureHead(nn.Module):
    """Role/primitive pressure projection.

    Full vocab projections are fine for tiny smoke configs but too expensive at
    760M scale. A positive rank uses a low-rank factorization.
    """

    def __init__(self, input_size: int, vocab_size: int, rank: int = 0) -> None:
        super().__init__()
        if rank and rank > 0:
            self.proj = nn.Sequential(
                nn.Linear(input_size, rank, bias=False),
                nn.Linear(rank, vocab_size, bias=False),
            )
        else:
            self.proj = nn.Linear(input_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PATERBlock(nn.Module):
    """One PAT-ER decoder block.

    This block keeps the ordinary token stream while adding event-role and
    primitive side-state updates. Cross-stream paths can be skipped on layers
    that are not selected by cross_attention_every_n_layers.
    """

    def __init__(self, config: PATERConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_cross_stream = (layer_idx + 1) % config.cross_attention_every_n_layers == 0

        h = config.hidden_size
        self.input_norm = RMSNorm(h, config.rms_norm_eps)
        self.self_attn = GQACausalSelfAttention(
            hidden_size=h,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            use_qk_norm=config.use_qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )

        if self.use_cross_stream:
            self.event_norm = RMSNorm(h, config.rms_norm_eps)
            self.event_read = CrossAttention(h, config.num_attention_heads)
            self.role_inject_norm = RMSNorm(h, config.rms_norm_eps)
            self.role_token_read = CrossAttention(h, config.num_attention_heads)
            # injection_fusion_mode controls fuse input size:
            #   "residual"      cat([x, z, x*z]) → Linear(h*3, h)  (from-scratch default)
            #   "register_only" z only            → Linear(h, h)   (warm-start safe)
            _fuse_in = h if config.injection_fusion_mode == "register_only" else h * 3
            self.role_fuse = nn.Linear(_fuse_in, h, bias=False)

            self.primitive_norm = RMSNorm(h, config.rms_norm_eps)
            self.primitive_read = CrossAttention(h, config.num_attention_heads)
            self.primitive_inject_norm = RMSNorm(h, config.rms_norm_eps)
            self.primitive_token_read = CrossAttention(h, config.num_attention_heads)
            self.primitive_fuse = nn.Linear(_fuse_in, h, bias=False)
        else:
            self.event_norm = None
            self.event_read = None
            self.role_inject_norm = None
            self.role_token_read = None
            self.role_fuse = None
            self.primitive_norm = None
            self.primitive_read = None
            self.primitive_inject_norm = None
            self.primitive_token_read = None
            self.primitive_fuse = None

        self.ffn_norm = RMSNorm(h, config.rms_norm_eps)
        self.ffn = RolePrimitiveConditionedFFN(
            hidden_size=h,
            intermediate_size=config.intermediate_size,
            adapter_rank=config.adapter_rank,
            num_adapters=config.num_ffn_adapters,
            enabled=config.use_role_primitive_ffn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        event_registers: torch.Tensor,
        primitive_registers: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = hidden_states + self.self_attn(self.input_norm(hidden_states), attention_mask=attention_mask)

        if self.use_cross_stream and self.config.generic_register_stream:
            # Generic-register control: same register tensors and same
            # cross-attention/fuse modules as PAT-ER, but no typed
            # event-role -> primitive flow. All registers are updated as one
            # homogeneous memory bank and split only so the existing aux-head
            # readout shapes remain comparable.
            split = event_registers.shape[1]
            generic_registers = torch.cat([event_registers, primitive_registers], dim=1)

            if self.config.use_event_stream:
                assert self.event_norm is not None
                assert self.event_read is not None
                assert self.role_inject_norm is not None
                assert self.role_token_read is not None
                assert self.role_fuse is not None
                generic_update = self.event_read(
                    query=self.event_norm(generic_registers),
                    context=x,
                    context_mask=attention_mask,
                )
                generic_registers = generic_registers + generic_update
                z_generic = self.role_token_read(
                    query=self.role_inject_norm(x),
                    context=generic_registers,
                    context_mask=None,
                )
                if self.config.injection_fusion_mode == "register_only":
                    x = x + self.role_fuse(z_generic.detach())
                else:
                    x = x + self.role_fuse(torch.cat([x, z_generic, x * z_generic], dim=-1))

            if self.config.use_primitive_stream:
                assert self.primitive_norm is not None
                assert self.primitive_read is not None
                assert self.primitive_inject_norm is not None
                assert self.primitive_token_read is not None
                assert self.primitive_fuse is not None
                generic_update = self.primitive_read(
                    query=self.primitive_norm(generic_registers),
                    context=x,
                    context_mask=attention_mask,
                )
                generic_registers = generic_registers + generic_update
                z_generic = self.primitive_token_read(
                    query=self.primitive_inject_norm(x),
                    context=generic_registers,
                    context_mask=None,
                )
                if self.config.injection_fusion_mode == "register_only":
                    x = x + self.primitive_fuse(z_generic.detach())
                else:
                    x = x + self.primitive_fuse(torch.cat([x, z_generic, x * z_generic], dim=-1))

            event_registers = generic_registers[:, :split, :]
            primitive_registers = generic_registers[:, split:, :]
            x = x + self.ffn(self.ffn_norm(x), event_registers, primitive_registers)
            return x, event_registers, primitive_registers

        if self.use_cross_stream and self.config.use_event_stream:
            assert self.event_norm is not None
            assert self.event_read is not None
            assert self.role_inject_norm is not None
            assert self.role_token_read is not None
            assert self.role_fuse is not None
            event_update = self.event_read(
                query=self.event_norm(event_registers),
                context=x,
                context_mask=attention_mask,
            )
            event_registers = event_registers + event_update

            z_event = self.role_token_read(
                query=self.role_inject_norm(x),
                context=event_registers,
                context_mask=None,
            )
            if self.config.injection_fusion_mode == "register_only":
                # Detach z_event: aux gradient must NOT flow back through role_fuse.
                # Without detach, the aux loss drives role_fuse toward primitive/role
                # representation (corrupting the backbone). With detach, only the LM
                # gradient reaches role_fuse, which stays in the LM-helpful regime.
                x = x + self.role_fuse(z_event.detach())
            else:
                x = x + self.role_fuse(torch.cat([x, z_event, x * z_event], dim=-1))

        if self.use_cross_stream and self.config.use_primitive_stream:
            assert self.primitive_norm is not None
            assert self.primitive_read is not None
            assert self.primitive_inject_norm is not None
            assert self.primitive_token_read is not None
            assert self.primitive_fuse is not None
            primitive_context = x
            primitive_context_mask = attention_mask
            if self.config.use_event_stream and event_registers.numel() > 0:
                primitive_context = torch.cat([x, event_registers], dim=1)
                if attention_mask is not None:
                    event_mask = attention_mask.new_ones(attention_mask.shape[0], event_registers.shape[1])
                    primitive_context_mask = torch.cat([attention_mask, event_mask], dim=1)

            primitive_update = self.primitive_read(
                query=self.primitive_norm(primitive_registers),
                context=primitive_context,
                context_mask=primitive_context_mask,
            )
            primitive_registers = primitive_registers + primitive_update

            z_primitive = self.primitive_token_read(
                query=self.primitive_inject_norm(x),
                context=primitive_registers,
                context_mask=None,
            )
            if self.config.injection_fusion_mode == "register_only":
                x = x + self.primitive_fuse(z_primitive.detach())
            else:
                x = x + self.primitive_fuse(torch.cat([x, z_primitive, x * z_primitive], dim=-1))

        x = x + self.ffn(self.ffn_norm(x), event_registers, primitive_registers)
        return x, event_registers, primitive_registers


class PATERForCausalLM(nn.Module):
    """First runnable PAT-ER Causal LM prototype."""

    config_class = PATERConfig

    def __init__(self, config: PATERConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.register_bank = RegisterBank(
            hidden_size=h,
            num_event_registers=config.num_event_registers,
            num_argument_registers=config.num_argument_registers,
            num_primitive_registers=config.num_primitive_registers,
        )
        self.layers = nn.ModuleList([PATERBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        # Mode-gated interface copies of the top-K blocks (used only in interface_mode).
        k = config.interface_adapt_layers
        if k > 0:
            n = config.num_hidden_layers
            self.interface_layers = nn.ModuleList(
                [PATERBlock(config, layer_idx=n - k + j) for j in range(k)]
            )
        else:
            self.interface_layers = None
        self.norm = RMSNorm(h, config.rms_norm_eps)

        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        self.primitive_pressure = VocabPressureHead(h, config.vocab_size, rank=config.pressure_rank)
        self.role_pressure = VocabPressureHead(h, config.vocab_size, rank=config.pressure_rank)
        self.mix_pressure = VocabPressureHead(h * 2, config.vocab_size, rank=config.pressure_rank)
        self.aux_heads = PATERAuxHeads(config)

        # Decoupled interface output head: a logit-only delta read off the (otherwise
        # frozen) final hidden state. Off (None) unless interface_head_rank > 0. It does
        # not touch hidden states or registers, so the aux heads / side-state are
        # unaffected when it is trained — only generation changes in interface_mode.
        if config.interface_head_rank > 0:
            self.interface_head = nn.Sequential(
                nn.Linear(h, config.interface_head_rank, bias=False),
                nn.GELU(),
                nn.Linear(config.interface_head_rank, config.vocab_size, bias=False),
            )
        else:
            self.interface_head = None

        self.apply(init_weights)
        if config.tie_word_embeddings:
            self.tie_weights()
        # Zero the output projection so interface_mode == base behavior at init; the
        # delta is learned from zero during decoupled output-head SFT.
        if self.interface_head is not None:
            nn.init.zeros_(self.interface_head[-1].weight)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_aux: bool = True,
        interface_mode: bool = False,
        **kwargs: Any,
    ) -> PATEROutput:
        evidence_item_mask = kwargs.pop("evidence_item_mask", None)  # [B, I, T] item pooling mask
        del kwargs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        batch_size = input_ids.shape[0]
        hidden_states = self.embed_tokens(input_ids)
        event_registers, primitive_registers = self.register_bank(batch_size)
        event_registers = event_registers.to(device=hidden_states.device, dtype=hidden_states.dtype)
        primitive_registers = primitive_registers.to(device=hidden_states.device, dtype=hidden_states.dtype)

        if not self.config.use_event_stream:
            event_registers = torch.zeros_like(event_registers)
        if not self.config.use_primitive_stream:
            primitive_registers = torch.zeros_like(primitive_registers)

        n_layers = len(self.layers)
        k = self.config.interface_adapt_layers
        for i, layer in enumerate(self.layers):
            # In interface_mode, the top-K blocks are replaced by their trainable
            # interface copies; base mode always runs the original (frozen) blocks.
            if interface_mode and self.interface_layers is not None and i >= n_layers - k:
                layer = self.interface_layers[i - (n_layers - k)]
            hidden_states, event_registers, primitive_registers = layer(
                hidden_states=hidden_states,
                event_registers=event_registers,
                primitive_registers=primitive_registers,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        logits = self._lm_logits(hidden_states, event_registers, primitive_registers,
                                 interface_mode=interface_mode)

        loss = None
        if labels is not None:
            loss = self._causal_lm_loss(logits, labels, attention_mask)

        aux_outputs = {}
        if return_aux:
            aux_event, aux_primitive = event_registers, primitive_registers
            if self.config.aux_from_token_state:
                # Baseline mode: aux heads read pooled token state instead of the
                # side-state registers, so a non-side-state decoder gets a fair
                # shot at the auxiliary labels.
                pooled = self._pooled_token_state(hidden_states, attention_mask)
                aux_event = pooled.unsqueeze(1).expand(-1, event_registers.shape[1], -1).contiguous()
                aux_primitive = pooled.unsqueeze(1).expand(-1, primitive_registers.shape[1], -1).contiguous()
            aux_outputs = self.aux_heads(
                hidden_states=hidden_states,
                event_registers=aux_event,
                primitive_registers=aux_primitive,
                attention_mask=attention_mask,
                evidence_item_mask=evidence_item_mask,
            )

        return PATEROutput(
            loss=loss,
            logits=logits,
            event_registers=event_registers,
            primitive_registers=primitive_registers,
            aux_outputs=aux_outputs,
        )

    @staticmethod
    def _pooled_token_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask[..., None].to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def _lm_logits(
        self,
        hidden_states: torch.Tensor,
        event_registers: torch.Tensor,
        primitive_registers: torch.Tensor,
        interface_mode: bool = False,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if not self.config.use_vocab_pressure:
            # Ablation: no role/primitive pressure at the LM logits.
            if interface_mode and self.interface_head is not None:
                logits = logits + self.interface_head(hidden_states)
            return logits
        batch_size, seq_len, hidden_size = hidden_states.shape
        event_pool = event_registers.mean(dim=1) if event_registers.numel() else hidden_states.new_zeros(batch_size, hidden_size)
        primitive_pool = (
            primitive_registers.mean(dim=1)
            if primitive_registers.numel()
            else hidden_states.new_zeros(batch_size, hidden_size)
        )
        role_pressure = self.role_pressure(event_pool).unsqueeze(1)
        primitive_pressure = self.primitive_pressure(primitive_pool).unsqueeze(1)
        mix = torch.cat(
            [
                hidden_states * primitive_pool.unsqueeze(1),
                hidden_states * event_pool.unsqueeze(1),
            ],
            dim=-1,
        )
        # In-place additions avoid two temporary [batch, seq, vocab] allocations.
        logits.add_(role_pressure)
        logits.add_(primitive_pressure)
        logits.add_(self.mix_pressure(mix))
        # Decoupled interface delta (logit-only; representation untouched).
        if interface_mode and self.interface_head is not None:
            logits = logits + self.interface_head(hidden_states)
        return logits

    def _causal_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        labels = labels.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
