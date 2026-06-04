from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from pat_er.configuration_pat_er import PATERConfig


class PATERAuxHeads(nn.Module):
    """Auxiliary readouts for PAT-ER.

    These heads are smoke-ready scaffolding. They produce shaped tensors for
    future supervised losses without claiming trained semantics.
    """

    num_support_statuses = 6
    num_tool_intents = 5
    num_schema_classes = 2
    num_idk_actions = 4

    def __init__(self, config: PATERConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.config = config

        self.predicate_event_head = nn.Linear(h, 4)
        self.event_token_proj = nn.Linear(h, h, bias=False)
        self.argument_query_proj = nn.Linear(h, h, bias=False)
        self.argument_start_proj = nn.Linear(h, h, bias=False)
        self.argument_end_proj = nn.Linear(h, h, bias=False)

        self.proto_role_head = nn.Linear(h, config.num_proto_role_properties)
        self.arg_role_proj = nn.Linear(h, h, bias=False)
        self.role_embeddings = nn.Parameter(torch.empty(config.num_proto_role_properties, h))

        self.event_arg_proj = nn.Linear(h, h, bias=False)
        self.role_to_primitive_head = nn.Linear(h * 3, config.num_primitive_registers)
        self.primitive_class_head = nn.Linear(h, config.num_primitive_registers)
        self.support_status_head = nn.Linear(h, self.num_support_statuses)
        self.evidence_pointer_query = nn.Linear(h, h, bias=False)
        self.evidence_pointer_key = nn.Linear(h, h, bias=False)
        self.tool_intent_head = nn.Linear(h * 3, self.num_tool_intents)
        self.schema_validity_head = nn.Linear(h * 2, self.num_schema_classes)
        self.verifier_likelihood_head = nn.Linear(h * 3, 1)
        self.uncertainty_head = nn.Linear(h, 2)
        self.idk_abstention_head = nn.Linear(h * 3, self.num_idk_actions)
        self.role_ambiguity_head = nn.Linear(h * 2, 4)

        # Reasoning-supervision heads (opt-in). entailment/depth/chain read the
        # side-state (event-role + primitive pools) so reasoning flows through the
        # streams; the two fact pointers are multi-label token pointers.
        if getattr(config, "use_reasoning_heads", False):
            self.entailment_state_head = nn.Linear(h * 3, config.num_entailment_states)
            self.proof_depth_head = nn.Linear(h * 3, config.num_proof_depth_classes)
            self.rule_chain_length_head = nn.Linear(h * 3, config.num_rule_chain_classes)
        if getattr(config, "use_fact_pointer_heads", False):  # tested, not kept (see config note)
            self.supporting_fact_query = nn.Linear(h, h, bias=False)
            self.supporting_fact_key = nn.Linear(h, h, bias=False)
            self.contradicting_fact_query = nn.Linear(h, h, bias=False)
            self.contradicting_fact_key = nn.Linear(h, h, bias=False)
        if getattr(config, "use_evidence_item_heads", False):
            # Per-evidence-item support/refute: pool each item's token span, then
            # score each item independently. The reasoning conclusion (last_state)
            # conditions the scoring.
            self.evidence_item_proj = nn.Linear(h * 2, h)
            self.supporting_item_head = nn.Linear(h, 1)
            self.contradicting_item_head = nn.Linear(h, 1)

        nn.init.normal_(self.role_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        event_registers: torch.Tensor,
        primitive_registers: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        evidence_item_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        batch_size, seq_len, hidden_size = hidden_states.shape
        event_slots = event_registers[:, : cfg.num_event_registers, :]
        argument_slots = event_registers[:, cfg.num_event_registers :, :]

        if event_slots.numel() == 0:
            event_pool = hidden_states.new_zeros(batch_size, hidden_size)
        else:
            event_pool = event_slots.mean(dim=1)
        if argument_slots.numel() == 0:
            argument_pool = hidden_states.new_zeros(batch_size, hidden_size)
        else:
            argument_pool = argument_slots.mean(dim=1)
        if primitive_registers.numel() == 0:
            primitive_pool = hidden_states.new_zeros(batch_size, hidden_size)
        else:
            primitive_pool = primitive_registers.mean(dim=1)

        last_state = self._last_valid_state(hidden_states, attention_mask)

        # Event Binding Graph components. Shapes are kept explicit for later labels.
        event_token_logits = torch.matmul(
            self.event_token_proj(event_slots),
            hidden_states.transpose(1, 2),
        )
        argument_start_logits = torch.matmul(
            self.argument_start_proj(argument_slots),
            hidden_states.transpose(1, 2),
        )
        argument_end_logits = torch.matmul(
            self.argument_end_proj(argument_slots),
            hidden_states.transpose(1, 2),
        )
        event_arg_logits = torch.matmul(
            self.event_arg_proj(event_slots),
            argument_slots.transpose(1, 2),
        )
        arg_role_logits = torch.matmul(
            self.arg_role_proj(argument_slots),
            self.role_embeddings.transpose(0, 1),
        )

        if attention_mask is not None:
            token_mask = attention_mask[:, None, :].to(dtype=torch.bool)
            event_token_logits = event_token_logits.masked_fill(~token_mask, -1.0e9)
            argument_start_logits = argument_start_logits.masked_fill(~token_mask, -1.0e9)
            argument_end_logits = argument_end_logits.masked_fill(~token_mask, -1.0e9)

        proto_role_logits = self.proto_role_head(argument_slots)
        bridge_input = torch.cat([event_pool, argument_pool, primitive_pool], dim=-1)
        event_arg_pool = torch.cat([event_pool, argument_pool], dim=-1)
        primitive_mix = torch.cat([last_state, primitive_pool, event_pool], dim=-1)

        evidence_query = self.evidence_pointer_query(last_state).unsqueeze(1)
        evidence_keys = self.evidence_pointer_key(hidden_states).transpose(1, 2)
        evidence_pointer_logits = torch.matmul(evidence_query, evidence_keys).squeeze(1)
        if attention_mask is not None:
            evidence_pointer_logits = evidence_pointer_logits.masked_fill(~attention_mask.bool(), -1.0e9)

        uncertainty_raw = self.uncertainty_head(last_state)
        uncertainty_alpha_beta = F.softplus(uncertainty_raw) + 1.0

        outputs = {
            # [B, num_event_registers, 4], event type/count sketch.
            "predicate_event_logits": self.predicate_event_head(event_slots),
            # [B, num_argument_registers, T], start and end span logits.
            "argument_start_logits": argument_start_logits,
            "argument_end_logits": argument_end_logits,
            # [B, num_argument_registers, num_proto_role_properties].
            "proto_role_logits": proto_role_logits,
            # Event Binding Graph matrices.
            "event_token_logits": event_token_logits,
            "event_arg_logits": event_arg_logits,
            "arg_role_logits": arg_role_logits,
            # Primitive and bridge heads.
            "role_to_primitive_logits": self.role_to_primitive_head(bridge_input),
            "primitive_class_logits": self.primitive_class_head(primitive_pool),
            "support_status_logits": self.support_status_head(last_state),
            # Simplified evidence pointer: [B, T] over input tokens.
            "evidence_pointer_logits": evidence_pointer_logits,
            "tool_intent_logits": self.tool_intent_head(primitive_mix),
            "schema_validity_logits": self.schema_validity_head(torch.cat([last_state, primitive_pool], dim=-1)),
            "verifier_likelihood_logit": self.verifier_likelihood_head(primitive_mix),
            "uncertainty_alpha_beta": uncertainty_alpha_beta,
            "idk_abstention_logits": self.idk_abstention_head(primitive_mix),
            # [B, 4], overlap/symmetry/with-PP/reversal-risk scaffold.
            "role_ambiguity_logits": self.role_ambiguity_head(event_arg_pool),
        }
        if getattr(cfg, "use_reasoning_heads", False):
            outputs["entailment_state_logits"] = self.entailment_state_head(bridge_input)
            outputs["proof_depth_logits"] = self.proof_depth_head(bridge_input)
            outputs["rule_chain_length_logits"] = self.rule_chain_length_head(bridge_input)
        if getattr(cfg, "use_fact_pointer_heads", False):
            sf_q = self.supporting_fact_query(last_state).unsqueeze(1)
            sf = torch.matmul(sf_q, self.supporting_fact_key(hidden_states).transpose(1, 2)).squeeze(1)
            cf_q = self.contradicting_fact_query(last_state).unsqueeze(1)
            cf = torch.matmul(cf_q, self.contradicting_fact_key(hidden_states).transpose(1, 2)).squeeze(1)
            if attention_mask is not None:
                fill = ~attention_mask.bool()
                sf = sf.masked_fill(fill, -1.0e9)
                cf = cf.masked_fill(fill, -1.0e9)
            # [B, T] multi-label token pointers (BCE over fact token positions).
            outputs["supporting_fact_pointer_logits"] = sf
            outputs["contradicting_fact_pointer_logits"] = cf
        if getattr(cfg, "use_evidence_item_heads", False) and evidence_item_mask is not None:
            # evidence_item_mask: [B, I, T] (1 where a token belongs to item i).
            eim = evidence_item_mask.to(dtype=hidden_states.dtype)
            denom = eim.sum(dim=-1, keepdim=True).clamp(min=1.0)           # [B, I, 1]
            pooled_items = torch.bmm(eim, hidden_states) / denom            # [B, I, h]
            cond = last_state.unsqueeze(1).expand(-1, pooled_items.size(1), -1)
            item_rep = torch.tanh(self.evidence_item_proj(torch.cat([pooled_items, cond], dim=-1)))
            outputs["supporting_item_logits"] = self.supporting_item_head(item_rep).squeeze(-1)   # [B, I]
            outputs["contradicting_item_logits"] = self.contradicting_item_head(item_rep).squeeze(-1)
        return outputs

    @staticmethod
    def _last_valid_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, lengths, :]

