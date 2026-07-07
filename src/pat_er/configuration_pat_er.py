from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PATERConfig:
    """Small HuggingFace-style config object for PAT-ER.

    The class intentionally avoids depending on transformers.PretrainedConfig in
    this first prototype. It keeps the same shape: serializable attributes,
    from_dict/from_yaml constructors, and a to_dict method.
    """

    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 176
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1.0e-5
    num_event_registers: int = 4
    num_argument_registers: int = 6
    num_primitive_registers: int = 14
    num_proto_role_properties: int = 17
    cross_attention_every_n_layers: int = 1
    adapter_rank: int = 8
    use_event_stream: bool = True
    use_primitive_stream: bool = True
    use_role_primitive_ffn: bool = True
    # Reviewer-control baseline: keep the same learned register count and the
    # same cross-attention/fuse modules, but update all registers as one
    # homogeneous latent bank. The bank is split only for existing aux-head
    # shapes. This tests "generic learned memory/registers" against the typed
    # PAT-ER event-role -> primitive flow. Default off => existing configs are
    # byte-identical.
    generic_register_stream: bool = False
    num_ffn_adapters: int = 6
    pressure_rank: int = 0
    use_vocab_pressure: bool = True
    aux_from_token_state: bool = False
    # Reasoning-supervision heads (Phase 1.1). Opt-in: off => model is byte-identical
    # to the frozen baseline. Targets come from ProofWriter proof metadata (aux-head
    # targets only, never model input).
    use_reasoning_heads: bool = False
    # Separate opt-in for the fact-pointer heads/losses. Tested and NOT kept: the
    # single-query [B,T] multi-label pointer did not learn multi-token fact sets
    # (F1 ~0.06) and regressed MP (docs/results/fact_pointer_increment.md). Default
    # off so --use-reasoning-heads alone = the validated heads-only extension.
    use_fact_pointer_heads: bool = False
    # Per-evidence-item support/refute heads (the redesign: fact support is a set
    # over evidence ITEMS, scored independently, not a token pointer).
    use_evidence_item_heads: bool = False
    max_evidence_items: int = 24
    num_entailment_states: int = 3        # entailed / refuted / unknown
    num_proof_depth_classes: int = 6      # depth 0..4, 5+ bucketed
    num_rule_chain_classes: int = 6       # chain length 0..4, 5+ bucketed
    # Warm-start (Phase 2) backbone-compatibility knobs. Both default to the
    # from-scratch behaviour so existing 450M configs stay byte-identical:
    #   attn_head_dim == 0    -> head_dim = hidden_size // num_attention_heads
    #   use_qk_norm   == False -> no per-head q/k RMSNorm
    # Set them (e.g. attn_head_dim=128, use_qk_norm=True) to match a pretrained
    # Qwen3-style decoder whose head_dim is decoupled from hidden/heads.
    attn_head_dim: int = 0
    use_qk_norm: bool = False
    # Injection fusion mode — controls what is fed into role_fuse / primitive_fuse:
    #   "residual"      (default) cat([x, z_event, x*z_event]) → fuse, shape [h, h*3→h]
    #                   From-scratch behavior: the residual stream and cross-attention
    #                   output interact multiplicatively before injection.
    #   "register_only" z_event → fuse only, shape [h, h→h]
    #                   Warm-start-safe: x is absent from the injection path.
    #                   With a pretrained backbone, x has magnitude ~1.0; including it
    #                   amplifies role_fuse's gradient 55× vs register_only, making
    #                   inject_lr=1e-6 insufficient to keep LM delta < 0.5 nats
    #                   (Stage 4 negative result, docs/results/warmstart_stage4_neg.md).
    injection_fusion_mode: str = "residual"
    # Decoupled interface output head (D-interface decoupled SFT). 0 disables it and
    # the architecture is byte-identical to the proven path. When > 0, a low-rank
    # output module (Linear(h, rank) → GELU → Linear(rank, vocab), zero-init output)
    # adds a logit-only delta in `interface_mode`. It reads the final hidden state but
    # NEVER feeds back into hidden states/registers, so every aux head — including the
    # role→primitive bridge — is unaffected. This separates product-format generation
    # from the representation the side-state depends on.
    interface_head_rank: int = 0
    # Mode-gated representation adaptation (D-interface decoupled SFT, option 3). 0
    # disables it (architecture byte-identical to the proven path). When > 0, trainable
    # COPIES of the top-K decoder blocks are used ONLY in `interface_mode`; base mode
    # runs the original frozen blocks, so the aux heads / side-state read exactly the
    # Condition-D representation (r2p preserved). Interface mode gets adapted attention,
    # which a logit-only head cannot supply (in-context tool-name copying).
    interface_adapt_layers: int = 0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    model_type: str = "pat_er"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.cross_attention_every_n_layers < 1:
            raise ValueError("cross_attention_every_n_layers must be >= 1")
        if self.num_event_registers < 0 or self.num_argument_registers < 0:
            raise ValueError("event and argument register counts must be non-negative")
        if self.num_primitive_registers < 0:
            raise ValueError("num_primitive_registers must be non-negative")

    @property
    def num_event_role_registers(self) -> int:
        return self.num_event_registers + self.num_argument_registers

    @property
    def head_dim(self) -> int:
        if self.attn_head_dim > 0:
            return self.attn_head_dim
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PATERConfig":
        filtered = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PATERConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls.from_dict(values)

    def save_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=True)
