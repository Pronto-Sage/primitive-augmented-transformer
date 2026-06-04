from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STRUCTURAL_SPECIAL_TOKENS: list[str] = [
    "<pat_er>",
    "</pat_er>",
    "<text>",
    "</text>",
    "<evidence>",
    "</evidence>",
    "<claim>",
    "</claim>",
    "<formula>",
    "</formula>",
    "<atom>",
    "</atom>",
    "<predicate>",
    "</predicate>",
    "<argument>",
    "</argument>",
    "<event_graph>",
    "</event_graph>",
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
    "<tool>",
    "</tool>",
    "<tool_response>",
    "</tool_response>",
    "<schema>",
    "</schema>",
    "<provenance>",
    "</provenance>",
    "<output>",
    "</output>",
]

EVENT_ROLE_SPECIAL_TOKENS: list[str] = [
    "<evt:0>",
    "<evt:1>",
    "<evt:2>",
    "<evt:3>",
    "<arg:0>",
    "<arg:1>",
    "<arg:2>",
    "<arg:3>",
    "<arg:4>",
    "<arg:adjunct>",
    "<role:volition>",
    "<role:sentience>",
    "<role:causes_change>",
    "<role:moves>",
    "<role:undergoes_change>",
    "<role:affected>",
    "<role:incremental_theme>",
    "<role:stationary>",
    "<role:exists_independently>",
    "<role:source>",
    "<role:goal>",
    "<role:path>",
    "<role:instrument>",
    "<role:comitative>",
    "<role:benefactive>",
    "<role:location>",
    "<role:measure_extent>",
    "<role:ambiguity>",
    "<role:symmetric>",
    "<role:role_reversal>",
]

PRIMITIVE_SPECIAL_TOKENS: list[str] = [
    "<prim:axiom>",
    "<prim:observation>",
    "<prim:contingency>",
    "<prim:contradiction>",
    "<prim:tautology>",
    "<prim:modus_ponens>",
    "<prim:syllogism>",
    "<prim:abduction>",
    "<prim:semantic_conflict>",
    "<reg:axiom>",
    "<reg:observation>",
    "<reg:contingency>",
    "<reg:contradiction>",
    "<reg:tautology>",
    "<reg:modus_ponens>",
    "<reg:syllogism>",
    "<reg:abduction>",
    "<reg:semantic_conflict>",
    "<reg:uncertainty>",
    "<reg:provenance>",
    "<reg:tool>",
    "<reg:schema>",
    "<reg:idk>",
]

SUPPORT_CONTROL_SPECIAL_TOKENS: list[str] = [
    "<support:proof>",
    "<support:belief>",
    "<support:hypothesis>",
    "<support:unknown>",
    "<support:conflict>",
    "<support:no_progress>",
    "<IDK>",
    "<needs_evidence>",
    "<needs_verification>",
    "<ask_clarification>",
    "<unsupported>",
    "<conflict>",
    "<no_progress>",
]

FORMULA_NORMAL_TOKENS: list[str] = [
    "AND",
    "OR",
    "NOT",
    "IMPLIES",
    "IFF",
    "FORALL",
    "EXISTS",
    "TRUE",
    "FALSE",
    "LAMBDA",
    "CAUSES",
    "SUPPORTS",
    "REFUTES",
    "ENTAILS",
    "CONTRADICTS",
    "UNKNOWN",
]

SCHEMA_KEY_CANDIDATE_TOKENS: list[str] = [
    "name",
    "arguments",
    "parameters",
    "properties",
    "required",
    "description",
    "type",
    "function",
    "evidence_ids",
    "claim_id",
    "source_id",
    "tool_id",
    "predicate_span",
    "argument_span",
    "proto_role",
    "verifier_likelihood",
    "tool_choice",
    "tool_calls",
]


@dataclass(frozen=True)
class PATERTokenizerSpec:
    """Tokenizer extension spec for PATERTokenizer-v1.

    This is dependency-free by design. Projects that use Hugging Face
    tokenizers can pass the returned lists to add_special_tokens/add_tokens.
    """

    special_tokens: tuple[str, ...]
    normal_tokens: tuple[str, ...]
    schema_key_candidates: tuple[str, ...]

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def num_normal_tokens(self) -> int:
        return len(self.normal_tokens)

    @property
    def total_added_tokens_without_schema_keys(self) -> int:
        return self.num_special_tokens + self.num_normal_tokens

    def to_hf_add_special_tokens_dict(self) -> dict[str, list[str]]:
        return {"additional_special_tokens": list(self.special_tokens)}


def build_pater_tokenizer_spec(include_schema_key_candidates: bool = True) -> PATERTokenizerSpec:
    special = (
        STRUCTURAL_SPECIAL_TOKENS
        + EVENT_ROLE_SPECIAL_TOKENS
        + PRIMITIVE_SPECIAL_TOKENS
        + SUPPORT_CONTROL_SPECIAL_TOKENS
    )
    normal = list(FORMULA_NORMAL_TOKENS)
    if include_schema_key_candidates:
        normal.extend(SCHEMA_KEY_CANDIDATE_TOKENS)
    return PATERTokenizerSpec(
        special_tokens=tuple(dict.fromkeys(special)),
        normal_tokens=tuple(dict.fromkeys(normal)),
        schema_key_candidates=tuple(SCHEMA_KEY_CANDIDATE_TOKENS),
    )


def apply_to_hf_tokenizer(tokenizer: Any, include_schema_key_candidates: bool = True) -> dict[str, int]:
    """Apply PAT-ER tokens to a Hugging Face tokenizer-like object.

    The caller is responsible for resizing model embeddings after this:
    model.resize_token_embeddings(len(tokenizer))
    """

    spec = build_pater_tokenizer_spec(include_schema_key_candidates=include_schema_key_candidates)
    added_special = tokenizer.add_special_tokens(spec.to_hf_add_special_tokens_dict())
    added_normal = tokenizer.add_tokens(list(spec.normal_tokens))
    return {"added_special": int(added_special), "added_normal": int(added_normal)}
