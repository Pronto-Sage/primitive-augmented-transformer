#!/usr/bin/env python3
"""Tiny supervised PAT-ER overfit: LM + auxiliary heads on the synthetic dataset.

This trains ``PATERForCausalLM`` (tiny_smoke config by default) on the synthetic
PAT-ER dataset, combining the causal LM loss with masked auxiliary losses.

Per-record (sequence) heads:

    primitive_class   <- primitive_class_logits        (14-way CE)
    support_status    <- support_status_logits          (6-way CE)
    tool_intent       <- tool_intent_logits             (5-way CE)
    schema_validity   <- schema_validity_logits          (2-way CE)
    idk_action        <- idk_abstention_logits           (4-way CE)
    role_to_primitive <- role_to_primitive_logits       (14-way CE, bridge)
    role_ambiguity    <- role_ambiguity_logits           (4-way CE)
    verifier_accept   <- verifier_likelihood_logit         (BCE)
    evidence_pointer  <- evidence_pointer_logits          (token-index CE)

Event-role / event-binding heads (docs/datasets_aug.md section 4):

    predicate_event   <- predicate_event_logits   (event-type CE on event slot 0)
    event_token       <- event_token_logits       (event slot 0 -> predicate token)
    argument_start    <- argument_start_logits     (arg register -> start token)
    argument_end      <- argument_end_logits       (arg register -> end token)
    proto_role        <- proto_role_logits         (arg register -> 17 props, BCE)
    arg_role          <- arg_role_logits           (arg register -> 17 props, BCE)
    event_arg         <- event_arg_logits          (event slot -> arg slots, BCE)

Missing labels are masked (ignore_index / per-slot masks), never coerced. Each
loss is logged separately. Checkpoints (model + tokenizer vocab + config) go
under artifacts/checkpoints/tiny_aux_overfit/.

This module is imported by eval_lm_aux.py for shared dataset/tokenizer/encoding/
target/loss/metric logic. CPU/GPU-safe; no 770M; no downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

import torch
import torch.nn.functional as F

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import count_parameters, maybe_autocast
from pat_er.sample_data import ReferenceTokenizer, build_reference_tokenizer
from pat_er.serialization import render_pater_prompt
from pat_er.tokenizer_spec import build_pater_tokenizer_spec

import build_synthetic_pater_dataset as builder
import pater_hf_tokenizer as H
import warmstart_from_qwen as WS

# Config flags the ablation gate can flip off, as (cli_dest, config_attr).
ABLATION_FLAGS = (
    ("no_event_stream", "use_event_stream"),
    ("no_primitive_stream", "use_primitive_stream"),
    ("no_ffn_adapters", "use_role_primitive_ffn"),
    ("no_pressure", "use_vocab_pressure"),
)

DEFAULT_DATASET_DIR = ROOT / "artifacts" / "datasets" / "pater_synthetic"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "checkpoints" / "tiny_aux_overfit"
DEFAULT_CONFIG = ROOT / "configs" / "tiny_smoke.yaml"

# Loss weights, explicit and reported. LM and the categorical heads at full
# strength; pointers/links/multilabel at half to avoid swamping the LM.
LOSS_WEIGHTS: dict[str, float] = {
    "lm": 1.0,
    "primitive": 1.0,
    "support": 1.0,
    "tool_intent": 1.0,
    "schema": 1.0,
    "idk": 1.0,
    "role_to_primitive": 1.0,
    "role_ambiguity": 1.0,
    "verifier": 0.5,
    "evidence_pointer": 0.5,
    "predicate_event": 1.0,
    "event_token": 0.5,
    "argument_start": 0.5,
    "argument_end": 0.5,
    # Boundary-aware joint span objective: scores valid (start, end) pairs so the
    # two boundaries are trained jointly, not as independent pointers. This is the
    # span-END repair lever; it reuses the existing start/end logits (no new params).
    "argument_span": 0.5,
    "proto_role": 1.0,
    "arg_role": 1.0,
    "event_arg": 0.5,
    # Reasoning-supervision heads (opt-in).
    "entailment_state": 1.0,
    "proof_depth": 0.5,
    "rule_chain_length": 0.5,
    "supporting_fact": 0.5,
    "contradicting_fact": 0.5,
    "supporting_item": 0.5,
    "contradicting_item": 0.5,
}

# Max argument-span width (in subword tokens) the joint span objective scores and
# decodes over. External (FOLIO/ProofWriter) spans reach ~p95=15 tokens; 24 covers
# them with headroom while keeping the (start, width) grid small. Spans wider than
# this fall back to the marginal start/end CE only.
SPAN_MAX_WIDTH = 24

# Proto-role labels are sparse (~2 positive of 17 properties per argument), so
# the multi-label BCE uses a positive-class weight to keep recall from collapsing.
PROTO_ROLE_POS_WEIGHT = 5.0
ITEM_POS_WEIGHT = 10.0  # supporting/contradicting items are a rare minority of items


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    """Convert Paths (incl. lists/tuples of Paths) to str so checkpoints load with
    weights_only=True (no arbitrary unpickling)."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _index_in(vocab: Sequence[str], value: Any) -> int | None:
    if not isinstance(value, str) or value not in vocab:
        return None
    return vocab.index(value)


@dataclass(frozen=True)
class AuxTask:
    name: str
    logits_key: str
    num_classes: int
    kind: str  # "ce" | "bce"
    target: Callable[[dict[str, Any]], Any]  # class idx / 0|1 / float, or None to mask


# Per-record categorical/binary heads with a clean label. Class orderings come
# from the builder's canonical vocabularies, matching the head dimensions.
AUX_TASKS: tuple[AuxTask, ...] = (
    AuxTask("primitive", "primitive_class_logits", 14, "ce",
            lambda r: _index_in(builder.PRIMITIVE_CLASSES, r.get("labels", {}).get("primitive_class"))),
    AuxTask("support", "support_status_logits", 6, "ce",
            lambda r: _index_in(builder.SUPPORT_STATUSES, r.get("labels", {}).get("support_status"))),
    AuxTask("tool_intent", "tool_intent_logits", 5, "ce",
            lambda r: _index_in(builder.TOOL_INTENTS, r.get("labels", {}).get("tool_intent"))),
    AuxTask("schema", "schema_validity_logits", 2, "ce",
            lambda r: (int(bool(r["labels"]["schema_validity"]))
                       if isinstance(r.get("labels", {}).get("schema_validity"), bool) else None)),
    AuxTask("idk", "idk_abstention_logits", 4, "ce",
            lambda r: _index_in(builder.IDK_ACTIONS, r.get("labels", {}).get("idk_action"))),
    AuxTask("role_to_primitive", "role_to_primitive_logits", 14, "ce",
            lambda r: _index_in(builder.PRIMITIVE_CLASSES, r.get("labels", {}).get("primitive_class"))),
    AuxTask("role_ambiguity", "role_ambiguity_logits", 4, "ce",
            lambda r: _index_in(builder.ROLE_AMBIGUITY_CLASSES, r.get("labels", {}).get("role_ambiguity"))),
    AuxTask("verifier", "verifier_likelihood_logit", 1, "bce",
            lambda r: (float(bool(r["labels"]["verifier_accept"]))
                       if isinstance(r.get("labels", {}).get("verifier_accept"), bool) else None)),
)

# Reasoning-supervision classification heads (opt-in; --use-reasoning-heads). Targets
# come from ProofWriter proof metadata in record["reasoning"]; absent elsewhere (None
# -> masked). Only consumed when the model emits the matching logits.
ENTAILMENT_STATES = ("entailed", "refuted", "unknown")
_REASON_CAP = 5  # 0..4, 5+ bucketed -> 6 classes


def _reason(r: dict[str, Any], key: str):
    return (r.get("reasoning") or {}).get(key)


def _capped(r: dict[str, Any], key: str):
    v = _reason(r, key)
    return min(int(v), _REASON_CAP) if v is not None else None


REASONING_AUX_TASKS: tuple[AuxTask, ...] = (
    AuxTask("entailment_state", "entailment_state_logits", 3, "ce",
            lambda r: _index_in(ENTAILMENT_STATES, _reason(r, "entailment_state"))),
    AuxTask("proof_depth", "proof_depth_logits", _REASON_CAP + 1, "ce",
            lambda r: _capped(r, "proof_depth")),
    AuxTask("rule_chain_length", "rule_chain_length_logits", _REASON_CAP + 1, "ce",
            lambda r: _capped(r, "rule_chain_length")),
)
ALL_CE_BCE_TASKS = AUX_TASKS + REASONING_AUX_TASKS

# Reporting groups for event-role heads.
POINTER_HEADS = ("event_token", "argument_start", "argument_end", "evidence_pointer")
F1_HEADS = ("proto_role", "arg_role")
RANDOM_BASELINE = {
    "primitive": 1 / 14, "support": 1 / 6, "tool_intent": 1 / 5, "schema": 1 / 2,
    "idk": 1 / 4, "role_to_primitive": 1 / 14, "role_ambiguity": 1 / 4,
    "verifier": 1 / 2, "predicate_event": 1 / 4,
}
# Heads for which we track a full confusion matrix + macro-F1 (the bridge heads).
CONFUSION_HEADS = {"primitive": 14, "role_to_primitive": 14, "entailment_state": 3}
# Heads that get inverse-frequency class weights when --balanced-sampler / weights
# are requested (the primitive bridge).
CLASS_WEIGHTED_HEADS = ("primitive", "role_to_primitive")


class Dims(NamedTuple):
    n_evt: int
    n_arg: int
    n_props: int


def dims_from_config(config: PATERConfig) -> Dims:
    return Dims(config.num_event_registers, config.num_argument_registers, config.num_proto_role_properties)


# ---------------------------------------------------------------------------
# Dataset + tokenizer.
# ---------------------------------------------------------------------------
def load_record_files(paths: Sequence[Path], split: str | None = None) -> list[dict[str, Any]]:
    """Load explicit JSONL files (e.g. --dataset a.jsonl b.jsonl), optionally filtered by split."""

    records: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if split is None or record.get("split") == split:
                    records.append(record)
    return records


def load_records(dataset_dir: Path, split: str | None = None, auto_build: bool = True) -> list[dict[str, Any]]:
    files = sorted(dataset_dir.glob("*.jsonl"))
    if not files and auto_build:
        print(f"no dataset found in {dataset_dir}; building it via build_synthetic_pater_dataset (100/family)")
        families = builder.build_dataset(records_per_family=100, seed=0, val_ratio=0.1, test_ratio=0.1)
        builder.write_dataset(dataset_dir, families)
        files = sorted(dataset_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no dataset in {dataset_dir}. Run: python3 scripts/build_synthetic_pater_dataset.py --records-per-family 100")
    records: list[dict[str, Any]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if split is None or record.get("split") == split:
                    records.append(record)
    return records


def build_tokenizer(all_records: Sequence[dict[str, Any]]) -> ReferenceTokenizer:
    texts = [r["model_text"] for r in all_records if isinstance(r.get("model_text"), str)]
    return build_reference_tokenizer(extra_texts=texts)


def reconstruct_tokenizer(vocab: dict[str, int]) -> ReferenceTokenizer:
    spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
    id_to_token = {int(i): t for t, i in vocab.items()}
    return ReferenceTokenizer(token_to_id={t: int(i) for t, i in vocab.items()}, id_to_token=id_to_token, spec=spec)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ce_targets: dict[str, torch.Tensor]
    verifier_target: torch.Tensor
    verifier_mask: torch.Tensor
    evidence_gold: torch.Tensor
    er: dict[str, torch.Tensor]  # event-role structured targets
    num_lm_tokens: int

    def to(self, device: str) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            ce_targets={k: v.to(device) for k, v in self.ce_targets.items()},
            verifier_target=self.verifier_target.to(device),
            verifier_mask=self.verifier_mask.to(device),
            evidence_gold=self.evidence_gold.to(device),
            er={k: v.to(device) for k, v in self.er.items()},
            num_lm_tokens=self.num_lm_tokens,
        )


def render_input_only(record: dict[str, Any]) -> str:
    """Model input with the serialized event_graph and output target removed.

    This is the lower-leakage rendering for the input-only ablation: the model
    must predict the structured labels from text/evidence/formula/tools rather
    than reading them off the serialized event_graph. Token-span and evidence
    positions are unchanged because those blocks are rendered before the
    event_graph, so the dataset's stored spans stay valid.
    """

    return render_pater_prompt(
        text=record["text"],
        evidence=record["evidence"] or None,
        formula=record["formula"] or None,
        events=None,
        tools=record["tools"] or None,
        output_prelude="",
    )


RENDER_MODES = ("full", "input_only", "plain", "prefix")

# Fixed register-token prefix for the prefix_register baseline: the event/arg
# slot tokens plus the primitive <reg:*> register tokens, prepended as ordinary
# input tokens (compatibility registers) with no true side-state.
_REGISTER_PREFIX = " ".join(
    ["<evt:0>", "<evt:1>", "<evt:2>", "<evt:3>",
     "<arg:0>", "<arg:1>", "<arg:2>", "<arg:3>", "<arg:4>", "<arg:adjunct>"]
    + [t for t in builder.PRIMITIVE_SPECIAL_TOKENS if t.startswith("<reg:")]
)


def render_for_mode(record: dict[str, Any], render_mode: str) -> str:
    """full = serialized record; input_only = no event_graph/output; plain = raw
    text only; prefix = register-token prefix + input_only."""

    if render_mode == "full":
        return record["model_text"]
    if render_mode == "plain":
        return record["text"]
    base = render_input_only(record)
    if render_mode == "prefix":
        return f"{_REGISTER_PREFIX}\n{base}"
    return base  # input_only


def encode_ids(tokenizer: ReferenceTokenizer, record: dict[str, Any], max_len: int, render_mode: str = "full") -> list[int]:
    ids = tokenizer.encode(render_for_mode(record, render_mode), add_special_tokens=True)
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


class Encoded(NamedTuple):
    """Token ids plus, for offset-capable (HF) tokenizers, the char-offset map and
    the rendered text -- everything needed to relocate spans/evidence."""

    ids: list[int]
    offsets: list[tuple[int, int]] | None
    text: str


def encode_for_targets(tokenizer: Any, record: dict[str, Any], max_len: int, render_mode: str = "full") -> Encoded:
    """Encode the rendered input and, when the tokenizer exposes char offsets,
    carry the aligned offset map so subword span/evidence labels can be relocated
    by character range instead of by (BPE-fragile) id-subsequence matching."""

    text = render_for_mode(record, render_mode)
    if getattr(tokenizer, "supports_offsets", False):
        ids, offsets = tokenizer.encode_with_offsets(text, add_special_tokens=True)
        return Encoded(ids[:max_len], offsets[:max_len], text)
    return Encoded(tokenizer.encode(text, add_special_tokens=True)[:max_len], None, text)


def _evidence_gold_index(tokenizer: Any, record: dict[str, Any], enc: Encoded) -> int:
    evidence_ids = record.get("labels", {}).get("evidence_ids") or []
    if not evidence_ids:
        return -100
    if getattr(tokenizer, "supports_offsets", False):
        # Locate the evidence id's first occurrence in the rendered text and map
        # it to its first subword token (in-bounds + non-pad by construction).
        loc = H.locate_text_span(enc.text, enc.offsets, evidence_ids[0])
        return loc[0] if loc is not None else -100
    token_id = tokenizer.token_to_id.get(evidence_ids[0])
    if token_id is None:
        return -100
    for pos, tok in enumerate(enc.ids):
        if tok == token_id:
            return pos
    return -100


def _find_subseq(ids: Sequence[int], sub: tuple[int, ...]) -> tuple[int, int] | None:
    n = len(sub)
    if n == 0 or n > len(ids):
        return None
    for i in range(len(ids) - n + 1):
        if tuple(ids[i : i + n]) == sub:
            return (i, i + n - 1)
    return None


def _all_subseq(ids: Sequence[int], sub: tuple[int, ...]) -> list[tuple[int, int]]:
    n = len(sub)
    if n == 0 or n > len(ids):
        return []
    return [(i, i + n - 1) for i in range(len(ids) - n + 1) if tuple(ids[i : i + n]) == sub]


def all_span_occurrences(enc: "Encoded", span_text: str, tokenizer: Any) -> list[tuple[int, int]]:
    """Every token (start, end) range where the gold argument text occurs. Repeated
    logical atoms are co-referent mentions of the same argument, so the set-valued
    span objective credits any of them, not one arbitrary occurrence."""

    if getattr(enc, "offsets", None) is not None:
        spans: list[tuple[int, int]] = []
        cursor = 0
        while True:
            cs = enc.text.find(span_text, cursor)
            if cs < 0:
                break
            hit = H.locate_char_span(enc.offsets, cs, cs + len(span_text))
            if hit is not None:
                spans.append(hit)
            cursor = cs + 1
        return sorted(set(spans))
    sub = tuple(tokenizer.encode(span_text, add_special_tokens=False))
    return _all_subseq(enc.ids, sub)


def _event_role_targets(records: Sequence[dict[str, Any]], encoded: Sequence["Encoded"],
                        tokenizer: Any, dims: Dims, seq_len: int) -> dict[str, torch.Tensor]:
    """Build event-role targets by re-locating each predicate/argument span in the
    actual encoded sequence. This is render-mode agnostic (positions are correct
    whether or not there is a register prefix or stripped event_graph) and
    naturally masks spans that fall outside the truncated window.

    Offset-capable (HF/subword) tokenizers relocate spans by character range; the
    word-level reference tokenizer relocates by id-subsequence match. Either way
    a span that does not resolve stays masked (-100)."""

    bsz = len(records)
    predicate_event = torch.full((bsz, dims.n_evt), -100, dtype=torch.long)
    event_token = torch.full((bsz, dims.n_evt), -100, dtype=torch.long)
    arg_start = torch.full((bsz, dims.n_arg), -100, dtype=torch.long)
    arg_end = torch.full((bsz, dims.n_arg), -100, dtype=torch.long)
    # Set-valued span targets: True at every co-referent occurrence of the gold
    # argument text. arg_width carries the (shared) span token-width minus one.
    arg_start_mask = torch.zeros((bsz, dims.n_arg, seq_len), dtype=torch.bool)
    arg_end_mask = torch.zeros((bsz, dims.n_arg, seq_len), dtype=torch.bool)
    arg_width = torch.full((bsz, dims.n_arg), -100, dtype=torch.long)
    proto_role = torch.zeros((bsz, dims.n_arg, dims.n_props), dtype=torch.float)
    proto_role_mask = torch.zeros((bsz, dims.n_arg), dtype=torch.bool)
    event_arg = torch.zeros((bsz, dims.n_evt, dims.n_arg), dtype=torch.float)
    event_arg_mask = torch.zeros((bsz, dims.n_evt), dtype=torch.bool)

    use_offsets = getattr(tokenizer, "supports_offsets", False)

    def span_ids(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    def locate(enc: "Encoded", span_text: str) -> tuple[int, int] | None:
        if use_offsets:
            return H.locate_text_span(enc.text, enc.offsets, span_text)
        return _find_subseq(enc.ids, span_ids(span_text))

    for b, record in enumerate(records):
        enc = encoded[b]
        events = record.get("event_graph") or []
        if not events:
            continue
        event = events[0]  # synthetic records carry a single active event (slot 0)
        labels = record.get("labels", {})

        et = _index_in(builder.EVENT_TYPES, labels.get("event_type"))
        if et is not None:
            predicate_event[b, 0] = et

        predicate = event.get("predicate")
        if isinstance(predicate, str):
            found = locate(enc, predicate)
            if found is not None:
                event_token[b, 0] = found[0]

        arguments = event.get("arguments") or []
        event_arg_mask[b, 0] = True
        for j, arg in enumerate(arguments[: dims.n_arg]):
            event_arg[b, 0, j] = 1.0
            proto_role_mask[b, j] = True
            roles = set(arg.get("proto_roles") or [])
            for p, prop in enumerate(builder.PROTO_ROLE_PROPERTIES):
                if prop in roles:
                    proto_role[b, j, p] = 1.0
            span = arg.get("span")
            if isinstance(span, str):
                occurrences = all_span_occurrences(enc, span, tokenizer)
                occurrences = [(s, e) for (s, e) in occurrences if 0 <= s < seq_len and 0 <= e < seq_len]
                if occurrences:
                    first = min(occurrences)
                    arg_start[b, j] = first[0]
                    arg_end[b, j] = first[1]
                    arg_width[b, j] = first[1] - first[0]
                    for s, e in occurrences:
                        arg_start_mask[b, j, s] = True
                        arg_end_mask[b, j, e] = True

    return {
        "predicate_event": predicate_event,
        "event_token": event_token,
        "arg_start": arg_start,
        "arg_end": arg_end,
        "arg_start_mask": arg_start_mask,
        "arg_end_mask": arg_end_mask,
        "arg_width": arg_width,
        "proto_role": proto_role,
        "proto_role_mask": proto_role_mask,
        "event_arg": event_arg,
        "event_arg_mask": event_arg_mask,
    }


def _fact_pointer_targets(records: Sequence[dict[str, Any]], encoded: Sequence["Encoded"],
                          tokenizer: Any, seq_len: int) -> dict[str, torch.Tensor]:
    """Multi-label [B, T] masks over the token positions of supporting/contradicting
    fact texts (ProofWriter proof metadata). Records with no fact texts (unknown
    answers, non-reasoning records) get valid=False -> excluded from the loss."""

    bsz = len(encoded)
    sup = torch.zeros((bsz, seq_len)); con = torch.zeros((bsz, seq_len))
    sup_valid = torch.zeros(bsz, dtype=torch.bool); con_valid = torch.zeros(bsz, dtype=torch.bool)
    use_offsets = getattr(tokenizer, "supports_offsets", False)
    for b, (r, enc) in enumerate(zip(records, encoded)):
        reasoning = r.get("reasoning") or {}
        for key, mask, valid in (("supporting_fact_texts", sup, sup_valid),
                                  ("contradicting_fact_texts", con, con_valid)):
            texts = reasoning.get(key)
            if not texts:
                continue
            any_loc = False
            for t in texts:
                if use_offsets:
                    hit = H.locate_text_span(enc.text, enc.offsets, t)
                else:
                    sub = tuple(tokenizer.encode(t, add_special_tokens=False))
                    hit = _find_subseq(enc.ids, sub) if sub else None
                if hit and hit[1] < seq_len:
                    mask[b, hit[0]: hit[1] + 1] = 1.0
                    any_loc = True
            if any_loc:
                valid[b] = True
    return {"supporting_fact_mask": sup, "contradicting_fact_mask": con,
            "supporting_fact_valid": sup_valid, "contradicting_fact_valid": con_valid}


MAX_EVIDENCE_ITEMS = 24


def _evidence_item_targets(records: Sequence[dict[str, Any]], encoded: Sequence["Encoded"],
                           tokenizer: Any, seq_len: int) -> dict[str, torch.Tensor]:
    """Per-evidence-item targets for the support/refute heads. Each evidence item
    (a fact) becomes a slot with a [T] pooling mask over its token span; the target
    is item-level membership in supporting/contradicting_fact_texts. Records without
    a clean fact target (unknown answers, non-reasoning) are masked out."""

    bsz = len(encoded)
    I = MAX_EVIDENCE_ITEMS
    eim = torch.zeros((bsz, I, seq_len))
    sup = torch.zeros((bsz, I)); con = torch.zeros((bsz, I))
    item_valid = torch.zeros((bsz, I), dtype=torch.bool)
    rec_valid = torch.zeros(bsz, dtype=torch.bool)
    use_offsets = getattr(tokenizer, "supports_offsets", False)
    for b, (r, enc) in enumerate(zip(records, encoded)):
        reasoning = r.get("reasoning")
        if not reasoning:
            continue
        sup_texts = set(reasoning.get("supporting_fact_texts") or [])
        con_texts = set(reasoning.get("contradicting_fact_texts") or [])
        if not sup_texts and not con_texts:
            continue  # unknown / no clean target
        slot = 0
        for ev in (r.get("evidence") or []):
            if slot >= I:
                break
            text = ev.get("text")
            if not text:
                continue
            if use_offsets:
                hit = H.locate_text_span(enc.text, enc.offsets, text)
            else:
                subseq = tuple(tokenizer.encode(text, add_special_tokens=False))
                hit = _find_subseq(enc.ids, subseq) if subseq else None
            if not hit or hit[1] >= seq_len:
                continue
            eim[b, slot, hit[0]: hit[1] + 1] = 1.0
            item_valid[b, slot] = True
            if text in sup_texts:
                sup[b, slot] = 1.0
            if text in con_texts:
                con[b, slot] = 1.0
            slot += 1
        if item_valid[b].any():
            rec_valid[b] = True
    return {"evidence_item_mask": eim, "supporting_item_target": sup, "contradicting_item_target": con,
            "evidence_item_valid": item_valid, "evidence_record_valid": rec_valid}


def collate(tokenizer: ReferenceTokenizer, records: Sequence[dict[str, Any]], max_len: int, dims: Dims,
            render_mode: str = "full") -> Batch:
    encoded = [encode_for_targets(tokenizer, r, max_len, render_mode) for r in records]
    seq_len = max(len(e.ids) for e in encoded)
    bsz = len(encoded)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long)
    for row, enc in enumerate(encoded):
        input_ids[row, : len(enc.ids)] = torch.tensor(enc.ids, dtype=torch.long)
        attention_mask[row, : len(enc.ids)] = 1

    ce_targets: dict[str, torch.Tensor] = {}
    for task in ALL_CE_BCE_TASKS:
        if task.kind != "ce":
            continue
        col = [(-100 if task.target(r) is None else int(task.target(r))) for r in records]
        ce_targets[task.name] = torch.tensor(col, dtype=torch.long)

    verifier_task = next(t for t in AUX_TASKS if t.name == "verifier")
    vt = [verifier_task.target(r) for r in records]
    verifier_target = torch.tensor([0.0 if v is None else float(v) for v in vt], dtype=torch.float)
    verifier_mask = torch.tensor([v is not None for v in vt], dtype=torch.bool)

    evidence_gold = torch.tensor(
        [_evidence_gold_index(tokenizer, r, enc) for r, enc in zip(records, encoded)], dtype=torch.long)

    er = _event_role_targets(records, encoded, tokenizer, dims, seq_len)
    er.update(_fact_pointer_targets(records, encoded, tokenizer, seq_len))
    er.update(_evidence_item_targets(records, encoded, tokenizer, seq_len))
    num_lm_tokens = int(attention_mask[:, 1:].sum().item())
    return Batch(input_ids, attention_mask, ce_targets, verifier_target, verifier_mask, evidence_gold, er, num_lm_tokens)


def span_label_stats(tokenizer: Any, records: Sequence[dict[str, Any]], max_len: int, dims: Dims,
                     render_mode: str, batch_size: int = 16) -> dict[str, Any]:
    """Validate span/evidence label migration under the active tokenizer: how many
    source spans relocate, and whether any located index is out of bounds or
    points at a pad token. For subword tokenizers this is the key check that the
    argument_start/end, event_token, and evidence-pointer labels survive."""

    pad_id = tokenizer.pad_token_id
    source = {"arg": 0, "predicate": 0, "evidence": 0}
    for r in records:
        events = r.get("event_graph") or []
        if events:
            if isinstance(events[0].get("predicate"), str):
                source["predicate"] += 1
            for arg in (events[0].get("arguments") or [])[: dims.n_arg]:
                if isinstance(arg.get("span"), str):
                    source["arg"] += 1
        if r.get("labels", {}).get("evidence_ids"):
            source["evidence"] += 1

    located = {"arg": 0, "predicate": 0, "evidence": 0}
    out_of_bounds = 0
    point_to_pad = 0

    def scan(t: torch.Tensor, ids: torch.Tensor, seq_len: int) -> int:
        nonlocal out_of_bounds, point_to_pad
        count = 0
        for b in range(t.shape[0]):
            for v in t[b].tolist():
                if v == -100:
                    continue
                count += 1
                if v < 0 or v >= seq_len:
                    out_of_bounds += 1
                elif int(ids[b, v]) == pad_id:
                    point_to_pad += 1
        return count

    for i in range(0, len(records), batch_size):
        batch = collate(tokenizer, records[i:i + batch_size], max_len, dims, render_mode)
        ids, seq_len = batch.input_ids, batch.input_ids.shape[1]
        located["predicate"] += scan(batch.er["event_token"], ids, seq_len)
        located["arg"] += scan(batch.er["arg_start"], ids, seq_len)
        located["evidence"] += scan(batch.evidence_gold.unsqueeze(1), ids, seq_len)

    rate = {k: (round(located[k] / source[k], 4) if source[k] else None) for k in source}
    return {"source": source, "located": located, "located_rate": rate,
            "out_of_bounds": out_of_bounds, "point_to_pad": point_to_pad}


# ---------------------------------------------------------------------------
# Loss helpers.
# ---------------------------------------------------------------------------
def _ce_flat(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """CE over the last dim, flattening leading dims; None if all masked."""

    num_classes = logits.shape[-1]
    flat_target = target.reshape(-1)
    if not (flat_target != -100).any():
        return None
    return F.cross_entropy(logits.reshape(-1, num_classes), flat_target, ignore_index=-100)


def _bce_ml(logits: torch.Tensor, target: torch.Tensor, slot_mask: torch.Tensor,
            pos_weight: float | None = None) -> torch.Tensor | None:
    if not slot_mask.any():
        return None
    pw = logits.new_tensor(pos_weight) if pos_weight is not None else None
    return F.binary_cross_entropy_with_logits(logits[slot_mask], target[slot_mask], pos_weight=pw)


def _span_pair_scores(start_logits: torch.Tensor, end_logits: torch.Tensor,
                      max_width: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Banded joint span scores: score[b,a,i,w] = start[b,a,i] + end[b,a,i+w].

    Only widths 0..W-1 (end = start + w, so end >= start by construction) are
    scored; the band that runs off the end of the sequence is masked. start/end
    logits already carry -1e9 at pad positions (attention mask in the heads), so
    pad starts/ends stay out of the argmax. Returns (scores[B,A,T*W], band, W).
    """

    bsz, n_arg, seq_len = start_logits.shape
    width = min(max_width, seq_len)
    pos = torch.arange(seq_len, device=start_logits.device)
    offs = torch.arange(width, device=start_logits.device)
    end_idx = pos[:, None] + offs[None, :]            # (T, W)
    band = end_idx < seq_len                          # (T, W) valid end in range
    end_gather = end_logits[:, :, end_idx.clamp(max=seq_len - 1)]  # (B, A, T, W)
    scores = start_logits.unsqueeze(-1) + end_gather
    scores = scores.masked_fill(~band[None, None], -1.0e9)
    return scores.reshape(bsz, n_arg, seq_len * width), band, width


def _span_pair_loss(start_logits: torch.Tensor, end_logits: torch.Tensor,
                    arg_start: torch.Tensor, arg_end: torch.Tensor, max_width: int) -> torch.Tensor | None:
    """CE over the flattened (start, width) grid against the gold (start, end) pair."""

    if not (arg_start != -100).any():
        return None
    scores, _band, width = _span_pair_scores(start_logits, end_logits, max_width)
    bsz, n_arg, _ = scores.shape
    span_w = arg_end - arg_start
    gold = arg_start * width + span_w
    invalid = (arg_start < 0) | (arg_end < 0) | (span_w < 0) | (span_w >= width)
    gold = torch.where(invalid, torch.full_like(gold, -100), gold)
    if not (gold != -100).any():
        return None
    return F.cross_entropy(scores.reshape(bsz * n_arg, -1), gold.reshape(bsz * n_arg), ignore_index=-100)


def _multipositive_ce(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor | None:
    """Multi-positive cross-entropy: loss = logsumexp(all) - logsumexp(positives),
    averaged over rows with >=1 positive. Credits ANY valid position, so repeated
    co-referent argument mentions are all correct targets, not one arbitrary one."""

    has = pos_mask.any(dim=-1)
    if not has.any():
        return None
    all_lse = torch.logsumexp(logits, dim=-1)
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    pos_lse = torch.logsumexp(pos_logits, dim=-1)
    return (all_lse - pos_lse)[has].mean()


def _span_pair_pos_mask(start_mask: torch.Tensor, arg_width: torch.Tensor, seq_len: int, width: int) -> torch.Tensor:
    """Positive (start, width) cells in the banded T*W grid: every valid start with
    the slot's shared span width. Returns a (B, A, T*W) bool mask."""

    bsz, n_arg, _ = start_mask.shape
    w0 = arg_width.clamp(min=0)                                   # (B, A)
    t_idx = torch.arange(seq_len, device=start_mask.device).view(1, 1, seq_len)
    flat_idx = t_idx * width + w0.unsqueeze(-1)                   # (B, A, T)
    ok = start_mask & (arg_width.unsqueeze(-1) >= 0) & (arg_width.unsqueeze(-1) < width) & (flat_idx < seq_len * width)
    pos = torch.zeros((bsz, n_arg, seq_len * width), dtype=torch.bool, device=start_mask.device)
    pos.scatter_(-1, flat_idx.clamp(max=seq_len * width - 1), ok)
    return pos


def _span_pair_loss_set(start_logits: torch.Tensor, end_logits: torch.Tensor,
                        start_mask: torch.Tensor, arg_width: torch.Tensor, max_width: int) -> torch.Tensor | None:
    """Set-valued boundary-aware joint span loss over all valid (start, end) pairs."""

    if not start_mask.any():
        return None
    scores, _band, width = _span_pair_scores(start_logits, end_logits, max_width)
    bsz, n_arg, _ = scores.shape
    seq_len = start_logits.shape[-1]
    pos = _span_pair_pos_mask(start_mask, arg_width, seq_len, width)
    return _multipositive_ce(scores.reshape(bsz * n_arg, -1), pos.reshape(bsz * n_arg, -1))


def joint_span_decode(start_logits: torch.Tensor, end_logits: torch.Tensor,
                      max_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the highest-scoring valid (start, end) pair per argument slot."""

    scores, _band, width = _span_pair_scores(start_logits, end_logits, max_width)
    flat = scores.argmax(dim=-1)                      # (B, A)
    pred_start = flat // width
    pred_end = pred_start + (flat % width)
    return pred_start, pred_end


def record_span_widths(tokenizer: Any, record: dict[str, Any], max_len: int, render_mode: str) -> list[int]:
    """Located argument-span widths (subword tokens) for one record, for the
    span-focused sampler and the metric breakdown. Uses offsets when available."""

    enc = encode_for_targets(tokenizer, record, max_len, render_mode)
    use_offsets = getattr(enc, "offsets", None) is not None
    widths: list[int] = []
    for event in (record.get("event_graph") or [])[:1]:
        for arg in (event.get("arguments") or []):
            span = arg.get("span")
            if not isinstance(span, str):
                continue
            if use_offsets:
                hit = H.locate_text_span(enc.text, enc.offsets, span)
            else:
                sub = tuple(tokenizer.encode(span, add_special_tokens=False))
                hit = _find_subseq(enc.ids, sub) if sub else None
            if hit is not None:
                widths.append(hit[1] - hit[0] + 1)
    return widths


def _span_width_factor(width: int) -> float:
    """Per-span sampling factor, data-driven from the 450M eval breakdown: the
    hard bucket is SHORT multi-subword spans (width 2-4, predominantly synthetic),
    which sit at ~0.36 exact while long external spans (width 5-15) are already
    ~0.9. So oversample the short multi-subword spans, not the already-solved long
    or external ones."""

    if width <= 1:
        return 1.0          # single subword: end == start, not the bottleneck
    if width <= 4:
        return 3.0          # the hard bucket
    if width <= 15:
        return 1.0          # long external spans: already well learned
    return 1.5              # very long tail: mildly weak


def record_span_weight(tokenizer: Any, record: dict[str, Any], max_len: int, render_mode: str) -> float:
    """Sampling weight for the span-focused sampler: heavier for records carrying
    the hard short multi-subword argument spans (see _span_width_factor)."""

    widths = record_span_widths(tokenizer, record, max_len, render_mode)
    if not widths:
        return 0.5
    return float(sum(_span_width_factor(w) for w in widths))


def compute_losses(output: Any, batch: Batch, class_weights: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {"lm": output.loss}
    aux = output.aux_outputs
    class_weights = class_weights or {}

    for task in ALL_CE_BCE_TASKS:
        if task.logits_key not in aux:
            continue  # reasoning heads only present when --use-reasoning-heads
        if task.kind == "ce":
            target = batch.ce_targets[task.name]
            if (target != -100).any():
                weight = class_weights.get(task.name)
                if weight is not None:
                    weight = weight.to(aux[task.logits_key].device)
                losses[task.name] = F.cross_entropy(aux[task.logits_key], target, weight=weight, ignore_index=-100)
        else:  # bce (verifier)
            if batch.verifier_mask.any():
                logit = aux[task.logits_key].squeeze(-1)[batch.verifier_mask]
                losses[task.name] = F.binary_cross_entropy_with_logits(logit, batch.verifier_target[batch.verifier_mask])

    if (batch.evidence_gold != -100).any():
        losses["evidence_pointer"] = F.cross_entropy(aux["evidence_pointer_logits"], batch.evidence_gold, ignore_index=-100)

    # Reasoning fact pointers: multi-label BCE over real-token positions of the
    # supporting/contradicting fact spans (ProofWriter proof metadata). Only valid
    # rows (records that have those facts; unknown answers excluded) contribute.
    if "supporting_fact_pointer_logits" in aux:
        am = batch.attention_mask.float()
        for name, logit_key in (("supporting_fact", "supporting_fact_pointer_logits"),
                                 ("contradicting_fact", "contradicting_fact_pointer_logits")):
            valid = batch.er[f"{name}_valid"]
            if valid.any():
                logit = aux[logit_key][valid]
                target = batch.er[f"{name}_mask"][valid]
                m = am[valid]
                bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
                losses[name] = (bce * m).sum() / m.sum().clamp(min=1.0)

    # Per-evidence-item support/refute: BCE over real item slots of valid records.
    # Supporting/contradicting items are a small minority of items -> pos_weight so
    # the head does not collapse to all-negative.
    if "supporting_item_logits" in aux:
        valid = batch.er["evidence_item_valid"] & batch.er["evidence_record_valid"].unsqueeze(1)
        if valid.any():
            pw = aux["supporting_item_logits"].new_tensor(ITEM_POS_WEIGHT)
            for name, logit_key, tkey in (("supporting_item", "supporting_item_logits", "supporting_item_target"),
                                          ("contradicting_item", "contradicting_item_logits", "contradicting_item_target")):
                losses[name] = F.binary_cross_entropy_with_logits(aux[logit_key][valid], batch.er[tkey][valid], pos_weight=pw)

    er = batch.er
    # predicate / event_token point at a single token; keep single-target CE.
    for name, key, target_key in (
        ("predicate_event", "predicate_event_logits", "predicate_event"),
        ("event_token", "event_token_logits", "event_token"),
    ):
        loss = _ce_flat(aux[key], er[target_key])
        if loss is not None:
            losses[name] = loss

    # Argument spans use set-valued (multi-positive) targets: repeated co-referent
    # mentions are all correct, so the pointer is not punished for a non-identifiable
    # single occurrence. Marginal start/end plus the boundary-aware joint pair loss.
    bsz, n_arg, seq_len = aux["argument_start_logits"].shape
    start_loss = _multipositive_ce(aux["argument_start_logits"].reshape(bsz * n_arg, seq_len),
                                   er["arg_start_mask"].reshape(bsz * n_arg, seq_len))
    if start_loss is not None:
        losses["argument_start"] = start_loss
    end_loss = _multipositive_ce(aux["argument_end_logits"].reshape(bsz * n_arg, seq_len),
                                 er["arg_end_mask"].reshape(bsz * n_arg, seq_len))
    if end_loss is not None:
        losses["argument_end"] = end_loss
    span_loss = _span_pair_loss_set(aux["argument_start_logits"], aux["argument_end_logits"],
                                    er["arg_start_mask"], er["arg_width"], SPAN_MAX_WIDTH)
    if span_loss is not None:
        losses["argument_span"] = span_loss

    for name, key in (("proto_role", "proto_role_logits"), ("arg_role", "arg_role_logits")):
        loss = _bce_ml(aux[key], er["proto_role"], er["proto_role_mask"], pos_weight=PROTO_ROLE_POS_WEIGHT)
        if loss is not None:
            losses[name] = loss

    loss = _bce_ml(aux["event_arg_logits"], er["event_arg"], er["event_arg_mask"])
    if loss is not None:
        losses["event_arg"] = loss
    return losses


def total_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    # Skip weight-0 components entirely: 0.0 * NaN = NaN in IEEE 754, so a
    # zero-weighted head that produces NaN (e.g. span CE with softmax overflow)
    # would poison the total even though it's excluded from training.
    terms = [
        w * v
        for name, v in losses.items()
        if (w := LOSS_WEIGHTS.get(name, 1.0)) > 0
    ]
    return sum(terms) if terms else losses[next(iter(losses))].new_zeros(())


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def _flat_acc(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    flat_target = target.reshape(-1)
    valid = flat_target != -100
    if not valid.any():
        return 0, 0
    pred = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
    return int((pred[valid] == flat_target[valid]).sum()), int(valid.sum())


class MetricAccumulator:
    def __init__(self) -> None:
        self.correct: dict[str, int] = {}
        self.total: dict[str, int] = {}
        self.tp: dict[str, int] = {}
        self.fp: dict[str, int] = {}
        self.fn: dict[str, int] = {}
        self.confusion: dict[str, list[list[int]]] = {}
        self.lm_loss_sum = 0.0
        self.lm_tokens = 0

    def _confuse(self, name: str, gold: torch.Tensor, pred: torch.Tensor, num_classes: int) -> None:
        cm = self.confusion.setdefault(name, [[0] * num_classes for _ in range(num_classes)])
        for g, p in zip(gold.tolist(), pred.tolist()):
            cm[g][p] += 1

    @classmethod
    def empty(cls) -> "MetricAccumulator":
        return cls()

    def _add(self, name: str, correct: int, total: int) -> None:
        self.correct[name] = self.correct.get(name, 0) + correct
        self.total[name] = self.total.get(name, 0) + total

    def _add_f1(self, name: str, tp: int, fp: int, fn: int) -> None:
        self.tp[name] = self.tp.get(name, 0) + tp
        self.fp[name] = self.fp.get(name, 0) + fp
        self.fn[name] = self.fn.get(name, 0) + fn

    def update(self, output: Any, batch: Batch) -> None:
        aux = output.aux_outputs
        self.lm_loss_sum += float(output.loss.detach()) * batch.num_lm_tokens
        self.lm_tokens += batch.num_lm_tokens

        for task in ALL_CE_BCE_TASKS:
            if task.logits_key not in aux:
                continue
            logits = aux[task.logits_key]
            if task.kind == "ce":
                target = batch.ce_targets[task.name]
                valid = target != -100
                if valid.any():
                    pred = logits.argmax(dim=-1)
                    self._add(task.name, int((pred[valid] == target[valid]).sum()), int(valid.sum()))
                    if task.name in CONFUSION_HEADS:
                        self._confuse(task.name, target[valid], pred[valid], task.num_classes)
            else:
                mask = batch.verifier_mask
                if mask.any():
                    pred = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                    self._add(task.name, int((pred[mask] == batch.verifier_target[mask]).sum()), int(mask.sum()))

        gold = batch.evidence_gold
        if (gold != -100).any():
            valid = gold != -100
            pred = aux["evidence_pointer_logits"].argmax(dim=-1)
            self._add("evidence_pointer", int((pred[valid] == gold[valid]).sum()), int(valid.sum()))

        er = batch.er
        # Slot / pointer accuracies.
        self._add("predicate_event", *_flat_acc(aux["predicate_event_logits"], er["predicate_event"]))
        self._add("event_token", *_flat_acc(aux["event_token_logits"], er["event_token"]))
        self._add("argument_start", *_flat_acc(aux["argument_start_logits"], er["arg_start"]))
        self._add("argument_end", *_flat_acc(aux["argument_end_logits"], er["arg_end"]))

        # Argument span accuracy, strict (== the first/canonical gold occurrence)
        # and any-occurrence (the predicted boundary is any co-referent mention).
        # Any-occurrence is the meaningful metric for repeated logical atoms; strict
        # is kept as a diagnostic. Both for independent and joint decode.
        st, en = er["arg_start"], er["arg_end"]
        valid = (st != -100) & (en != -100)
        if valid.any():
            ps = aux["argument_start_logits"].argmax(dim=-1)
            pe = aux["argument_end_logits"].argmax(dim=-1)
            start_mask, end_mask, width = er["arg_start_mask"], er["arg_end_mask"], er["arg_width"]

            def _in(mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
                return mask.gather(-1, pred.clamp(min=0).unsqueeze(-1)).squeeze(-1).bool()

            js, je = joint_span_decode(aux["argument_start_logits"], aux["argument_end_logits"], SPAN_MAX_WIDTH)
            # strict (first gold occurrence)
            self._add("arg_span_exact", int(((ps == st) & (pe == en))[valid].sum()), int(valid.sum()))
            self._add("arg_span_joint", int(((js == st) & (je == en))[valid].sum()), int(valid.sum()))
            # any-occurrence
            sa = _in(start_mask, ps)
            ea = _in(end_mask, pe)
            self._add("argument_start_any", int(sa[valid].sum()), int(valid.sum()))
            self._add("argument_end_any", int(ea[valid].sum()), int(valid.sum()))
            span_any = sa & (pe - ps == width)
            self._add("arg_span_any", int(span_any[valid].sum()), int(valid.sum()))
            jspan_any = _in(start_mask, js) & (je - js == width)
            self._add("arg_span_joint_any", int(jspan_any[valid].sum()), int(valid.sum()))

        # Event-arg link: element-wise accuracy over labeled event rows.
        mask = er["event_arg_mask"]
        if mask.any():
            pred = (torch.sigmoid(aux["event_arg_logits"]) > 0.5).float()[mask]
            tgt = er["event_arg"][mask]
            self._add("event_arg", int((pred == tgt).sum()), int(pred.numel()))

        # Multi-label micro-F1 for proto-role / arg-role.
        rmask = er["proto_role_mask"]
        if rmask.any():
            tgt = er["proto_role"][rmask]
            for name, key in (("proto_role", "proto_role_logits"), ("arg_role", "arg_role_logits")):
                pred = (torch.sigmoid(aux[key][rmask]) > 0.5).float()
                tp = int(((pred == 1) & (tgt == 1)).sum())
                fp = int(((pred == 1) & (tgt == 0)).sum())
                fn = int(((pred == 0) & (tgt == 1)).sum())
                self._add_f1(name, tp, fp, fn)

        # Reasoning fact pointers: multi-label micro-F1 over real-token positions.
        if "supporting_fact_pointer_logits" in aux:
            am = batch.attention_mask.bool()
            for name, key in (("supporting_fact", "supporting_fact_pointer_logits"),
                              ("contradicting_fact", "contradicting_fact_pointer_logits")):
                valid = er[f"{name}_valid"]
                if valid.any():
                    sel = am[valid]
                    pred = (torch.sigmoid(aux[key][valid]) > 0.5).float()[sel]
                    tgt = er[f"{name}_mask"][valid][sel]
                    self._add_f1(name, int(((pred == 1) & (tgt == 1)).sum()),
                                 int(((pred == 1) & (tgt == 0)).sum()), int(((pred == 0) & (tgt == 1)).sum()))

        # Per-evidence-item support/refute micro-F1 over valid item slots.
        if "supporting_item_logits" in aux:
            valid = er["evidence_item_valid"] & er["evidence_record_valid"].unsqueeze(1)
            if valid.any():
                for name, key, tkey in (("supporting_item", "supporting_item_logits", "supporting_item_target"),
                                        ("contradicting_item", "contradicting_item_logits", "contradicting_item_target")):
                    pred = (torch.sigmoid(aux[key][valid]) > 0.5).float()
                    tgt = er[tkey][valid]
                    self._add_f1(name, int(((pred == 1) & (tgt == 1)).sum()),
                                 int(((pred == 1) & (tgt == 0)).sum()), int(((pred == 0) & (tgt == 1)).sum()))

    def accuracy(self, name: str) -> float | None:
        if self.total.get(name, 0) == 0:
            return None
        return self.correct[name] / self.total[name]

    def macro_f1(self, name: str) -> float | None:
        cm = self.confusion.get(name)
        if not cm:
            return None
        f1s = []
        for c in range(len(cm)):
            tp = cm[c][c]
            fp = sum(cm[g][c] for g in range(len(cm))) - tp
            fn = sum(cm[c]) - tp
            if tp + fp + fn == 0:
                continue  # class never present nor predicted
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
        return sum(f1s) / len(f1s) if f1s else None

    def per_class_recall(self, name: str, labels: Sequence[str]) -> dict[str, tuple[int, int]]:
        cm = self.confusion.get(name)
        if not cm:
            return {}
        return {labels[c]: (cm[c][c], sum(cm[c])) for c in range(len(cm)) if sum(cm[c]) > 0}

    def per_class_prf(self, name: str, labels: Sequence[str]) -> dict[str, dict[str, float | int]]:
        cm = self.confusion.get(name)
        if not cm:
            return {}
        out: dict[str, dict[str, float | int]] = {}
        for c in range(len(cm)):
            support = sum(cm[c])
            predicted = sum(cm[g][c] for g in range(len(cm)))
            tp = cm[c][c]
            fp = predicted - tp
            fn = support - tp
            if support == 0 and predicted == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            out[labels[c]] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return out

    def top_confusions(self, name: str, labels: Sequence[str], k: int = 5) -> list[tuple[str, str, int]]:
        cm = self.confusion.get(name)
        if not cm:
            return []
        offdiag = [(labels[g], labels[p], cm[g][p]) for g in range(len(cm)) for p in range(len(cm))
                   if g != p and cm[g][p] > 0]
        return sorted(offdiag, key=lambda t: -t[2])[:k]

    def micro_f1(self, name: str) -> tuple[float, float, float] | None:
        tp, fp, fn = self.tp.get(name, 0), self.fp.get(name, 0), self.fn.get(name, 0)
        if tp + fp + fn == 0:
            return None
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return f1, precision, recall

    def lm_loss(self) -> float:
        return self.lm_loss_sum / self.lm_tokens if self.lm_tokens else float("nan")


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return requested


def make_model(config: PATERConfig, tokenizer: ReferenceTokenizer, device: str) -> PATERForCausalLM:
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
    return PATERForCausalLM(config).to(device)


def freeze_warmstart_backbone(model: PATERForCausalLM) -> tuple[int, int, int]:
    """Freeze the pretrained Qwen backbone (token embeddings — incl. the tied LM
    head, self-attn incl. q/k-norm, base SwiGLU FFNs, input/post-attn/final
    RMSNorms) PLUS the PAT-ER→token injection entry paths (role_fuse, primitive_fuse,
    adapter up-projections, vocab-pressure proj.1).

    The injection entry paths are zero-initialized by zero_init_warmstart_side_state
    and must also be FROZEN during Stage 3: because role_fuse input includes the full
    backbone hidden state x (magnitude ~1.0), d(LM)/d(role_fuse.weight) is large even
    at init, and one Adam step at lr=8e-5 creates a ~1.7× perturbation to the residual
    stream — immediately corrupting the pretrained LM.

    Frozen during Stage 3:
      backbone  — all Qwen-loaded params (self-attn, FFN, norms, embedding)
      injection — role_fuse, primitive_fuse (zero-weight, protected from large gradient)
      adapters  — adapter up-projections (zero-weight, protected)
      pressure  — vocab pressure proj.1 (zero-weight, protected)

    Trainable during Stage 3 (register attachment):
      event_read / role_token_read   — register←token and token←register cross-attn
      primitive_read / primitive_token_read
      register_bank                  — learned register parameters
      ffn.adapters.*.down            — adapter read paths (non-injecting)
      role/primitive/mix_pressure.proj.0 — pressure bottleneck projections
      aux heads                      — all aux head classification layers

    The injection entry paths open in Stage 4 (partial unfreeze at small lr).
    Returns (frozen_numel, trainable_numel, n_trainable_params)."""

    frozen_names = set(WS.qwen_backbone_name_map(model.config.num_hidden_layers)) | {"embed_tokens.weight"}
    injection_frozen = set()
    for name in dict(model.named_parameters()):
        if (
            name.endswith("role_fuse.weight")
            or name.endswith("primitive_fuse.weight")
            or (".ffn.adapters." in name and name.endswith(".up.weight"))
            or name in {
                "role_pressure.proj.1.weight",
                "primitive_pressure.proj.1.weight",
                "mix_pressure.proj.1.weight",
            }
        ):
            injection_frozen.add(name)
    all_frozen = frozen_names | injection_frozen
    frozen_numel = trainable_numel = n_trainable = 0
    for name, p in model.named_parameters():
        if name in all_frozen:
            p.requires_grad_(False)
            frozen_numel += p.numel()
        else:
            trainable_numel += p.numel()
            n_trainable += 1
    return frozen_numel, trainable_numel, n_trainable


_INJECT_ENTRY_NAMES = {
    "role_pressure.proj.1.weight",
    "primitive_pressure.proj.1.weight",
    "mix_pressure.proj.1.weight",
}


def open_injection_paths(model: PATERForCausalLM, which: list[str]) -> tuple[int, list[str]]:
    """Selectively unfreeze PAT-ER injection entry paths after Stage 3.

    ``which`` is a list of path-group identifiers:
      "role_fuse"      — role_fuse.weight (role register → token injection)
      "primitive_fuse" — primitive_fuse.weight (primitive register → token injection)
      "adapter_up"     — all FFN adapter up-projections (zero-init bottleneck)
      "pressure_proj1" — vocab pressure proj.1 (all three heads)

    Returns (n_opened, list of opened param names). Call this AFTER
    freeze_warmstart_backbone so the injection paths start frozen, then
    selectively re-enable only the desired set for Stage 4 opening.
    """
    which_set = set(which)
    opened = []
    for name, p in model.named_parameters():
        match = (
            ("role_fuse" in which_set and name.endswith("role_fuse.weight"))
            or ("primitive_fuse" in which_set and name.endswith("primitive_fuse.weight"))
            or ("adapter_up" in which_set and ".ffn.adapters." in name and name.endswith(".up.weight"))
            or ("pressure_proj1" in which_set and name in _INJECT_ENTRY_NAMES)
        )
        if match:
            p.requires_grad_(True)
            opened.append(name)
    return len(opened), opened


def open_top_backbone_layers(model: PATERForCausalLM, n_top: int) -> tuple[int, list[str]]:
    """Unfreeze Qwen backbone params in the top n_top layers (Stage 5B).

    Unfreezes the FULL Qwen decoder layer set (self-attn q/k/v/o + q/k-norm,
    base FFN gate/up/down, input/post-attn norms) for layers >= num_layers - n_top.
    Everything else (lower backbone layers, embedding, norm.weight) stays frozen.

    Returns (n_opened, list of opened param names).
    """
    import re as _re_bb
    num_layers = model.config.num_hidden_layers
    upper_start = max(0, num_layers - n_top)
    backbone_all = set(WS.qwen_backbone_name_map(num_layers)) | {"embed_tokens.weight"}
    opened = []
    for name, p in model.named_parameters():
        if name not in backbone_all:
            continue
        m = _re_bb.match(r"layers\.(\d+)\.", name)
        if m and int(m.group(1)) >= upper_start:
            p.requires_grad_(True)
            opened.append(name)
    return len(opened), opened


def open_upper_layer_adapters(model: PATERForCausalLM, n_upper: int) -> tuple[int, list[str]]:
    """Unfreeze FFN adapter up+down weights in the top n_upper layers (Stage 5A).

    The LowRankAdapter has:
      down.weight [rank, h]  — input projection (Stage-3 trained values, non-zero)
      up.weight   [h, rank]  — output projection (zero-init, blocks gradient to down)

    Because up.weight starts at zero, d(loss)/d(down.weight) = up.weight^T * upstream = 0.
    No explicit detach is needed: the adapter opens gradually and safely as up.weight grows
    from zero at adapter_lr. The backbone residual stream is protected by the same
    zero-init principle that governs injection-only Stage 4 fuse tensors.

    Backbone (self-attn, base FFN, norms), injection fuse tensors (Stage-4 values),
    registers, cross-attn, aux heads, gate, and lower-layer adapters all stay frozen.

    Returns (n_opened, opened_param_names).
    """
    import re as _re_mod
    num_layers = model.config.num_hidden_layers
    upper_start = max(0, num_layers - n_upper)
    opened = []
    for name, p in model.named_parameters():
        m = _re_mod.match(r"layers\.(\d+)\.ffn\.adapters\.\d+\.(up|down)\.weight$", name)
        if m and int(m.group(1)) >= upper_start:
            p.requires_grad_(True)
            opened.append(name)
    return len(opened), opened


SNAPSHOT_HEADS = [
    "primitive", "support", "tool_intent", "schema", "idk", "verifier",
    "role_to_primitive", "role_ambiguity", "predicate_event",
    "event_token", "argument_start", "argument_end", "argument_start_any", "argument_end_any",
    "arg_span_exact", "arg_span_joint", "arg_span_any", "arg_span_joint_any", "event_arg",
    "evidence_pointer",
]


def _print_snapshot(acc: MetricAccumulator) -> None:
    print(f"  lm_loss={acc.lm_loss():.4f}")
    for name in SNAPSHOT_HEADS:
        a = acc.accuracy(name)
        base = RANDOM_BASELINE.get(name)
        base_str = f" (random={base:.3f})" if base else ""
        print(f"  {name:<16} acc={a:.3f}{base_str}" if a is not None else f"  {name:<16} acc=n/a")
    for name in F1_HEADS:
        f1 = acc.micro_f1(name)
        print(f"  {name:<16} microF1={f1[0]:.3f} (P={f1[1]:.3f} R={f1[2]:.3f})" if f1 else f"  {name:<16} F1=n/a")
    for name in CONFUSION_HEADS:
        mf1 = acc.macro_f1(name)
        print(f"  {name:<16} macroF1={mf1:.3f}" if mf1 is not None else f"  {name:<16} macroF1=n/a")


def train(args: argparse.Namespace) -> Path:
    global SPAN_MAX_WIDTH, LOSS_WEIGHTS
    SPAN_MAX_WIDTH = int(getattr(args, "span_max_width", SPAN_MAX_WIDTH) or SPAN_MAX_WIDTH)
    for kv in getattr(args, "loss_weight", None) or []:
        k, _, v = kv.partition("=")
        if k and _ and k in LOSS_WEIGHTS:
            LOSS_WEIGHTS[k] = float(v)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    if getattr(args, "dataset", None):
        all_records = load_record_files(args.dataset, split=None)
    else:
        all_records = load_records(Path(args.dataset_dir), split=None)
    train_records = [r for r in all_records if r.get("split") == "train"]
    if not train_records:
        raise RuntimeError("no train-split records found")

    if getattr(args, "tokenizer", None):
        tokenizer = H.load_pater_hf_tokenizer(args.tokenizer)
    else:
        tokenizer = build_tokenizer(all_records)
    config = PATERConfig.from_yaml(args.config)
    for dest, attr in ABLATION_FLAGS:
        if getattr(args, dest, False):
            setattr(config, attr, False)
    config.aux_from_token_state = bool(getattr(args, "aux_from_token_state", False))
    config.generic_register_stream = bool(getattr(args, "generic_register_stream", False))
    if getattr(args, "use_reasoning_heads", False):
        config.use_reasoning_heads = True
    if getattr(args, "fact_pointer_heads", False):
        config.use_fact_pointer_heads = True
    if getattr(args, "evidence_item_heads", False):
        config.use_evidence_item_heads = True
    render_mode = getattr(args, "render_mode", None) or ("input_only" if getattr(args, "input_only", False) else "full")
    if render_mode not in RENDER_MODES:
        raise ValueError(f"render_mode {render_mode!r} not in {RENDER_MODES}")
    dims = dims_from_config(config)
    # Default the truncation cap to the model's context window (so a larger
    # max_position_embeddings, e.g. tiny_external, is actually used).
    max_len = config.max_position_embeddings if args.max_seq_len is None else min(config.max_position_embeddings, args.max_seq_len)
    truncated = sum(1 for r in train_records if len(encode_ids(tokenizer, r, max_len + 10_000, render_mode)) > max_len)

    model = make_model(config, tokenizer, device)

    # Phase 2 warm-start: load the pretrained Qwen3-0.6B backbone into the PAT-ER
    # token stream (zero-init side-state so init == pretrained backbone), then
    # optionally freeze the backbone so only the PAT-ER side-state trains.
    if getattr(args, "resume_from", None):
        # Stage 4+: load a previous checkpoint (backbone already warm-started);
        # re-apply freeze then selectively open injection paths.
        ckpt = torch.load(str(args.resume_from), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        print(f"resume: loaded model state from {args.resume_from}")
    elif getattr(args, "warmstart_qwen", None):
        qpath = WS.resolve_qwen_weights(args.warmstart_qwen)
        qwen_state = WS.load_safetensors_state_dict(Path(qpath))
        ws_report = WS.load_qwen_into_pater(model, qwen_state)
        WS.zero_init_warmstart_side_state(model)
        model.to(device)
        print(f"warmstart: {ws_report['n_loaded_params']} Qwen tensors -> backbone "
              f"({ws_report['pretrained_numel']/1e6:.0f}M), zero-init side-state; "
              f"missing={len(ws_report['missing_src'])} mismatch={len(ws_report['shape_mismatch'])} "
              f"src={qpath}")
    if getattr(args, "freeze_backbone", False):
        fz, tr, ntr = freeze_warmstart_backbone(model)
        print(f"freeze-backbone: frozen={fz/1e6:.1f}M  trainable={tr/1e6:.1f}M ({ntr} params)")

    # Stage 4: selectively open injection entry paths at a separate (lower) LR.
    inject_paths_arg = getattr(args, "inject_paths", None) or ""
    inject_paths = [p.strip() for p in inject_paths_arg.split(",") if p.strip()]
    inject_opened: list[str] = []
    if inject_paths:
        n_opened, inject_opened = open_injection_paths(model, inject_paths)
        print(f"inject-open: {n_opened} params  paths={inject_paths}")

    # Stage 5A: open FFN adapter (up+down) weights in the top N layers.
    adapter_opened: list[str] = []
    n_upper = int(getattr(args, "upper_adapter_layers", 0) or 0)
    if n_upper > 0:
        n_a, adapter_opened = open_upper_layer_adapters(model, n_upper)
        upper_start = max(0, model.config.num_hidden_layers - n_upper)
        print(f"upper-adapters: opened {n_a} params in layers {upper_start}–{model.config.num_hidden_layers - 1}")

    # Stage 5B: unfreeze top N Qwen backbone layers (self-attn, base FFN, norms).
    backbone_opened: list[str] = []
    n_bb = int(getattr(args, "backbone_layers", 0) or 0)
    if n_bb > 0:
        n_bb_opened, backbone_opened = open_top_backbone_layers(model, n_bb)
        bb_start = max(0, model.config.num_hidden_layers - n_bb)
        print(f"backbone-open: {n_bb_opened} params in layers {bb_start}–{model.config.num_hidden_layers - 1}")

    inject_opened_set = set(inject_opened) | set(adapter_opened) | set(backbone_opened)

    # Stage 4/5 opened-only mode: freeze ALL params except the opened set so the
    # Stage-3 registers/cross-attn/aux-heads and Stage-4 fuse tensors stay fixed.
    # Without this, main params overfit PAT-ER data and corrupt general-text LM.
    if getattr(args, "injection_only", False) and inject_opened_set:
        frozen_extra = 0
        for name, p in model.named_parameters():
            if name not in inject_opened_set and p.requires_grad:
                p.requires_grad_(False)
                frozen_extra += 1
        print(f"opened-only: froze {frozen_extra} additional params; "
              f"only {len(inject_opened_set)} opened params trainable")

    total_params, trainable = count_parameters(model)
    weight_decay = float(getattr(args, "weight_decay", 0.0) or 0.0)
    inject_lr    = float(getattr(args, "inject_lr",    0.0) or 0.0)
    adapter_lr   = float(getattr(args, "adapter_lr",   0.0) or 0.0) or inject_lr
    backbone_lr  = float(getattr(args, "backbone_lr",  0.0) or 0.0)
    inj_set      = set(inject_opened)
    adapt_set    = set(adapter_opened)
    bb_set       = set(backbone_opened)
    has_sep_lr   = ((inject_lr  > 0 and inject_lr  != args.lr)
                    or (adapter_lr > 0 and adapter_lr != args.lr)
                    or (backbone_lr > 0 and backbone_lr != args.lr))
    if has_sep_lr and inject_opened_set:
        # Four groups: main / injection fuse / upper adapters / top backbone layers.
        combined = inj_set | adapt_set | bb_set
        main_params  = [p for n, p in model.named_parameters() if p.requires_grad and n not in combined]
        inj_params   = [p for n, p in model.named_parameters() if p.requires_grad and n in inj_set]
        adapt_params = [p for n, p in model.named_parameters() if p.requires_grad and n in adapt_set]
        bb_params    = [p for n, p in model.named_parameters() if p.requires_grad and n in bb_set]
        param_groups: list[dict] = [{"params": main_params, "lr": args.lr, "lr_scale": 1.0}]
        if inj_params:
            param_groups.append({"params": inj_params, "lr": inject_lr,
                                  "lr_scale": inject_lr / max(args.lr, 1e-12)})
        if adapt_params:
            param_groups.append({"params": adapt_params, "lr": adapter_lr,
                                  "lr_scale": adapter_lr / max(args.lr, 1e-12)})
        if bb_params:
            _bb_lr = backbone_lr if backbone_lr > 0 else args.lr
            param_groups.append({"params": bb_params, "lr": _bb_lr,
                                  "lr_scale": _bb_lr / max(args.lr, 1e-12)})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        print(f"lr groups: main={args.lr:.1e}"
              + (f"  inject={inject_lr:.1e}" if inj_params else "")
              + (f"  adapter={adapter_lr:.1e}" if adapt_params else "")
              + (f"  backbone={_bb_lr:.1e}" if bb_params else ""))
    else:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=weight_decay
        )

    # Linear warmup then cosine decay to 10% of peak. Larger configs (e.g. 450M)
    # diverge from a random init under a fixed LR; warmup is what keeps the deep
    # register/cross-attention stack stable in the first steps. warmup_steps=0
    # reproduces the old fixed-LR behaviour for the tiny configs.
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
    min_lr_frac = 0.1

    def lr_at(step: int) -> float:
        # Schedule is opt-in: warmup_steps == 0 keeps the fixed LR the tiny
        # configs were tuned with. warmup_steps > 0 enables linear warmup then
        # cosine decay to min_lr_frac of peak (needed to stabilise larger configs).
        if warmup_steps <= 0:
            return args.lr
        if step <= warmup_steps:
            return args.lr * step / warmup_steps
        if args.steps <= warmup_steps:
            return args.lr
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)

    ablation = {attr: getattr(config, attr) for _, attr in ABLATION_FLAGS}
    ablation["aux_from_token_state"] = config.aux_from_token_state
    ablation["generic_register_stream"] = config.generic_register_stream
    print("PAT-ER tiny supervised overfit (train)")
    print(f"tag={getattr(args, 'tag', '') or 'full'} render_mode={render_mode} ablation={ablation}")
    print(f"device={device} dtype={args.dtype} steps={args.steps} batch_size={args.batch_size} lr={args.lr} "
          f"warmup={warmup_steps} weight_decay={weight_decay} seed={args.seed}")
    tokenizer_desc = f"hf:{args.tokenizer}" if getattr(args, "tokenizer", None) else "reference"
    print(f"records: total={len(all_records)} train={len(train_records)} tokenizer={tokenizer_desc} "
          f"vocab={tokenizer.vocab_size} max_len={max_len} truncated_train={truncated}")
    spans = span_label_stats(tokenizer, train_records, max_len, dims, render_mode)
    print(f"span-label migration (train): "
          f"arg {spans['located']['arg']}/{spans['source']['arg']} (rate={spans['located_rate']['arg']}), "
          f"predicate {spans['located']['predicate']}/{spans['source']['predicate']} (rate={spans['located_rate']['predicate']}), "
          f"evidence {spans['located']['evidence']}/{spans['source']['evidence']} (rate={spans['located_rate']['evidence']})  "
          f"out_of_bounds={spans['out_of_bounds']} point_to_pad={spans['point_to_pad']}")
    # Inverse-frequency class weights for the primitive bridge heads, from the
    # train-split primitive_class distribution.
    balanced = bool(getattr(args, "balanced_sampler", False))
    class_weights: dict[str, torch.Tensor] = {}
    if balanced:
        counts = [0] * len(builder.PRIMITIVE_CLASSES)
        for r in train_records:
            idx = _index_in(builder.PRIMITIVE_CLASSES, r["labels"].get("primitive_class"))
            if idx is not None:
                counts[idx] += 1
        total_c = sum(counts) or 1
        weights = torch.tensor([(total_c / (len(counts) * c)) if c else 0.0 for c in counts], dtype=torch.float)
        for head in CLASS_WEIGHTED_HEADS:
            class_weights[head] = weights
        print(f"balanced_sampler=True class_weights(primitive)={[round(float(w), 2) for w in weights]}")

    # Optional class-balanced sampler over primitive_class.
    by_class: dict[int, list[int]] = {}
    for i, r in enumerate(train_records):
        idx = _index_in(builder.PRIMITIVE_CLASSES, r["labels"].get("primitive_class"))
        by_class.setdefault(idx if idx is not None else -1, []).append(i)
    class_keys = sorted(by_class)
    class_cursor = {k: 0 for k in class_keys}
    key_cursor = 0
    order = list(range(len(train_records)))
    cursor = len(order)

    # Span-focused oversampling weights (multi-subword + external spans).
    span_weight: list[float] | None = None
    if getattr(args, "span_sampler", False):
        alpha = float(getattr(args, "span_sampler_alpha", 1.0) or 1.0)
        raw = [record_span_weight(tokenizer, r, max_len, render_mode) for r in train_records]
        span_weight = [w ** alpha for w in raw]
        ext = sum(1 for r in train_records if (r.get("provenance") or {}).get("mix_origin") == "external")
        multi = sum(1 for w in raw if w > 1.0)
        print(f"span_sampler=True alpha={alpha} external={ext}/{len(train_records)} "
              f"multi_subword_or_ext={multi}/{len(train_records)} mean_weight={sum(span_weight) / len(span_weight):.2f}")

    print(f"params: total={total_params:,} trainable={trainable:,}")
    print(f"loss weights: {LOSS_WEIGHTS}")
    print(f"span_max_width={SPAN_MAX_WIDTH}")

    def next_batch() -> Batch:
        nonlocal cursor, key_cursor
        picks: list[int] = []
        if balanced:
            # Class-balanced backbone protects the bridge guardrail; within each
            # primitive-class bucket, span-focused weighting (when on) oversamples
            # the hard multi-subword / external spans.
            while len(picks) < args.batch_size:
                k = class_keys[key_cursor % len(class_keys)]
                key_cursor += 1
                bucket = by_class[k]
                if span_weight is not None:
                    w = [span_weight[i] for i in bucket]
                    picks.append(rng.choices(bucket, weights=w, k=1)[0])
                    continue
                if class_cursor[k] >= len(bucket):
                    rng.shuffle(bucket)
                    class_cursor[k] = 0
                picks.append(bucket[class_cursor[k]])
                class_cursor[k] += 1
        elif span_weight is not None:
            while len(picks) < args.batch_size:
                picks.append(rng.choices(order, weights=span_weight, k=1)[0])
        else:
            while len(picks) < args.batch_size:
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                picks.append(order[cursor])
                cursor += 1
        return collate(tokenizer, [train_records[i] for i in picks], max_len, dims, render_mode)

    # Stage 5B: KL guard and backbone weight-delta tracking.
    kl_guard_thr   = float(getattr(args, "kl_guard_threshold",  0.0) or 0.0)
    lm_guard_thr   = float(getattr(args, "lm_guard_threshold",  0.0) or 0.0)
    kl_eval_itvl   = int(getattr(args,   "kl_eval_interval",   50)   or 50)
    # Store initial backbone-layer weights to compute delta norms per log.
    _bb_init: dict[str, torch.Tensor] = {}
    if bb_set:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in bb_set:
                    _bb_init[name] = p.data.clone()
    # Fixed probe sentences for KL guard evaluation.
    _KL_PROBES = [
        "The capital of France is Paris, a city on the river Seine.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "If all men are mortal and Socrates is a man, then Socrates is mortal.",
    ]

    @torch.no_grad()
    def _eval_kl_guard() -> tuple[float, float]:
        """LM delta and KL(full||clean) on fixed probe sentences."""
        from transformers import AutoTokenizer as _ATok
        _tok_dir = getattr(args, "tokenizer", None) or checkpoint.get("tokenizer_path", "")
        _tok = _ATok.from_pretrained(str(_tok_dir), local_files_only=True, use_fast=True) if _tok_dir else None
        if _tok is None:
            return 0.0, 0.0
        enc = _tok(_KL_PROBES, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        lbl = ids.clone(); lbl[attn == 0] = -100
        model.eval()
        model.config.use_event_stream = False
        model.config.use_primitive_stream = False
        model.config.use_vocab_pressure = False
        for _layer in model.layers:
            _layer.ffn.enabled = False
        clean_loss = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False).loss.item()
        clean_logits = model(input_ids=ids, attention_mask=attn, return_aux=False).logits.detach().clone()
        model.config.use_event_stream = True
        model.config.use_primitive_stream = True
        model.config.use_vocab_pressure = True
        for _layer in model.layers:
            _layer.ffn.enabled = True
        full_loss = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False).loss.item()
        full_logits = model(input_ids=ids, attention_mask=attn, return_aux=False).logits.detach()
        import torch.nn.functional as _F
        mask = (attn == 1)
        kl = _F.kl_div(
            _F.log_softmax(full_logits[mask], dim=-1),
            _F.softmax(clean_logits[mask], dim=-1),
            reduction="batchmean", log_target=False
        ).item() if mask.any() else 0.0
        model.train()
        return (full_loss - clean_loss, kl)

    model.train()
    first_log: dict[str, float] | None = None
    last_log: dict[str, float] = {}
    history: list[dict[str, float]] = []
    skipped_steps = 0
    kl_guard_stop = False
    grad_clip = float(getattr(args, "grad_clip", 1.0) or 1.0)

    for step in range(1, args.steps + 1):
        if kl_guard_stop:
            break
        cur_lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr * group.get("lr_scale", 1.0)
        batch = next_batch().to(device)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, args.dtype):
            output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                           labels=batch.input_ids, return_aux=True,
                           evidence_item_mask=batch.er.get("evidence_item_mask"))
            losses = compute_losses(output, batch, class_weights)
            loss = total_loss(losses)
        # Non-finite-skip guard: a single pathological batch (a NaN/inf loss or
        # gradient) must not poison the weights of a larger model. Skip the
        # update instead of corrupting the run, and keep the last finite log.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                print(f"  step {step:>4}: lr={cur_lr:.2e} SKIPPED (non-finite loss)")
            continue
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(total_norm):
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                print(f"  step {step:>4}: lr={cur_lr:.2e} SKIPPED (non-finite grad-norm)")
            continue
        optimizer.step()

        record = {"step": step, "total": float(loss.detach()), **{k: float(v.detach()) for k, v in losses.items()}}
        history.append(record)
        last_log = record
        if first_log is None:
            first_log = record
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            comps = f"lr={cur_lr:.2e} " + " ".join(f"{k}={record[k]:.3f}" for k in record if k != "step")
            if inj_set:
                inj_w_norm = sum(p.data.norm().item() for n, p in model.named_parameters() if n in inj_set)
                comps += f" inj_wnorm={inj_w_norm:.4e}"
            if adapt_set:
                adp_up = sum(p.data.norm().item() for n, p in model.named_parameters()
                             if n in adapt_set and n.endswith(".up.weight"))
                adp_dn = sum(p.data.norm().item() for n, p in model.named_parameters()
                             if n in adapt_set and n.endswith(".down.weight"))
                comps += f" adp_up={adp_up:.4e} adp_dn={adp_dn:.4e}"
            if _bb_init:
                bb_delta = sum(
                    (p.data - _bb_init[n]).norm().item()
                    for n, p in model.named_parameters() if n in _bb_init
                )
                comps += f" bb_delta={bb_delta:.4e}"
            print(f"  step {step:>4}: {comps}")
            # KL guard: evaluate on fixed probes and stop if thresholds exceeded.
            if (kl_guard_thr > 0 or lm_guard_thr > 0) and step % kl_eval_itvl == 0:
                _probe_delta, _probe_kl = _eval_kl_guard()
                print(f"  [kl-guard] step={step} lm_delta={_probe_delta:+.4f} KL={_probe_kl:.4f}", flush=True)
                if (kl_guard_thr > 0 and _probe_kl > kl_guard_thr) or \
                   (lm_guard_thr > 0 and _probe_delta > lm_guard_thr):
                    print(f"  KL guard triggered — stopping early at step {step}")
                    kl_guard_stop = True

    if skipped_steps:
        print(f"skipped {skipped_steps}/{args.steps} steps (non-finite loss/grad)")

    model.eval()
    acc = MetricAccumulator.empty()
    with torch.no_grad():
        for i in range(0, len(train_records), args.batch_size):
            sub = train_records[i:i + args.batch_size]
            batch = collate(tokenizer, sub, max_len, dims, render_mode).to(device)
            with maybe_autocast(device, args.dtype):
                output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                               labels=batch.input_ids, return_aux=True,
                           evidence_item_mask=batch.er.get("evidence_item_mask"))
            acc.update(output, batch)
    print("train-subset snapshot (after training):")
    _print_snapshot(acc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": config.to_dict(),
        "vocab": tokenizer.token_to_id if not getattr(args, "tokenizer", None) else None,
        "tokenizer_path": str(args.tokenizer) if getattr(args, "tokenizer", None) else None,
        "model_state": model.state_dict(),
        "loss_weights": LOSS_WEIGHTS,
        "max_len": max_len,
        "render_mode": render_mode,
        "dataset": [str(p) for p in args.dataset] if getattr(args, "dataset", None) else None,
        "dataset_dir": str(args.dataset_dir),
        "step": args.steps,
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "first_loss": first_log,
        "last_loss": last_log,
    }
    latest = out_dir / "latest.pt"
    torch.save(checkpoint, latest)
    torch.save(checkpoint, out_dir / f"step_{args.steps}.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if first_log is not None:
        print(f"first total loss: {first_log['total']:.4f}  last total loss: {last_log['total']:.4f}")
    print(f"checkpoint: {_rel(latest)}")
    return latest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny supervised PAT-ER overfit (LM + aux heads).")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset", type=Path, nargs="+", default=None,
                        help="Explicit JSONL file(s) to train on (e.g. a glob); overrides --dataset-dir.")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Path to a saved PAT-ER-extended HF tokenizer dir (subword vocab + offset-based span "
                             "location). Default: build the word-level reference tokenizer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear LR warmup steps, then cosine decay to 10%% of peak. 0 keeps a fixed LR.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global grad-norm clip; steps with non-finite loss/grad are skipped.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Truncation cap; default is the config's max_position_embeddings.")
    parser.add_argument("--tag", type=str, default="", help="Label for this run (e.g. ablation variant name).")
    # Ablation overrides (flip off an architectural component).
    parser.add_argument("--no-event-stream", action="store_true", help="Ablate the event-role stream (use_event_stream=False).")
    parser.add_argument("--no-primitive-stream", action="store_true", help="Ablate the primitive stream (use_primitive_stream=False).")
    parser.add_argument("--no-ffn-adapters", action="store_true", help="Ablate role/primitive FFN adapters (use_role_primitive_ffn=False).")
    parser.add_argument("--no-pressure", action="store_true", help="Ablate role/primitive pressure at the LM logits (use_vocab_pressure=False).")
    parser.add_argument("--input-only", action="store_true", help="Render input without the serialized event_graph/output (lower leakage).")
    parser.add_argument("--render-mode", choices=list(RENDER_MODES), default=None,
                        help="Input rendering: full | input_only | plain | prefix (overrides --input-only).")
    parser.add_argument("--aux-from-token-state", action="store_true",
                        help="Baseline: aux heads read pooled token state instead of side-state registers.")
    parser.add_argument("--generic-register-stream", action="store_true",
                        help="Control: same learned register count/modules, but a homogeneous register bank with no typed event-role->primitive flow.")
    parser.add_argument("--use-reasoning-heads", action="store_true",
                        help="Enable reasoning-supervision heads (entailment_state/proof_depth/rule_chain_length) "
                             "from ProofWriter proof metadata (aux targets only, no CoT in input).")
    parser.add_argument("--fact-pointer-heads", action="store_true",
                        help="(Tested, not kept) add supporting/contradicting fact-pointer heads + BCE losses.")
    parser.add_argument("--evidence-item-heads", action="store_true",
                        help="Per-evidence-item support/refute heads (pool each item span, score per item).")
    parser.add_argument("--span-max-width", type=int, default=SPAN_MAX_WIDTH,
                        help="Max argument-span width (subword tokens) for the joint span objective + decode.")
    parser.add_argument("--span-sampler", action="store_true",
                        help="Oversample records with multi-subword argument spans and external (FOLIO/ProofWriter) origin.")
    parser.add_argument("--span-sampler-alpha", type=float, default=1.0,
                        help="Strength of span-focused oversampling (0 = uniform, higher = more skew to hard spans).")
    parser.add_argument("--warmstart-qwen", type=str, default=None,
                        help="Warm-start the token backbone from Qwen3-0.6B weights. Path to "
                             "model.safetensors, or 'auto' to resolve from the offline HF cache. "
                             "Use with configs/pat_er_qwen3_warmstart.yaml + the extended tokenizer.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze the pretrained Qwen backbone (embeddings/self-attn/base FFN/"
                             "norms); train only the PAT-ER side-state (Phase 2 Stage 3).")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Class-balanced batches over primitive_class + inverse-frequency class weights for the bridge heads.")
    parser.add_argument("--loss-weight", nargs="*", metavar="KEY=VALUE", default=None,
                        help="Override individual LOSS_WEIGHTS entries, e.g. "
                             "--loss-weight event_arg=0 argument_span=0.01. "
                             "Useful for stage-gated training (e.g. Stage 3: zero span/pointer heads).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Stage 4+: load model state from a prior checkpoint (.pt) instead of "
                             "running warm-start. Still applies --freeze-backbone and --inject-paths.")
    parser.add_argument("--inject-paths", type=str, default=None,
                        help="Comma-separated injection entry paths to open after freeze-backbone. "
                             "Values: role_fuse, primitive_fuse, adapter_up, pressure_proj1. "
                             "E.g. --inject-paths role_fuse,primitive_fuse (Stage 4A).")
    parser.add_argument("--inject-lr", type=float, default=0.0,
                        help="Separate learning rate for opened injection paths (Stage 4). "
                             "Lower than main --lr (e.g. 1e-6 vs 8e-5) to prevent residual "
                             "stream corruption. 0 = use main lr for injection paths too.")
    parser.add_argument("--injection-only", action="store_true",
                        help="Stage 4/5: after opening injection/adapter paths, freeze ALL other "
                             "params (registers, cross-attn, aux heads, gate). Only opened params "
                             "train. Prevents Stage-3 params from drifting and corrupting LM.")
    parser.add_argument("--backbone-layers", type=int, default=0,
                        help="Stage 5B: unfreeze top N Qwen decoder layers (self-attn, FFN, norms). "
                             "Use with --backbone-lr and --kl-guard-threshold.")
    parser.add_argument("--backbone-lr", type=float, default=0.0,
                        help="Separate LR for opened backbone layers (Stage 5B). "
                             "Should be 1e-7 to 5e-7 to prevent residual stream corruption.")
    parser.add_argument("--kl-guard-threshold", type=float, default=0.0,
                        help="Stage 5B: stop training if KL(full||clean) on fixed probes exceeds "
                             "this value. Evaluated every --kl-eval-interval steps. 0 = disabled.")
    parser.add_argument("--lm-guard-threshold", type=float, default=0.0,
                        help="Stage 5B: stop training if LM_delta (full-clean) on fixed probes "
                             "exceeds this value. 0 = disabled.")
    parser.add_argument("--kl-eval-interval", type=int, default=50,
                        help="How often (in steps) to evaluate the KL guard probes.")
    parser.add_argument("--upper-adapter-layers", type=int, default=0,
                        help="Stage 5A: number of layers from the top in which to open FFN "
                             "adapter up+down weights. E.g. 7 opens layers 21-27 in a 28-layer "
                             "model. Zero-init on up.weight provides natural gradient isolation "
                             "(no detach needed). Use with --injection-only + --adapter-lr.")
    parser.add_argument("--adapter-lr", type=float, default=0.0,
                        help="Separate learning rate for opened upper-layer adapter params "
                             "(Stage 5A). Defaults to --inject-lr if set, else --lr.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        train(args)
    except Exception as exc:  # noqa: BLE001
        if args.dtype == "bf16":
            print(f"bf16 path failed: {exc}; rerunning in fp32")
            args.dtype = "fp32"
            train(args)
        else:
            raise


if __name__ == "__main__":
    main()
