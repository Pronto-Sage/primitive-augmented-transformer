#!/usr/bin/env python3
"""Build the first CPU-only PAT-ER synthetic dataset.

This generates the minimum first end-to-end dataset across eight primitive
families. It supervises the token stream, event-role stream, primitive stream,
evidence links, tool interface, and abstention behavior — not free text with a
few labels.

Families (one JSONL file each):

    pat_syn_modus_ponens
    pat_syn_abduction_trap
    pat_syn_contradiction
    pat_syn_tautology_no_progress
    pat_syn_contingency
    pat_syn_event_roles
    pat_syn_tool_grounding
    pat_syn_idk_abstention

Canonical record schema is the one in NOTES[dataset].md section 1 (the requested
``docs/datasets_aug.md`` does not exist in this repo; NOTES[dataset].md is the
authoritative schema source). Label vocabularies come from NOTES[dataset].md
section 5. Model-facing text is rendered with ``pat_er.serialization`` only.

Split policy (NOTES[dataset].md section 6): family/template-level holdout. Each
structural ``template_id`` is assigned to exactly one split, so renamed-variable
variants never straddle train/val/test. Record counts are allocated ~80/10/10
per family, stratified across positive and hard-negative templates so hard
negatives appear in both training and evaluation.

Non-canonical helper fields (documented):

    model_text          rendered prompt (render_pater_prompt over the fields)
    template_id         structural template identity used for holdout
    is_hard_negative    whether the record is a mandated hard negative
    hard_negative_type  which hard-negative class, if any
    rationale           short human note (NOT a hidden chain-of-thought target)

CPU-only, deterministic (seeded), no training, no downloads. Outputs go under
artifacts/datasets/pater_synthetic/ and are git-ignored.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er.sample_data import tokenize_with_pater_spec  # noqa: E402
from pat_er.serialization import (  # noqa: E402
    render_hermes_tool_call,
    render_pater_prompt,
)
from pat_er.tokenizer_spec import (  # noqa: E402
    EVENT_ROLE_SPECIAL_TOKENS,
    PRIMITIVE_SPECIAL_TOKENS,
    SUPPORT_CONTROL_SPECIAL_TOKENS,
    build_pater_tokenizer_spec,
)

DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "datasets" / "pater_synthetic"

# ---------------------------------------------------------------------------
# Canonical label vocabularies (NOTES[dataset].md section 5). Single source of
# truth shared with the validator and summarizer.
# ---------------------------------------------------------------------------
PRIMITIVE_CLASSES = (
    "axiom",
    "observation",
    "contingency",
    "contradiction",
    "tautology",
    "modus_ponens",
    "syllogism",
    "abduction",
    "semantic_conflict",
    "uncertainty",
    "provenance",
    "tool",
    "schema",
    "idk",
)
SUPPORT_STATUSES = ("proof", "belief", "hypothesis", "unknown", "conflict", "no_progress")
TOOL_INTENTS = ("none", "read", "verify", "write", "irreversible")
IDK_ACTIONS = ("answer", "needs_evidence", "needs_verification", "ask_clarification")
SPLITS = ("train", "val", "test")

FAMILIES = (
    "pat_syn_modus_ponens",
    "pat_syn_abduction_trap",
    "pat_syn_contradiction",
    "pat_syn_tautology_no_progress",
    "pat_syn_contingency",
    "pat_syn_event_roles",
    "pat_syn_tool_grounding",
    "pat_syn_idk_abstention",
    "pat_syn_evidence_grounding",
)

# Spec token sets used for self-validation (every role/primitive/support token
# we emit must be an atomic special token in the tokenizer spec).
ROLE_TOKENS = set(EVENT_ROLE_SPECIAL_TOKENS)
PRIMITIVE_TOKENS = set(PRIMITIVE_SPECIAL_TOKENS)
SUPPORT_TOKENS = set(SUPPORT_CONTROL_SPECIAL_TOKENS)

# Event-role / bridge label vocabularies (docs/datasets_aug.md section 4).
SPEC = build_pater_tokenizer_spec(include_schema_key_candidates=True)
# The proto_role head emits num_proto_role_properties=17 outputs. Those are the
# <role:*> property tokens MINUS the three ambiguity-class tokens, which instead
# drive the 4-way role_ambiguity head.
_AMBIGUITY_ROLE_TOKENS = {"<role:ambiguity>", "<role:symmetric>", "<role:role_reversal>"}
PROTO_ROLE_PROPERTIES = tuple(
    t for t in EVENT_ROLE_SPECIAL_TOKENS if t.startswith("<role:") and t not in _AMBIGUITY_ROLE_TOKENS
)
assert len(PROTO_ROLE_PROPERTIES) == 17, f"expected 17 proto-role properties, got {len(PROTO_ROLE_PROPERTIES)}"
ROLE_AMBIGUITY_CLASSES = ("none", "with_pp", "role_reversal", "symmetric")
EVENT_TYPES = ("observation", "rule_application", "conflict", "abstention")
_EVENT_TYPE_BY_PRIMITIVE = {
    "modus_ponens": "rule_application",
    "abduction": "rule_application",
    "syllogism": "rule_application",
    "tool": "rule_application",
    "axiom": "rule_application",
    "contradiction": "conflict",
    "semantic_conflict": "conflict",
    "tautology": "conflict",
    "contingency": "abstention",
    "idk": "abstention",
    "uncertainty": "abstention",
    "observation": "observation",
    "provenance": "observation",
    "schema": "observation",
}

SUBJECTS = ("alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi")


def find_token_span(content_tokens: Sequence[str], span_string: str) -> list[int] | None:
    """Inclusive [start, end] of span_string's first occurrence in content_tokens.

    Indices are into the *content* tokenization (tokenize_with_pater_spec) of the
    model_text, i.e. WITHOUT the bos that encode() prepends. The trainer applies
    the +1 bos offset. Returns None if the span is not present (label masked).
    """

    span_tokens = tokenize_with_pater_spec(span_string, SPEC)
    n = len(span_tokens)
    if n == 0:
        return None
    for i in range(len(content_tokens) - n + 1):
        if list(content_tokens[i : i + n]) == span_tokens:
            return [i, i + n - 1]
    return None


class TemplateSpec(NamedTuple):
    template_id: str
    kind: str  # "positive" | "negative"
    build: Callable[[random.Random], dict[str, Any]]


# ---------------------------------------------------------------------------
# Low-level helpers.
# ---------------------------------------------------------------------------
def _evidence_id(rng: random.Random) -> str:
    return f"e_{rng.randint(100, 999)}"


def _run_id(rng: random.Random) -> str:
    return f"r_{rng.randint(1000, 9999)}"


def _arg(slot: str, span: str, roles: Sequence[str]) -> dict[str, Any]:
    return {"slot": slot, "span": span, "proto_roles": list(roles)}


def _model_facing_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Drop training-only token-index keys so they never enter the model text."""

    clean = {k: v for k, v in event.items() if k != "predicate_span"}
    if "arguments" in clean:
        clean["arguments"] = [
            {k: v for k, v in arg.items() if k != "token_span"} for arg in clean["arguments"]
        ]
    return clean


def render_model_text(record: Mapping[str, Any]) -> str:
    """Render the model-facing prompt from canonical fields (the only renderer).

    predicate_span / token_span are training targets, not model input, so they
    are stripped before rendering. Everything else (including evidence.verified)
    is part of the model-facing serialization.
    """

    events = [_model_facing_event(e) for e in record["event_graph"]] or None
    return render_pater_prompt(
        text=record["text"],
        evidence=record["evidence"] or None,
        formula=record["formula"] or None,
        events=events,
        tools=record["tools"] or None,
        output_prelude=record["target"],
    )


def assemble(
    *,
    family: str,
    template_id: str,
    primitive_class: str,
    support_status: str,
    tool_intent: str,
    idk_action: str,
    verifier_accept: bool,
    schema_validity: bool,
    text: str,
    target: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    formula: str | None = None,
    event_graph: Sequence[Mapping[str, Any]] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    evidence_ids: Sequence[str] | None = None,
    is_hard_negative: bool = False,
    hard_negative_type: str | None = None,
    rationale: str = "",
    event_type: str | None = None,
    role_ambiguity: str = "none",
    bridge_binding: str | None = None,
    source: str = "synthetic_primitive",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Deep-copy so per-record span keys never leak into shared template dicts,
    # and add the model-facing evidence.verified flag BEFORE rendering.
    evidence = [dict(item) for item in (evidence or [])]
    for item in evidence:
        item.setdefault("verified", False)
    event_graph = copy.deepcopy(list(event_graph or []))
    tools = list(tools or [])
    if evidence_ids is None:
        evidence_ids = [item["id"] for item in evidence]
    if event_type is None:
        event_type = _EVENT_TYPE_BY_PRIMITIVE.get(primitive_class, "observation")

    record: dict[str, Any] = {
        # canonical fields (docs/datasets_aug.md section 3)
        "id": None,  # assigned after generation
        "split": None,  # assigned per template
        "split_group": template_id,
        "source": source,
        "task_family": family,
        "text": text,
        "evidence": evidence,
        "formula": formula,
        "event_graph": event_graph,
        "tools": tools,
        "target": target,
        "labels": {
            "primitive_class": primitive_class,
            "support_status": support_status,
            "event_type": event_type,
            "role_ambiguity": role_ambiguity,
            "tool_intent": tool_intent,
            "schema_validity": bool(schema_validity),
            "verifier_accept": bool(verifier_accept),
            "idk_action": idk_action,
            "evidence_ids": list(evidence_ids),
            "is_hard_negative": bool(is_hard_negative),
            # Which event/argument binding licenses the primitive label (bridge
            # supervision aligned to event-role state, not copied output tokens).
            "bridge_binding": bridge_binding,
        },
        # documented helper fields
        "template_id": template_id,
        "is_hard_negative": bool(is_hard_negative),
        "hard_negative_type": hard_negative_type,
        "rationale": rationale,
    }
    if provenance is not None:
        record["provenance"] = dict(provenance)
    # Render first (spans not yet added), then compute token-index labels against
    # the content tokenization and attach them to the stored event graph.
    record["model_text"] = render_model_text(record)
    content_tokens = tokenize_with_pater_spec(record["model_text"], SPEC)
    for event in record["event_graph"]:
        predicate = event.get("predicate")
        event["predicate_span"] = (
            find_token_span(content_tokens, predicate) if isinstance(predicate, str) else None
        )
        for arg in event.get("arguments", []):
            span = arg.get("span")
            arg["token_span"] = find_token_span(content_tokens, span) if isinstance(span, str) else None
    return record


# ---------------------------------------------------------------------------
# Family: modus ponens (P, P IMPLIES Q, P observed -> Q by modus ponens).
# ---------------------------------------------------------------------------
IMPLICATION_TEMPLATES = (
    ("release", "tests_passed", "release_ok", "verify_release", "ci_log"),
    ("build", "compile_ok", "artifact_ready", "build_artifact", "ci_log"),
    ("access", "auth_ok", "access_granted", "grant_access", "audit_log"),
    ("payment", "funds_available", "charge_ok", "settle_payment", "metrics_db"),
    ("ingest", "schema_valid", "ingest_ok", "ingest_records", "tool_output"),
    ("incident", "alert_fired", "page_oncall", "page_engineer", "pager_alert"),
)


def _mp_record(rng, *, template_id, premise, conclusion, predicate, source) -> dict[str, Any]:
    ev_id = _evidence_id(rng)
    return assemble(
        family="pat_syn_modus_ponens",
        template_id=template_id,
        primitive_class="modus_ponens",
        support_status="proof",
        tool_intent="none",
        idk_action="answer",
        verifier_accept=True,
        schema_validity=True,
        text=f"rule if {premise} implies {conclusion} evidence {ev_id} {premise}",
        target="<support:proof>",
        evidence=[{"id": ev_id, "source": source, "text": premise, "reliability": round(rng.uniform(0.9, 0.99), 2)}],
        formula=f"<atom> {premise} </atom> IMPLIES <atom> {conclusion} </atom>",
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": predicate,
                "arguments": [
                    _arg("<arg:0>", premise, ["<role:source>", "<role:causes_change>"]),
                    _arg("<arg:1>", conclusion, ["<role:goal>", "<role:undergoes_change>"]),
                ],
                "primitive": "<prim:modus_ponens>",
                "support": "<support:proof>",
            }
        ],
        rationale="premise observed and rule fires: Q follows by modus ponens",
    )


def templates_modus_ponens() -> list[TemplateSpec]:
    specs = []
    for name, premise, conclusion, predicate, source in IMPLICATION_TEMPLATES:
        tid = f"pat_syn_modus_ponens:{name}"
        specs.append(TemplateSpec(tid, "positive", partial(
            _mp_record, template_id=tid, premise=premise, conclusion=conclusion, predicate=predicate, source=source)))
    return specs


# ---------------------------------------------------------------------------
# Family: abduction trap (Q, P IMPLIES Q -> P is hypothesis; the trap labels it
# proof). Mandatory hard negative: abduction_as_proof.
# ---------------------------------------------------------------------------
ABDUCTION_TEMPLATES = (
    ("cleanup", "cleanup_job_ran", "file_deleted", "abduce_cleanup", "audit_log"),
    ("deploy", "deploy_done", "service_restarted", "abduce_deploy", "metrics_db"),
    ("prefetch", "prefetch_ran", "cache_warmed", "abduce_prefetch", "tool_output"),
    ("migration", "migration_ran", "rows_changed", "abduce_migration", "ci_log"),
    ("gc", "gc_ran", "memory_freed", "abduce_gc", "metrics_db"),
)


def _abduction_record(rng, *, template_id, cause, observed, predicate, source, trap: bool) -> dict[str, Any]:
    ev_id = _evidence_id(rng)
    evidence = [{"id": ev_id, "source": source, "text": observed, "reliability": round(rng.uniform(0.5, 0.7), 2)}]
    formula = f"<atom> {cause} </atom> IMPLIES <atom> {observed} </atom>"
    if trap:
        return assemble(
            family="pat_syn_abduction_trap",
            template_id=template_id,
            primitive_class="abduction",
            support_status="proof",  # the trap
            tool_intent="none",
            idk_action="answer",
            verifier_accept=False,
            schema_validity=True,
            text=f"rule if {cause} implies {observed} evidence {ev_id} {observed} therefore {cause}",
            target="<support:proof>",
            evidence=evidence,
            formula=formula,
            event_graph=[
                {
                    "event": "<evt:0>",
                    "predicate": predicate,
                    "arguments": [
                        _arg("<arg:0>", cause, ["<role:source>", "<role:causes_change>"]),
                        _arg("<arg:1>", observed, ["<role:undergoes_change>"]),
                    ],
                    "primitive": "<prim:abduction>",
                    "support": "<support:proof>",
                }
            ],
            is_hard_negative=True,
            hard_negative_type="abduction_as_proof",
            rationale="affirming the consequent: abduction asserted as proof",
        )
    return assemble(
        family="pat_syn_abduction_trap",
        template_id=template_id,
        primitive_class="abduction",
        support_status="hypothesis",
        tool_intent="none",
        idk_action="needs_verification",
        verifier_accept=False,
        schema_validity=True,
        text=f"rule if {cause} implies {observed} evidence {ev_id} {observed} maybe {cause}",
        target="<support:hypothesis>",
        evidence=evidence,
        formula=formula,
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": predicate,
                "arguments": [
                    _arg("<arg:0>", cause, ["<role:source>", "<role:causes_change>"]),
                    _arg("<arg:1>", observed, ["<role:undergoes_change>", "<role:affected>"]),
                ],
                "primitive": "<prim:abduction>",
                "support": "<support:hypothesis>",
            }
        ],
        rationale="Q observed and P IMPLIES Q: P is a hypothesis, never proof",
    )


def templates_abduction_trap() -> list[TemplateSpec]:
    specs = []
    for name, cause, observed, predicate, source in ABDUCTION_TEMPLATES:
        pos_id = f"pat_syn_abduction_trap:{name}_hypothesis"
        neg_id = f"pat_syn_abduction_trap:{name}_proof"
        specs.append(TemplateSpec(pos_id, "positive", partial(
            _abduction_record, template_id=pos_id, cause=cause, observed=observed, predicate=predicate, source=source, trap=False)))
        specs.append(TemplateSpec(neg_id, "negative", partial(
            _abduction_record, template_id=neg_id, cause=cause, observed=observed, predicate=predicate, source=source, trap=True)))
    return specs


# ---------------------------------------------------------------------------
# Family: contradiction / semantic conflict.
# ---------------------------------------------------------------------------
CONTRADICTION_SERVICES = ("service_a", "service_b", "payments_api", "auth_gateway", "billing_worker")
CONTRADICTION_ATOMS = ("rollback_safe", "deploy_done", "release_ok")
SEMANTIC_PAIRS = (
    ("bachelor", "married", "a bachelor is married"),
    ("empty_set", "has_member", "the empty set has a member"),
    ("frozen", "boiling", "the sample is frozen and boiling"),
)


def _contradiction_evidence(rng, *, template_id) -> dict[str, Any]:
    service = rng.choice(CONTRADICTION_SERVICES)
    e_up, e_down = _evidence_id(rng), _evidence_id(rng)
    return assemble(
        family="pat_syn_contradiction",
        template_id=template_id,
        primitive_class="contradiction",
        support_status="conflict",
        tool_intent="none",
        idk_action="needs_verification",
        verifier_accept=False,
        schema_validity=True,
        text=f"evidence {e_up} {service} up and evidence {e_down} {service} down",
        target="<conflict>",
        evidence=[
            {"id": e_up, "source": "pager_alert", "text": f"{service} up", "reliability": 0.9},
            {"id": e_down, "source": "pager_alert", "text": f"{service} down", "reliability": 0.9},
        ],
        formula=f"<atom> {service}_up </atom> CONTRADICTS <atom> {service}_down </atom>",
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": "contradict",
                "arguments": [
                    _arg("<arg:0>", f"{service} up", ["<role:exists_independently>"]),
                    _arg("<arg:1>", f"{service} down", ["<role:exists_independently>"]),
                ],
                "primitive": "<prim:contradiction>",
                "support": "<support:conflict>",
            }
        ],
        rationale="two evidence items assert P and NOT P",
    )


def _contradiction_not_p(rng, *, template_id) -> dict[str, Any]:
    atom = rng.choice(CONTRADICTION_ATOMS)
    ev = _evidence_id(rng)
    return assemble(
        family="pat_syn_contradiction",
        template_id=template_id,
        primitive_class="contradiction",
        support_status="conflict",
        tool_intent="none",
        idk_action="needs_verification",
        verifier_accept=False,
        schema_validity=True,
        text=f"claim {atom} and claim not {atom} evidence {ev} {atom}",
        target="<conflict>",
        evidence=[{"id": ev, "source": "ci_log", "text": atom, "reliability": 0.95}],
        formula=f"<atom> {atom} </atom> AND NOT <atom> {atom} </atom>",
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": "assert",
                "arguments": [_arg("<arg:0>", atom, ["<role:exists_independently>"])],
                "primitive": "<prim:contradiction>",
                "support": "<support:conflict>",
            }
        ],
        rationale="P and NOT P asserted together",
    )


def _contradiction_semantic(rng, *, template_id, a, b, sentence) -> dict[str, Any]:
    return assemble(
        family="pat_syn_contradiction",
        template_id=template_id,
        primitive_class="semantic_conflict",
        support_status="conflict",
        tool_intent="none",
        idk_action="answer",
        verifier_accept=False,
        schema_validity=True,
        text=sentence,
        target="<conflict>",
        formula=f"<atom> {a} </atom> CONTRADICTS <atom> {b} </atom>",
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": "be",
                "arguments": [_arg("<arg:0>", a, ["<role:exists_independently>"])],
                "primitive": "<prim:semantic_conflict>",
                "support": "<support:conflict>",
            }
        ],
        rationale="lexical/type-level contradiction, not evidence conflict",
    )


def templates_contradiction() -> list[TemplateSpec]:
    specs = [
        TemplateSpec("pat_syn_contradiction:evidence_conflict", "positive",
                     partial(_contradiction_evidence, template_id="pat_syn_contradiction:evidence_conflict")),
        TemplateSpec("pat_syn_contradiction:not_p", "positive",
                     partial(_contradiction_not_p, template_id="pat_syn_contradiction:not_p")),
    ]
    for a, b, sentence in SEMANTIC_PAIRS:
        tid = f"pat_syn_contradiction:semantic_{a}"
        specs.append(TemplateSpec(tid, "positive", partial(
            _contradiction_semantic, template_id=tid, a=a, b=b, sentence=sentence)))
    return specs


# ---------------------------------------------------------------------------
# Family: tautology / no-progress. Hard negative: no_progress sold as proof.
# ---------------------------------------------------------------------------
TAUTOLOGY_ATOMS = ("rollback_safe", "tests_passed", "release_ok", "deploy_done", "cache_warmed")
TAUTOLOGY_SHAPES = {
    "excluded_middle": ("either {a} or not {a}", "<atom> {a} </atom> OR NOT <atom> {a} </atom>", "disjoin", ["<role:exists_independently>"]),
    "self_implication": ("{a} implies {a}", "<atom> {a} </atom> IMPLIES <atom> {a} </atom>", "imply", ["<role:source>", "<role:goal>"]),
    "restate_known": ("{a} is already known and the candidate restates {a}", "<atom> {a} </atom> IFF <atom> {a} </atom>", "restate", ["<role:exists_independently>"]),
}


def _tautology_record(rng, *, template_id, shape, trap: bool) -> dict[str, Any]:
    atom = rng.choice(TAUTOLOGY_ATOMS)
    text_tpl, formula_tpl, predicate, roles = TAUTOLOGY_SHAPES[shape]
    text = text_tpl.format(a=atom)
    formula = formula_tpl.format(a=atom)
    if trap:
        return assemble(
            family="pat_syn_tautology_no_progress",
            template_id=template_id,
            primitive_class="tautology",
            support_status="proof",  # the trap
            tool_intent="none",
            idk_action="answer",
            verifier_accept=False,
            schema_validity=True,
            text=f"{text} therefore the release is proven",
            target="<support:proof>",
            formula=formula,
            event_graph=[
                {
                    "event": "<evt:0>",
                    "predicate": predicate,
                    "arguments": [_arg("<arg:0>", atom, roles)],
                    "primitive": "<prim:tautology>",
                    "support": "<support:proof>",
                }
            ],
            is_hard_negative=True,
            hard_negative_type="no_progress",
            rationale="tautology presented as new proof/progress",
        )
    return assemble(
        family="pat_syn_tautology_no_progress",
        template_id=template_id,
        primitive_class="tautology",
        support_status="no_progress",
        tool_intent="none",
        idk_action="answer",
        verifier_accept=False,
        schema_validity=True,
        text=text,
        target="<no_progress>",
        formula=formula,
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": predicate,
                "arguments": [_arg("<arg:0>", atom, roles)],
                "primitive": "<prim:tautology>",
                "support": "<support:no_progress>",
            }
        ],
        rationale="adds no new information; no progress",
    )


def templates_tautology() -> list[TemplateSpec]:
    specs = []
    for shape in TAUTOLOGY_SHAPES:
        tid = f"pat_syn_tautology_no_progress:{shape}"
        specs.append(TemplateSpec(tid, "positive", partial(_tautology_record, template_id=tid, shape=shape, trap=False)))
    for shape in ("excluded_middle", "self_implication"):
        tid = f"pat_syn_tautology_no_progress:{shape}_trap"
        specs.append(TemplateSpec(tid, "negative", partial(_tautology_record, template_id=tid, shape=shape, trap=True)))
    return specs


# ---------------------------------------------------------------------------
# Family: contingency (missing discriminating evidence -> needs observation).
# ---------------------------------------------------------------------------
CONTINGENCY_SCENARIOS = (
    ("migration", "payments_api", "did the migration finish", "migration_done"),
    ("rollout", "auth_gateway", "is the canary healthy", "canary_healthy"),
    ("backup", "billing_worker", "did the nightly backup run", "backup_done"),
    ("index", "search_indexer", "is the index fully rebuilt", "index_ready"),
)


def _contingency_record(rng, *, template_id, service, question, atom) -> dict[str, Any]:
    has_partial = rng.random() < 0.5
    evidence: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    if has_partial:
        ev = _evidence_id(rng)
        evidence = [{"id": ev, "source": "metrics_db", "text": f"{service} reachable", "reliability": 0.6}]
        evidence_ids = [ev]
    return assemble(
        family="pat_syn_contingency",
        template_id=template_id,
        primitive_class="contingency",
        support_status="unknown",
        tool_intent="none",
        idk_action="needs_evidence",
        verifier_accept=False,
        schema_validity=True,
        text=f"{question} for {service}",
        target="<needs_evidence>",
        evidence=evidence,
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": "determine",
                "arguments": [_arg("<arg:0>", atom, ["<role:undergoes_change>"])],
                "primitive": "<prim:contingency>",
                "support": "<support:unknown>",
            }
        ],
        evidence_ids=evidence_ids,
        rationale="discriminating evidence missing; needs observation",
    )


def templates_contingency() -> list[TemplateSpec]:
    specs = []
    for name, service, question, atom in CONTINGENCY_SCENARIOS:
        tid = f"pat_syn_contingency:{name}"
        specs.append(TemplateSpec(tid, "positive", partial(
            _contingency_record, template_id=tid, service=service, question=question, atom=atom)))
    return specs


# ---------------------------------------------------------------------------
# Family: event roles (role reversal + with-PP ambiguity, positive and hard
# negatives). Mandatory hard negatives: role_reversal, with_pp_ambiguity.
# ---------------------------------------------------------------------------
REVERSAL_VERBS = ("approved", "reviewed", "paid")
PP_CASES = (
    ("cut", "meat", "knife", "<role:instrument>", "instrument"),
    ("burgled", "house", "accomplice", "<role:comitative>", "comitative"),
    ("loaded", "truck", "rocks", "<role:incremental_theme>", "content"),
    ("measured", "cloth", "ruler", "<role:instrument>", "instrument"),
)


def _reversal_record(rng, *, template_id, verb, swapped: bool) -> dict[str, Any]:
    agent = rng.choice(SUBJECTS)
    patient = rng.choice([s for s in SUBJECTS if s != agent])
    if swapped:
        return assemble(
            family="pat_syn_event_roles",
            template_id=template_id,
            primitive_class="semantic_conflict",
            support_status="conflict",
            tool_intent="none",
            idk_action="answer",
            verifier_accept=False,
            schema_validity=True,
            text=f"{agent} {verb} {patient}",
            target="<role:role_reversal>",
            event_graph=[
                {
                    "event": "<evt:0>",
                    "predicate": verb,
                    "arguments": [
                        _arg("<arg:0>", patient, ["<role:volition>", "<role:causes_change>"]),
                        _arg("<arg:1>", agent, ["<role:affected>"]),
                    ],
                    "primitive": "<prim:semantic_conflict>",
                    "support": "<support:conflict>",
                }
            ],
            is_hard_negative=True,
            hard_negative_type="role_reversal",
            role_ambiguity="role_reversal",
            rationale="event graph reverses agent/patient vs the surface text",
        )
    return assemble(
        family="pat_syn_event_roles",
        template_id=template_id,
        primitive_class="observation",
        support_status="belief",
        tool_intent="none",
        idk_action="answer",
        verifier_accept=True,
        schema_validity=True,
        text=f"{agent} {verb} {patient}",
        target="<support:belief>",
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": verb,
                "arguments": [
                    _arg("<arg:0>", agent, ["<role:volition>", "<role:sentience>", "<role:causes_change>"]),
                    _arg("<arg:1>", patient, ["<role:affected>", "<role:undergoes_change>"]),
                ],
                "primitive": "<prim:observation>",
                "support": "<support:belief>",
            }
        ],
        role_ambiguity="symmetric",
        rationale="canonical agent/patient binding",
    )


def _pp_record(rng, *, template_id, verb, obj, pp, role, reading, wrong: bool) -> dict[str, Any]:
    if wrong:
        wrong_role = "<role:comitative>" if role != "<role:comitative>" else "<role:instrument>"
        return assemble(
            family="pat_syn_event_roles",
            template_id=template_id,
            primitive_class="semantic_conflict",
            support_status="conflict",
            tool_intent="none",
            idk_action="answer",
            verifier_accept=False,
            schema_validity=True,
            text=f"john {verb} {obj} with {pp}",
            target="<role:ambiguity>",
            event_graph=[
                {
                    "event": "<evt:0>",
                    "predicate": verb,
                    "arguments": [
                        _arg("<arg:0>", "john", ["<role:volition>"]),
                        _arg("<arg:1>", obj, ["<role:affected>"]),
                        _arg("<arg:adjunct>", pp, [wrong_role]),
                    ],
                    "primitive": "<prim:semantic_conflict>",
                    "support": "<support:conflict>",
                }
            ],
            is_hard_negative=True,
            hard_negative_type="with_pp_ambiguity",
            role_ambiguity="with_pp",
            rationale=f"PP wrongly attached (should be {reading})",
        )
    return assemble(
        family="pat_syn_event_roles",
        template_id=template_id,
        primitive_class="observation",
        support_status="belief",
        tool_intent="none",
        idk_action="answer",
        verifier_accept=True,
        schema_validity=True,
        text=f"john {verb} {obj} with {pp}",
        target="<support:belief>",
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": verb,
                "arguments": [
                    _arg("<arg:0>", "john", ["<role:volition>", "<role:causes_change>"]),
                    _arg("<arg:1>", obj, ["<role:affected>"]),
                    _arg("<arg:adjunct>", pp, [role]),
                ],
                "primitive": "<prim:observation>",
                "support": "<support:belief>",
            }
        ],
        role_ambiguity="with_pp",
        rationale=f"PP correctly attached as {reading}",
    )


def templates_event_roles() -> list[TemplateSpec]:
    specs = []
    for verb in REVERSAL_VERBS:
        tid = f"pat_syn_event_roles:reversal_{verb}"
        specs.append(TemplateSpec(tid, "positive", partial(_reversal_record, template_id=tid, verb=verb, swapped=False)))
    for verb in ("approved", "reviewed"):
        tid = f"pat_syn_event_roles:reversal_{verb}_swapped"
        specs.append(TemplateSpec(tid, "negative", partial(_reversal_record, template_id=tid, verb=verb, swapped=True)))
    for verb, obj, pp, role, reading in PP_CASES:
        tid = f"pat_syn_event_roles:pp_{verb}"
        specs.append(TemplateSpec(tid, "positive", partial(
            _pp_record, template_id=tid, verb=verb, obj=obj, pp=pp, role=role, reading=reading, wrong=False)))
    for verb, obj, pp, role, reading in (PP_CASES[0], PP_CASES[2]):
        tid = f"pat_syn_event_roles:pp_{verb}_wrong"
        specs.append(TemplateSpec(tid, "negative", partial(
            _pp_record, template_id=tid, verb=verb, obj=obj, pp=pp, role=role, reading=reading, wrong=True)))
    return specs


# ---------------------------------------------------------------------------
# Family: tool grounding. Mandatory hard negative: unsupported_tool_action.
# ---------------------------------------------------------------------------
def _tool_schema(name: str, properties: Mapping[str, str], required: Sequence[str], description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": kind} for key, kind in properties.items()},
                "required": list(required),
            },
        },
    }


VERIFY_RELEASE_SCHEMA = _tool_schema("verify_release", {"run_id": "string", "requires": "string"}, ["run_id", "requires"], "Verify a release candidate has passing tests.")
CHECK_EVIDENCE_SCHEMA = _tool_schema("check_evidence", {"evidence_id": "string", "claim_id": "string"}, ["evidence_id", "claim_id"], "Check whether evidence supports a claim.")
QUERY_LOGS_SCHEMA = _tool_schema("query_logs", {"service": "string", "since": "string"}, ["service"], "Read recent logs for a service.")


def _tool_grounded_verify(rng, *, template_id) -> dict[str, Any]:
    run_id = _run_id(rng)
    fact = "tests_passed"
    ev = _evidence_id(rng)
    call = render_hermes_tool_call("verify_release", {"run_id": run_id, "requires": fact})
    return assemble(
        family="pat_syn_tool_grounding",
        template_id=template_id,
        primitive_class="tool",
        support_status="belief",
        tool_intent="verify",
        idk_action="needs_verification",
        verifier_accept=True,
        schema_validity=True,
        text=f"call verify_release for run {run_id} given {fact}",
        target=f"<needs_verification>\n{call}",
        evidence=[
            {"id": ev, "source": "ci_log", "text": fact, "reliability": 0.97},
            {"id": _evidence_id(rng), "source": "tool_output", "text": f"run_id {run_id}", "reliability": 0.99},
        ],
        tools=[VERIFY_RELEASE_SCHEMA],
        evidence_ids=[ev],
        rationale="tool call arguments trace to evidence and match the offered schema",
    )


def _tool_grounded_check(rng, *, template_id) -> dict[str, Any]:
    ev = _evidence_id(rng)
    fact = "build_green"
    call = render_hermes_tool_call("check_evidence", {"evidence_id": ev, "claim_id": "claim_1"})
    return assemble(
        family="pat_syn_tool_grounding",
        template_id=template_id,
        primitive_class="tool",
        support_status="belief",
        tool_intent="verify",
        idk_action="needs_verification",
        verifier_accept=True,
        schema_validity=True,
        text=f"check whether evidence {ev} supports claim_1",
        target=f"<needs_verification>\n{call}",
        evidence=[{"id": ev, "source": "ci_log", "text": fact, "reliability": 0.98}],
        tools=[CHECK_EVIDENCE_SCHEMA],
        evidence_ids=[ev],
        rationale="evidence-check call grounded in a present evidence id",
    )


def _tool_grounded_query(rng, *, template_id) -> dict[str, Any]:
    service = rng.choice(("billing_worker", "auth_gateway", "search_indexer"))
    call = render_hermes_tool_call("query_logs", {"service": service, "since": "1h"})
    return assemble(
        family="pat_syn_tool_grounding",
        template_id=template_id,
        primitive_class="tool",
        support_status="belief",
        tool_intent="read",
        idk_action="needs_verification",
        verifier_accept=True,
        schema_validity=True,
        text=f"pull recent logs for {service}",
        target=f"<needs_verification>\n{call}",
        tools=[QUERY_LOGS_SCHEMA],
        evidence_ids=[],
        rationale="read-only log query matching the offered schema",
    )


def _tool_unsupported_absent(rng, *, template_id) -> dict[str, Any]:
    run_id = _run_id(rng)
    call = render_hermes_tool_call("deploy_release", {"run_id": run_id, "target": "production"})
    return assemble(
        family="pat_syn_tool_grounding",
        template_id=template_id,
        primitive_class="tool",
        support_status="unknown",
        tool_intent="write",
        idk_action="needs_verification",
        verifier_accept=False,
        schema_validity=False,  # deploy_release is not in the offered tools
        text=f"ship run {run_id} to production now",
        target=f"<unsupported>\n{call}",
        evidence=[{"id": _evidence_id(rng), "source": "ci_log", "text": "tests_passed", "reliability": 0.97}],
        tools=[VERIFY_RELEASE_SCHEMA],
        evidence_ids=[],
        is_hard_negative=True,
        hard_negative_type="unsupported_tool_action",
        rationale="syntactically valid call to a tool absent from <tools>",
    )


def _tool_unsupported_ungrounded(rng, *, template_id) -> dict[str, Any]:
    run_id = _run_id(rng)  # never grounded in evidence
    call = render_hermes_tool_call("verify_release", {"run_id": run_id, "requires": "tests_passed"})
    return assemble(
        family="pat_syn_tool_grounding",
        template_id=template_id,
        primitive_class="tool",
        support_status="unknown",
        tool_intent="verify",
        idk_action="needs_evidence",
        verifier_accept=False,
        schema_validity=True,  # correct tool/schema, but argument is ungrounded
        text="verify the release",
        target=f"<needs_evidence>\n{call}",
        evidence=[],
        tools=[VERIFY_RELEASE_SCHEMA],
        evidence_ids=[],
        is_hard_negative=True,
        hard_negative_type="unsupported_tool_action",
        rationale=f"required run_id {run_id} is not grounded in any evidence",
    )


def templates_tool_grounding() -> list[TemplateSpec]:
    return [
        TemplateSpec("pat_syn_tool_grounding:grounded_verify_release", "positive",
                     partial(_tool_grounded_verify, template_id="pat_syn_tool_grounding:grounded_verify_release")),
        TemplateSpec("pat_syn_tool_grounding:grounded_check_evidence", "positive",
                     partial(_tool_grounded_check, template_id="pat_syn_tool_grounding:grounded_check_evidence")),
        TemplateSpec("pat_syn_tool_grounding:grounded_query_logs", "positive",
                     partial(_tool_grounded_query, template_id="pat_syn_tool_grounding:grounded_query_logs")),
        TemplateSpec("pat_syn_tool_grounding:unsupported_absent_tool", "negative",
                     partial(_tool_unsupported_absent, template_id="pat_syn_tool_grounding:unsupported_absent_tool")),
        TemplateSpec("pat_syn_tool_grounding:unsupported_ungrounded_arg", "negative",
                     partial(_tool_unsupported_ungrounded, template_id="pat_syn_tool_grounding:unsupported_ungrounded_arg")),
    ]


# ---------------------------------------------------------------------------
# Family: IDK / abstention (abstention as an explicit action).
# ---------------------------------------------------------------------------
IDK_SCENARIOS = (
    ("no_evidence", "needs_evidence", "<needs_evidence>", "did the database migration complete on payments_api", "migration_done", "idk"),
    ("ambiguous", "ask_clarification", "<ask_clarification>", "which on-call engineer triggered the rollback", "rollback_actor", "idk"),
    ("needs_check", "needs_verification", "<needs_verification>", "is the latest release safe to deploy", "release_safe", "contingency"),
    ("unanswerable", "needs_evidence", "<needs_evidence>", "how many users were affected by the outage", "affected_users", "idk"),
)


def _idk_record(rng, *, template_id, idk_action, control, question, atom, primitive) -> dict[str, Any]:
    prim_token = "<prim:observation>" if primitive == "idk" else "<prim:contingency>"
    return assemble(
        family="pat_syn_idk_abstention",
        template_id=template_id,
        primitive_class=primitive,
        support_status="unknown",
        tool_intent="none",
        idk_action=idk_action,
        verifier_accept=False,
        schema_validity=True,
        text=question,
        target=control,
        event_graph=[
            {
                "event": "<evt:0>",
                "predicate": "answer",
                "arguments": [_arg("<arg:0>", atom, ["<role:undergoes_change>"])],
                "primitive": prim_token,
                "support": "<support:unknown>",
            }
        ],
        evidence_ids=[],
        rationale="insufficient grounding; abstain as an explicit action",
    )


def templates_idk_abstention() -> list[TemplateSpec]:
    specs = []
    for name, idk_action, control, question, atom, primitive in IDK_SCENARIOS:
        tid = f"pat_syn_idk_abstention:{name}"
        specs.append(TemplateSpec(tid, "positive", partial(
            _idk_record, template_id=tid, idk_action=idk_action, control=control, question=question, atom=atom, primitive=primitive)))
    return specs


# ---------------------------------------------------------------------------
# Family: evidence grounding (docs/datasets_aug.md 5.1/5.8/6.6). Correct,
# fake-id, stale, missing, and irrelevant evidence cases.
# ---------------------------------------------------------------------------
EVIDENCE_FACTS = ("service_a restarted", "tests_passed", "backup_done", "rows_written", "cache_warmed")


def _eg_event(atom: str, primitive_token: str, support_token: str) -> list[dict[str, Any]]:
    return [
        {
            "event": "<evt:0>",
            "predicate": "ground",
            "arguments": [_arg("<arg:0>", atom, ["<role:source>", "<role:exists_independently>"])],
            "primitive": primitive_token,
            "support": support_token,
        }
    ]


def _eg_correct(rng, *, template_id) -> dict[str, Any]:
    fact = rng.choice(EVIDENCE_FACTS)
    eid = _evidence_id(rng)
    return assemble(
        family="pat_syn_evidence_grounding",
        template_id=template_id,
        primitive_class="observation",
        support_status="belief",
        tool_intent="none",
        idk_action="answer",
        verifier_accept=True,
        schema_validity=True,
        text=f"evidence {eid} {fact} claim {fact}",
        target="<support:belief>",
        evidence=[{"id": eid, "source": "ci_log", "text": fact, "reliability": 0.97, "verified": True}],
        event_graph=_eg_event(fact, "<prim:observation>", "<support:belief>"),
        evidence_ids=[eid],
        rationale="claim is directly supported by a present, verified evidence span",
    )


def _eg_fake_id(rng, *, template_id) -> dict[str, Any]:
    fact = rng.choice(EVIDENCE_FACTS)
    real_id, fake_id = _evidence_id(rng), f"e_{rng.randint(1000, 9999)}"
    return assemble(
        family="pat_syn_evidence_grounding",
        template_id=template_id,
        primitive_class="observation",
        support_status="unknown",
        tool_intent="none",
        idk_action="needs_evidence",
        verifier_accept=False,
        schema_validity=True,
        text=f"claim {fact} per evidence {fake_id}",
        target="<unsupported>\n<needs_evidence>",
        evidence=[{"id": real_id, "source": "ci_log", "text": fact, "reliability": 0.9, "verified": False}],
        event_graph=_eg_event(fact, "<prim:observation>", "<support:unknown>"),
        evidence_ids=[fake_id],  # cites an id that is not in the evidence block
        is_hard_negative=True,
        hard_negative_type="fake_evidence",
        rationale="claim cites a fabricated evidence id absent from the block",
    )


def _eg_stale(rng, *, template_id) -> dict[str, Any]:
    fact = rng.choice(EVIDENCE_FACTS)
    eid = _evidence_id(rng)
    return assemble(
        family="pat_syn_evidence_grounding",
        template_id=template_id,
        primitive_class="observation",
        support_status="unknown",
        tool_intent="none",
        idk_action="needs_verification",
        verifier_accept=False,
        schema_validity=True,
        text=f"claim {fact} now per evidence {eid}",
        target="<needs_verification>",
        evidence=[{
            "id": eid, "source": "ci_log", "text": fact, "reliability": 0.8,
            "verified": False, "observed_at": "2021-01-01", "claim_window": "2026-05-30",
        }],
        event_graph=_eg_event(fact, "<prim:observation>", "<support:unknown>"),
        evidence_ids=[eid],
        is_hard_negative=True,
        hard_negative_type="stale_evidence",
        rationale="evidence is years older than the claim window",
    )


def _eg_missing(rng, *, template_id) -> dict[str, Any]:
    fact = rng.choice(EVIDENCE_FACTS)
    return assemble(
        family="pat_syn_evidence_grounding",
        template_id=template_id,
        primitive_class="contingency",
        support_status="unknown",
        tool_intent="none",
        idk_action="needs_evidence",
        verifier_accept=False,
        schema_validity=True,
        text=f"claim {fact} with no evidence provided",
        target="<needs_evidence>",
        evidence=[],
        event_graph=_eg_event(fact, "<prim:contingency>", "<support:unknown>"),
        evidence_ids=[],
        is_hard_negative=True,
        hard_negative_type="missing_evidence",
        rationale="claim made with no supporting evidence at all",
    )


def _eg_irrelevant(rng, *, template_id) -> dict[str, Any]:
    fact = rng.choice(EVIDENCE_FACTS)
    other = rng.choice([f for f in EVIDENCE_FACTS if f != fact])
    eid = _evidence_id(rng)
    return assemble(
        family="pat_syn_evidence_grounding",
        template_id=template_id,
        primitive_class="contingency",
        support_status="unknown",
        tool_intent="none",
        idk_action="needs_evidence",
        verifier_accept=False,
        schema_validity=True,
        text=f"claim {fact} citing evidence {eid} about {other}",
        target="<needs_evidence>",
        evidence=[{"id": eid, "source": "ci_log", "text": other, "reliability": 0.9, "verified": False}],
        event_graph=_eg_event(fact, "<prim:contingency>", "<support:unknown>"),
        evidence_ids=[],  # the present evidence does not support the claim
        is_hard_negative=True,
        hard_negative_type="irrelevant_evidence",
        rationale="present evidence is about a different fact than the claim",
    )


def templates_evidence_grounding() -> list[TemplateSpec]:
    return [
        TemplateSpec("pat_syn_evidence_grounding:correct", "positive",
                     partial(_eg_correct, template_id="pat_syn_evidence_grounding:correct")),
        TemplateSpec("pat_syn_evidence_grounding:fake_id", "negative",
                     partial(_eg_fake_id, template_id="pat_syn_evidence_grounding:fake_id")),
        TemplateSpec("pat_syn_evidence_grounding:stale", "negative",
                     partial(_eg_stale, template_id="pat_syn_evidence_grounding:stale")),
        TemplateSpec("pat_syn_evidence_grounding:missing", "negative",
                     partial(_eg_missing, template_id="pat_syn_evidence_grounding:missing")),
        TemplateSpec("pat_syn_evidence_grounding:irrelevant", "negative",
                     partial(_eg_irrelevant, template_id="pat_syn_evidence_grounding:irrelevant")),
    ]


FAMILY_TEMPLATES: dict[str, Callable[[], list[TemplateSpec]]] = {
    "pat_syn_modus_ponens": templates_modus_ponens,
    "pat_syn_abduction_trap": templates_abduction_trap,
    "pat_syn_contradiction": templates_contradiction,
    "pat_syn_tautology_no_progress": templates_tautology,
    "pat_syn_contingency": templates_contingency,
    "pat_syn_event_roles": templates_event_roles,
    "pat_syn_tool_grounding": templates_tool_grounding,
    "pat_syn_idk_abstention": templates_idk_abstention,
    "pat_syn_evidence_grounding": templates_evidence_grounding,
}


# ===========================================================================
# Balanced primitive-bridge families (--balanced-primitives).
#
# One family per primitive_class, balanced across the 14 classes. Each record
# carries event-role structure and a labels.bridge_binding that ties the
# primitive to the licensing argument, so the primitive is derivable from the
# event-role state (e.g. which atom is observed and its proto-roles) rather than
# from a copied <prim:*> token. Counterfactual pairs share a rule but differ in
# the observed binding (modus ponens vs abduction) or support ceiling.
# ===========================================================================
BRIDGE_RULES = (
    ("tests_passed", "release_ok", "verify_release"),
    ("compile_ok", "artifact_ready", "build_artifact"),
    ("auth_ok", "access_granted", "grant_access"),
    ("funds_available", "charge_ok", "settle_payment"),
    ("schema_valid", "ingest_ok", "ingest_records"),
    ("backup_done", "restore_ok", "restore_db"),
    ("canary_healthy", "rollout_ok", "promote_rollout"),
    ("cache_warm", "latency_ok", "warm_cache"),
    ("quota_ok", "request_admitted", "admit_request"),
    ("signature_valid", "package_trusted", "trust_package"),
    ("replica_synced", "failover_ready", "arm_failover"),
    ("lease_held", "writer_elected", "elect_writer"),
)
BRIDGE_TAILS = ("audited_ok", "billed_ok", "synced_ok", "verified_ok", "closed_ok", "archived_ok", "reconciled_ok")
SCOPES = ("tenant_7", "tenant_3", "region_eu", "project_x", "service_a", "region_us", "shard_9", "cluster_b")
BRIDGE_FACTS = ("service_a restarted", "payment_settled", "index_rebuilt", "user_notified",
                "snapshot_taken", "queue_drained", "cert_rotated", "alert_cleared",
                "token_revoked", "shard_rebalanced", "ledger_closed", "webhook_delivered")
SOURCES = ("ci_log", "tool_output", "pager_alert", "audit_log", "metrics_db", "human_note")

# ---------------------------------------------------------------------------
# Surface diversification (--diversify-surfaces).
#
# The frozen baseline rendered every logical atom as a bare repeated snake_case
# token (e.g. "tests_passed") in text, formula, AND evidence, so the gold
# argument span was a short, non-identifiable, context-free pointer target (see
# scripts/audit_span_ambiguity.py: ~76% of synthetic spans were repeated). On
# held-out rules the model had no learnable cue for *where* the argument was.
#
# When diversification is on, the argument MENTION in the natural-language text
# becomes a multi-token, context-bearing phrase ("evidence e_123 confirms the
# observed tests passed status"). The phrase is the gold span; it is introduced
# by a consistent frame (a START cue) and closed by a surface noun (an END cue),
# both of which generalise across held-out rules. The formula keeps the canonical
# <atom> token, so the rule/logic is untouched and the architecture claim holds.
_DIVERSIFY = False

_SURFACE_ADJ = ("observed", "reported", "recorded", "confirmed", "logged", "measured", "flagged")
_SURFACE_NOUN = ("status", "signal", "result", "state", "reading", "condition", "marker")
_SURFACE_FRAME = ("confirms", "shows", "indicates", "reports", "notes")


def _words(atom: str) -> str:
    """De-snake an atom into a natural multi-word surface ("tests passed")."""
    return atom.replace("_", " ")


def _mention(rng: random.Random, atom: str, *, negated: bool = False) -> str:
    """A natural, context-bearing mention of `atom` used as the gold argument
    span. Multi-token with a clear left frame word and right boundary noun, so
    the pointer cue generalises to held-out rules rather than memorising atoms."""

    adj = rng.choice(_SURFACE_ADJ)
    noun = rng.choice(_SURFACE_NOUN)
    if negated:
        return f"disputed {_words(atom)} {noun}"
    return f"{adj} {_words(atom)} {noun}"


# ---------------------------------------------------------------------------
# Support-status coverage (--support-coverage, composes with --diversify-surfaces).
#
# On the diversified baseline `support` showed a train/val generalisation gap
# (train ~0.97, val ~0.82): the head was partly reading rule-specific surface
# rather than the licensing condition. This layer states the support-licensing
# REASON in varied paraphrases (a rule-agnostic cue), so the same rule surface
# can carry different support depending only on the stated cue -- a counterfactual
# the model must read rather than memorise. Applied to honest records only; the
# adversarial hard negatives (abduction-as-proof, no-progress-as-proof, invalid
# schema) keep their misleading surface and are NOT given an honest justification.
_SUPPORT_COVERAGE = False

# Rule-agnostic licensing reason for each support status (varied surfaces).
_SUPPORT_CUE = {
    "proof": ("the antecedent is verified", "the premise checks out", "the cause is directly confirmed",
              "the supporting evidence is validated"),
    "belief": ("the premise is only plausible", "the antecedent is believed but unverified",
               "the evidence is suggestive not checked", "the support is provisional"),
    "hypothesis": ("only the consequent is observed", "the effect is seen but not the cause",
                   "this affirms the consequent", "the antecedent is merely guessed from the result"),
    "unknown": ("the required premise is missing", "no evidence is available either way",
                "the antecedent has not been established", "the determining fact is absent"),
    "conflict": ("one source affirms it and another denies it", "the claim and its negation are both asserted",
                 "the evidence points both ways", "two verified sources disagree"),
    "no_progress": ("it holds under every case", "it follows from the excluded middle",
                    "the statement is a tautology", "no case is ruled out"),
}
# Varied conclusion phrasings, so support is not tied to one trivial token.
_SUPPORT_PHRASE = {
    "proof": ("so it is proven", "so the conclusion is established", "so it holds by the rule", "so it is entailed"),
    "belief": ("so it is a working belief", "so it is taken as likely", "so it is provisionally accepted"),
    "hypothesis": ("so it is only a hypothesis", "so it is a candidate, not a proof", "so it stays conjectural"),
    "unknown": ("so the status is unknown", "so it cannot be determined yet", "so it stays open"),
    "conflict": ("so the evidence is in conflict", "so the claims are jointly unsatisfiable", "so it is contradictory"),
    "no_progress": ("so it adds no information", "so no progress is made", "so it is trivially true"),
}


def _support_clause(rng: random.Random, support: str) -> str:
    """A varied, rule-agnostic clause stating WHY `support` holds (cue + phrasing)."""
    cue = rng.choice(_SUPPORT_CUE[support])
    phrase = rng.choice(_SUPPORT_PHRASE[support])
    return f"{cue}, {phrase}"


# Register-primitive tokens for the classes that have no <prim:*> token. Binding
# the event to its own <reg:*> token (a valid special token) keeps idk/uncertainty
# from being serialized as <prim:contingency>. <reg:*> appears only in full render;
# event["primitive"] is never a training target, so this is safe.
REG_PRIMITIVE = {"idk": "<reg:idk>", "uncertainty": "<reg:uncertainty>",
                 "provenance": "<reg:provenance>", "schema": "<reg:schema>", "tool": "<reg:tool>"}
REG_PRIMITIVE_TOKENS = set(REG_PRIMITIVE.values())

# Primitive-distinguishing reason clauses for the three support=unknown families,
# so idk / uncertainty / contingency are separated by WHY they are unknown (the
# bridge reads this text under input-only), not just by the shared support cue.
_UNKNOWN_REASON = {
    "contingency": ("the required premise is missing, so the conclusion stays contingent",
                    "it depends on an unestablished premise, so it is undetermined",
                    "the deciding premise has not been settled, so it is contingent"),
    "uncertainty": ("the evidence is mixed and weak, so it cannot be settled",
                    "two weak sources disagree, so the estimate is uncertain",
                    "the signal is noisy and conflicting, so confidence is low"),
    "idk": ("there is no evidence either way, so the only honest action is to abstain",
            "nothing in the given facts decides this, so abstain rather than guess",
            "the question is ungrounded here, so abstain and ask for grounding"),
}


def _unknown_reason(rng: random.Random, primitive: str) -> str:
    return rng.choice(_UNKNOWN_REASON[primitive])


def _with_support(rng: random.Random, text: str, support: str) -> str:
    """Append the support-licensing clause when --support-coverage is on."""
    if _SUPPORT_COVERAGE and _DIVERSIFY and support in _SUPPORT_CUE:
        return f"{text}; {_support_clause(rng, support)}"
    return text


def _bridge_event(predicate: str, args: list[dict[str, Any]], prim_token: str, support_token: str) -> list[dict[str, Any]]:
    return [{"event": "<evt:0>", "predicate": predicate, "arguments": args,
             "primitive": prim_token, "support": support_token}]


def _b_modus_ponens(rng, *, template_id, rule, certified) -> dict[str, Any]:
    p, q, pred = rule
    ev = _evidence_id(rng)
    support = "proof" if certified else "belief"
    if _DIVERSIFY:
        scope = rng.choice(SCOPES)
        a0, a1 = _mention(rng, p), _mention(rng, q)
        text = (f"under the rule that {_words(p)} implies {_words(q)} in {scope}, "
                f"evidence {ev} {rng.choice(_SURFACE_FRAME)} {a0}, so {a1} follows by the rule")
        text = _with_support(rng, text, support)
    else:
        a0, a1 = p, q
        text = f"rule if {p} implies {q} evidence {ev} {p} therefore {q}"
    return assemble(
        family="pat_syn_modus_ponens", template_id=template_id, primitive_class="modus_ponens",
        support_status=support, tool_intent="none", idk_action="answer", verifier_accept=certified,
        schema_validity=True, text=text,
        target=f"<support:{support}>",
        evidence=[{"id": ev, "source": "ci_log", "text": p, "reliability": 0.98 if certified else 0.8, "verified": certified}],
        formula=f"<atom> {p} </atom> IMPLIES <atom> {q} </atom>",
        event_graph=_bridge_event(pred,
            [_arg("<arg:0>", a0, ["<role:source>", "<role:causes_change>"]),
             _arg("<arg:1>", a1, ["<role:goal>", "<role:undergoes_change>"])],
            "<prim:modus_ponens>", f"<support:{support}>"),
        evidence_ids=[ev], bridge_binding="<arg:0>",
        rationale="antecedent observed (arg0) -> consequent by modus ponens")


def _b_abduction(rng, *, template_id, rule, trap) -> dict[str, Any]:
    p, q, pred = rule
    ev = _evidence_id(rng)
    support = "proof" if trap else "hypothesis"
    connector = "therefore" if trap else "maybe"
    if _DIVERSIFY:
        scope = rng.choice(SCOPES)
        a0, a1 = _mention(rng, q), _mention(rng, p)
        text = (f"under the rule that {_words(p)} implies {_words(q)} in {scope}, "
                f"evidence {ev} {rng.choice(_SURFACE_FRAME)} {a0}, {connector} {a1} is the cause")
        if not trap:  # honest hypothesis only; the proof-trap keeps its misleading surface
            text = _with_support(rng, text, support)
    else:
        a0, a1 = q, p
        text = f"rule if {p} implies {q} evidence {ev} {q} {connector} {p}"
    return assemble(
        family="pat_syn_abduction", template_id=template_id, primitive_class="abduction",
        support_status=support, tool_intent="none", idk_action="answer" if trap else "needs_verification",
        verifier_accept=False, schema_validity=True,
        text=text,
        target=f"<support:{support}>",
        evidence=[{"id": ev, "source": "audit_log", "text": q, "reliability": 0.6, "verified": False}],
        formula=f"<atom> {p} </atom> IMPLIES <atom> {q} </atom>",
        event_graph=_bridge_event(pred,
            [_arg("<arg:0>", a0, ["<role:undergoes_change>", "<role:affected>"]),
             _arg("<arg:1>", a1, ["<role:source>"])],
            "<prim:abduction>", f"<support:{support}>"),
        evidence_ids=[ev], bridge_binding="<arg:0>",
        is_hard_negative=trap, hard_negative_type="abduction_as_proof" if trap else None,
        rationale="consequent observed (arg0) -> antecedent is hypothesis, not proof")


def _b_contradiction(rng, *, template_id, rule) -> dict[str, Any]:
    p = rule[0]
    e1, e2 = _evidence_id(rng), _evidence_id(rng)
    if _DIVERSIFY:
        a0, a1 = _mention(rng, p), _mention(rng, p, negated=True)
        text = (f"evidence {e1} {rng.choice(_SURFACE_FRAME)} {a0}, "
                f"while evidence {e2} {rng.choice(_SURFACE_FRAME)} {a1}")
        text = _with_support(rng, text, "conflict")
    else:
        a0, a1 = p, f"not {p}"
        text = f"evidence {e1} {p} and evidence {e2} not {p}"
    return assemble(
        family="pat_syn_contradiction", template_id=template_id, primitive_class="contradiction",
        support_status="conflict", tool_intent="none", idk_action="needs_verification", verifier_accept=False,
        schema_validity=True, text=text, target="<conflict>",
        evidence=[{"id": e1, "source": "pager_alert", "text": p, "reliability": 0.9, "verified": True},
                  {"id": e2, "source": "pager_alert", "text": f"not {p}", "reliability": 0.9, "verified": True}],
        formula=f"<atom> {p} </atom> AND NOT <atom> {p} </atom>",
        event_graph=_bridge_event("contradict",
            [_arg("<arg:0>", a0, ["<role:exists_independently>"]),
             _arg("<arg:1>", a1, ["<role:exists_independently>"])],
            "<prim:contradiction>", "<support:conflict>"),
        evidence_ids=[e1, e2], bridge_binding="<arg:1>",
        rationale="arg0 and its negation arg1 are jointly unsatisfiable")


def _b_tautology(rng, *, template_id, rule, trap) -> dict[str, Any]:
    atom = rule[0]
    support = "proof" if trap else "no_progress"
    if _DIVERSIFY:
        a0 = _mention(rng, atom)
        text = f"either {a0} holds or it does not, for the {_words(atom)} case"
        if trap:
            text = f"{text}, therefore proven"  # no-progress-as-proof trap keeps its surface
        else:
            text = _with_support(rng, text, "no_progress")
    else:
        a0 = atom
        text = f"either {atom} or not {atom}"
        text = f"{text} therefore proven" if trap else text
    return assemble(
        family="pat_syn_tautology", template_id=template_id, primitive_class="tautology",
        support_status=support, tool_intent="none", idk_action="answer", verifier_accept=False,
        schema_validity=True, text=text,
        target="<support:proof>" if trap else "<no_progress>",
        formula=f"<atom> {atom} </atom> OR NOT <atom> {atom} </atom>",
        event_graph=_bridge_event("disjoin",
            [_arg("<arg:0>", a0, ["<role:exists_independently>"])],
            "<prim:tautology>", f"<support:{support}>"),
        bridge_binding="<arg:0>", is_hard_negative=trap, hard_negative_type="no_progress" if trap else None,
        rationale="excluded middle on arg0 adds no information")


def _b_contingency(rng, *, template_id, rule) -> dict[str, Any]:
    p, q, _ = rule
    if _DIVERSIFY:
        a0, a1 = _mention(rng, p), _mention(rng, q)
        text = f"{a1} depends on {a0}, but {a0} has not been established yet"
        if _SUPPORT_COVERAGE:
            text = f"{text}; {_unknown_reason(rng, 'contingency')}"
    else:
        a0, a1 = p, q
        text = f"{q} depends on {p} but {p} is unknown"
    return assemble(
        family="pat_syn_contingency", template_id=template_id, primitive_class="contingency",
        support_status="unknown", tool_intent="none", idk_action="needs_evidence", verifier_accept=False,
        schema_validity=True, text=text, target="<needs_evidence>",
        evidence=[], formula=f"<atom> {p} </atom> IMPLIES <atom> {q} </atom>",
        event_graph=_bridge_event("determine",
            [_arg("<arg:0>", a0, ["<role:undergoes_change>"]),
             _arg("<arg:1>", a1, ["<role:goal>"])],
            "<prim:contingency>", "<support:unknown>"),
        evidence_ids=[], bridge_binding="<arg:0>",
        rationale="missing premise arg0 leaves the conclusion contingent")


def _b_observation(rng, *, template_id, fact, verified) -> dict[str, Any]:
    ev = _evidence_id(rng)
    support = "belief" if verified else "unknown"
    if _DIVERSIFY:
        a0 = _mention(rng, fact)
        text = f"evidence {ev} {rng.choice(_SURFACE_FRAME)} {a0}, recorded as the claim under review"
        text = _with_support(rng, text, support)
    else:
        a0 = fact
        text = f"evidence {ev} shows {fact} claim {fact}"
    return assemble(
        family="pat_syn_observation", template_id=template_id, primitive_class="observation",
        support_status=support, tool_intent="none", idk_action="answer" if verified else "needs_verification",
        verifier_accept=verified, schema_validity=True,
        text=text, target=f"<support:{support}>",
        evidence=[{"id": ev, "source": "tool_output", "text": fact, "reliability": 0.96 if verified else 0.55, "verified": verified}],
        event_graph=_bridge_event("observe",
            [_arg("<arg:0>", a0, ["<role:undergoes_change>", "<role:affected>"])],
            "<prim:observation>", f"<support:{support}>"),
        evidence_ids=[ev], bridge_binding="<arg:0>",
        rationale="claim arg0 is directly supported by an evidence span")


def _b_role_reversal(rng, *, template_id, verb) -> dict[str, Any]:
    a, b = rng.choice(SUBJECTS), rng.choice(SUBJECTS[::-1])
    if a == b:
        b = SUBJECTS[(SUBJECTS.index(a) + 1) % len(SUBJECTS)]
    return assemble(
        family="pat_syn_semantic_conflict", template_id=template_id, primitive_class="semantic_conflict",
        support_status="conflict", tool_intent="none", idk_action="ask_clarification", verifier_accept=False,
        schema_validity=True, text=f"{a} {verb} {b}", target="<role:role_reversal>",
        event_graph=_bridge_event(verb,
            [_arg("<arg:0>", b, ["<role:volition>", "<role:causes_change>"]),
             _arg("<arg:1>", a, ["<role:affected>"])],
            "<prim:semantic_conflict>", "<support:conflict>"),
        role_ambiguity="role_reversal", bridge_binding="<arg:0>",
        is_hard_negative=True, hard_negative_type="role_reversal",
        rationale="agent/patient roles reversed vs the surface text")


def _b_with_pp(rng, *, template_id, case) -> dict[str, Any]:
    verb, obj, pp = case
    return assemble(
        family="pat_syn_semantic_conflict", template_id=template_id, primitive_class="semantic_conflict",
        support_status="conflict", tool_intent="none", idk_action="ask_clarification", verifier_accept=False,
        schema_validity=True, text=f"john {verb} {obj} with {pp}", target="<role:ambiguity>",
        event_graph=_bridge_event(verb,
            [_arg("<arg:0>", "john", ["<role:volition>"]),
             _arg("<arg:1>", obj, ["<role:affected>"]),
             _arg("<arg:adjunct>", pp, ["<role:comitative>"])],
            "<prim:semantic_conflict>", "<support:conflict>"),
        role_ambiguity="with_pp", bridge_binding="<arg:adjunct>",
        is_hard_negative=True, hard_negative_type="with_pp_ambiguity",
        rationale="instrument PP wrongly attached as comitative")


def _b_policy_conflict(rng, *, template_id, rule) -> dict[str, Any]:
    p = rule[0]
    if _DIVERSIFY:
        a1 = _mention(rng, p)
        text = f"a read_only role attempted a write on {a1}, violating policy"
    else:
        a1 = p
        text = f"read_only role attempted write on {p}"
    return assemble(
        family="pat_syn_semantic_conflict", template_id=template_id, primitive_class="semantic_conflict",
        support_status="conflict", tool_intent="none", idk_action="needs_verification", verifier_accept=False,
        schema_validity=True, text=text, target="<conflict>",
        event_graph=_bridge_event("violate",
            [_arg("<arg:0>", "read_only", ["<role:exists_independently>"]),
             _arg("<arg:1>", a1, ["<role:affected>"])],
            "<prim:semantic_conflict>", "<support:conflict>"),
        bridge_binding="<arg:0>", rationale="policy conflict: read_only role with a write action")


def _b_axiom(rng, *, template_id, rule, scope) -> dict[str, Any]:
    p, q, _ = rule
    if _DIVERSIFY:
        a0, a1 = _mention(rng, p), _mention(rng, q)
        text = f"within scope {scope}, assume {a0} as given, so that {a1} is licensed there"
        text = _with_support(rng, text, "belief")
    else:
        a0, a1 = p, q
        text = f"scope {scope} rule if {p} implies {q} within {scope}"
    return assemble(
        family="pat_syn_axiom", template_id=template_id, primitive_class="axiom",
        support_status="belief", tool_intent="none", idk_action="answer", verifier_accept=False,
        schema_validity=True, text=text, target="<support:belief>",
        formula=f"<atom> {p} </atom> IMPLIES <atom> {q} </atom>",
        event_graph=_bridge_event("assume",
            [_arg("<arg:0>", a0, ["<role:source>", "<role:exists_independently>"]),
             _arg("<arg:1>", a1, ["<role:goal>"])],
            "<prim:axiom>", "<support:belief>"),
        bridge_binding="<arg:0>",
        rationale=f"scoped premise arg0 is an axiom inside {scope} (not global)")


def _b_syllogism(rng, *, template_id, rule, tail) -> dict[str, Any]:
    p, q, _ = rule
    r = tail
    if _DIVERSIFY:
        a0, a1, a2 = _mention(rng, q), _mention(rng, p), _mention(rng, r)
        text = (f"given that {_words(p)} implies {_words(q)} and {_words(q)} implies {_words(r)}, "
                f"the shared middle {a0} chains {a1} through to {a2}")
        text = _with_support(rng, text, "proof")
    else:
        a0, a1, a2 = q, p, r
        text = f"if {p} implies {q} and if {q} implies {r} therefore if {p} implies {r}"
    return assemble(
        family="pat_syn_syllogism", template_id=template_id, primitive_class="syllogism",
        support_status="proof", tool_intent="none", idk_action="answer", verifier_accept=True,
        schema_validity=True, text=text,
        target="<support:proof>",
        formula=f"( <atom> {p} </atom> IMPLIES <atom> {q} </atom> ) AND ( <atom> {q} </atom> IMPLIES <atom> {r} </atom> )",
        event_graph=_bridge_event("chain",
            [_arg("<arg:0>", a0, ["<role:source>", "<role:path>"]),
             _arg("<arg:1>", a1, ["<role:source>"]),
             _arg("<arg:2>", a2, ["<role:goal>"])],
            "<prim:syllogism>", "<support:proof>"),
        bridge_binding="<arg:0>",
        rationale="shared middle term arg0 licenses the compressed implication")


def _b_uncertainty(rng, *, template_id, fact) -> dict[str, Any]:
    e1, e2 = _evidence_id(rng), _evidence_id(rng)
    if _DIVERSIFY:
        a0 = _mention(rng, fact)
        text = f"whether {a0} actually holds is unclear"
        if _SUPPORT_COVERAGE:
            text = f"{text}; {_unknown_reason(rng, 'uncertainty')}"
        else:
            text = f"{text}, the evidence is mixed and weak"
    else:
        a0 = fact
        text = f"is {fact} true the evidence is mixed and weak"
    return assemble(
        family="pat_syn_uncertainty", template_id=template_id, primitive_class="uncertainty",
        support_status="unknown", tool_intent="none", idk_action="needs_verification", verifier_accept=False,
        schema_validity=True, text=text, target="<needs_verification>",
        evidence=[{"id": e1, "source": "metrics_db", "text": fact, "reliability": 0.45, "verified": False},
                  {"id": e2, "source": "human_note", "text": f"not {fact}", "reliability": 0.4, "verified": False}],
        # estimate (distinct), undergoes_change+affected (distinct from contingency),
        # uncertainty's own register token.
        event_graph=_bridge_event("estimate",
            [_arg("<arg:0>", a0, ["<role:undergoes_change>", "<role:affected>"])],
            REG_PRIMITIVE["uncertainty"], "<support:unknown>"),
        evidence_ids=[e1], bridge_binding="<arg:0>",
        rationale="conflicting weak evidence on arg0 -> epistemic uncertainty (not abstention or missing premise)")


def _b_provenance(rng, *, template_id, fact, src) -> dict[str, Any]:
    ev = _evidence_id(rng)
    if _DIVERSIFY:
        a0 = _mention(rng, fact)
        text = f"evidence {ev} from {src} {rng.choice(_SURFACE_FRAME)} {a0}, attributed to source {src}"
        text = _with_support(rng, text, "belief")
    else:
        a0 = fact
        text = f"evidence {ev} from {src} records {fact} provenance {src}"
    return assemble(
        family="pat_syn_provenance", template_id=template_id, primitive_class="provenance",
        support_status="belief", tool_intent="none", idk_action="answer", verifier_accept=True,
        schema_validity=True, text=text, target="<reg:provenance>",
        evidence=[{"id": ev, "source": src, "text": fact, "reliability": 0.95, "verified": True}],
        event_graph=_bridge_event("attribute",
            [_arg("<arg:0>", a0, ["<role:source>", "<role:exists_independently>"])],
            "<prim:observation>", "<support:belief>"),
        evidence_ids=[ev], bridge_binding="<arg:0>",
        rationale="arg0 tracked to its source for provenance")


_TOOL_INTENT_SCHEMAS = {
    "read": ("query_logs", QUERY_LOGS_SCHEMA, {"service": "billing_worker", "since": "1h"}, "logs_available"),
    "verify": ("verify_release", VERIFY_RELEASE_SCHEMA, None, "tests_passed"),
    "write": ("deploy_release", _tool_schema("deploy_release", {"run_id": "string", "target": "string"}, ["run_id", "target"], "Deploy a release."), None, "deploy_done"),
    "irreversible": ("rollback_release", _tool_schema("rollback_release", {"run_id": "string"}, ["run_id"], "Roll back a release."), None, "rollback_done"),
}


def _b_tool(rng, *, template_id, intent, fact) -> dict[str, Any]:
    tool_name, schema, fixed_args, default_fact = _TOOL_INTENT_SCHEMAS[intent]
    fact = fact or default_fact
    run = _run_id(rng)
    ev = _evidence_id(rng)
    args = fixed_args or ({"run_id": run, "requires": fact} if intent == "verify" else
                          {"run_id": run, "target": "prod"} if intent == "write" else {"run_id": run})
    call = render_hermes_tool_call(tool_name, args)
    if _DIVERSIFY:
        a0 = _mention(rng, fact)
        text = f"call {tool_name} for run {run}, grounded in {a0}"
        text = _with_support(rng, text, "belief")
    else:
        a0 = fact
        text = f"call {tool_name} for run {run} given {fact}"
    return assemble(
        family="pat_syn_tool", template_id=template_id, primitive_class="tool",
        support_status="belief", tool_intent=intent, idk_action="needs_verification", verifier_accept=True,
        schema_validity=True, text=text,
        target=f"<needs_verification>\n{call}",
        evidence=[{"id": ev, "source": "ci_log", "text": fact, "reliability": 0.96, "verified": True}],
        tools=[schema], evidence_ids=[ev],
        event_graph=_bridge_event(tool_name,
            [_arg("<arg:0>", a0, ["<role:source>"])],
            "<prim:observation>", "<support:belief>"),
        bridge_binding="<arg:0>",
        rationale=f"{intent} tool call grounded in evidence arg0")


def _b_schema(rng, *, template_id, rule, valid) -> dict[str, Any]:
    fact = rule[0]
    run = _run_id(rng)
    ev = _evidence_id(rng)
    args = {"run_id": run, "requires": fact} if valid else {"run_id": run}  # missing required field
    call = render_hermes_tool_call("verify_release", args)
    if _DIVERSIFY:
        a0 = _mention(rng, fact)
        text = f"validate the verify_release call for run {run}, which requires {a0}"
        if valid:  # invalid-schema conflict is a hard negative; keep its surface
            text = _with_support(rng, text, "belief")
    else:
        a0 = fact
        text = f"validate verify_release call for run {run} given {fact}"
    return assemble(
        family="pat_syn_schema", template_id=template_id, primitive_class="schema",
        support_status="belief" if valid else "conflict", tool_intent="verify",
        idk_action="needs_verification", verifier_accept=valid, schema_validity=valid,
        text=text,
        target=("<needs_verification>" if valid else "<conflict>") + f"\n{call}",
        evidence=[{"id": ev, "source": "ci_log", "text": fact, "reliability": 0.96, "verified": True}],
        tools=[VERIFY_RELEASE_SCHEMA], evidence_ids=[ev] if valid else [],
        event_graph=_bridge_event("validate",
            [_arg("<arg:0>", a0, ["<role:source>"])],
            "<prim:observation>" if valid else "<prim:semantic_conflict>", "<support:belief>" if valid else "<support:conflict>"),
        bridge_binding="<arg:0>", is_hard_negative=not valid,
        hard_negative_type=None if valid else "unsupported_tool_action",
        rationale="call matches schema" if valid else "call is missing a required field")


def _b_idk(rng, *, template_id, action, fact) -> dict[str, Any]:
    control = {"needs_evidence": "<needs_evidence>", "ask_clarification": "<ask_clarification>",
               "needs_verification": "<needs_verification>"}[action]
    if _DIVERSIFY:
        a0 = _mention(rng, fact)
        question = {"needs_evidence": f"did {a0} actually happen", "ask_clarification": f"who is responsible for {a0}",
                    "needs_verification": f"is {a0} safe to rely on"}[action]
        # idk-specific abstention reason (NOT the shared 'unknown' support clause,
        # which contingency/uncertainty also use) so the bridge can separate them.
        if _SUPPORT_COVERAGE:
            question = f"{question}; {_unknown_reason(rng, 'idk')}"
    else:
        a0 = fact
        question = {"needs_evidence": f"did {fact} happen", "ask_clarification": f"who caused {fact}",
                    "needs_verification": f"is {fact} safe to rely on"}[action]
    return assemble(
        family="pat_syn_idk", template_id=template_id, primitive_class="idk",
        support_status="unknown", tool_intent="none", idk_action=action, verifier_accept=False,
        schema_validity=True, text=question, target=control,
        # abstain (distinct predicate), exists_independently (distinct proto-role),
        # and the idk's own register token -- not <prim:contingency>.
        event_graph=_bridge_event("abstain",
            [_arg("<arg:0>", a0, ["<role:exists_independently>"])],
            REG_PRIMITIVE["idk"], "<support:unknown>"),
        evidence_ids=[], bridge_binding="<arg:0>",
        rationale="ungrounded query on arg0 -> abstain (not a missing-premise contingency)")


def _rule_name(rule: tuple[str, str, str]) -> str:
    return rule[2]


def templates_balanced(family: str, items: list[tuple[str, str, Any]]) -> list[TemplateSpec]:
    """Each template binds a distinct rule/fact so whole rules/domains are held
    out per split (no renamed-variable leakage). items: (suffix, kind, build_fn)
    where build_fn already has its varying parameter bound."""

    specs = []
    for suffix, kind, fn in items:
        tid = f"{family}:{suffix}"
        specs.append(TemplateSpec(tid, kind, partial(fn, template_id=tid)))
    return specs


def _mp_templates():
    return templates_balanced("pat_syn_modus_ponens", [
        (_rule_name(r), "positive", partial(_b_modus_ponens, rule=r, certified=(i % 2 == 0)))
        for i, r in enumerate(BRIDGE_RULES)])


def _abduction_templates():
    return templates_balanced("pat_syn_abduction", [
        (f"{_rule_name(r)}_{'trap' if i % 2 else 'hyp'}", "negative" if i % 2 else "positive",
         partial(_b_abduction, rule=r, trap=(i % 2 == 1)))
        for i, r in enumerate(BRIDGE_RULES)])


def _contradiction_templates():
    return templates_balanced("pat_syn_contradiction", [
        (_rule_name(r), "positive", partial(_b_contradiction, rule=r)) for r in BRIDGE_RULES])


def _tautology_templates():
    return templates_balanced("pat_syn_tautology", [
        (f"{_rule_name(r)}_{'trap' if i % 3 == 2 else 'tau'}", "negative" if i % 3 == 2 else "positive",
         partial(_b_tautology, rule=r, trap=(i % 3 == 2)))
        for i, r in enumerate(BRIDGE_RULES)])


def _contingency_templates():
    return templates_balanced("pat_syn_contingency", [
        (_rule_name(r), "positive", partial(_b_contingency, rule=r)) for r in BRIDGE_RULES])


def _observation_templates():
    return templates_balanced("pat_syn_observation", [
        (f"{f.replace(' ', '_')}_{'v' if i % 2 == 0 else 'u'}", "positive",
         partial(_b_observation, fact=f, verified=(i % 2 == 0)))
        for i, f in enumerate(BRIDGE_FACTS)])


def _semantic_conflict_templates():
    items = [(f"reversal_{v}", "negative", partial(_b_role_reversal, verb=v))
             for v in ("approved", "reviewed", "paid")]
    items += [(f"pp_{c[0]}", "negative", partial(_b_with_pp, case=c))
              for c in (("cut", "meat", "knife"), ("loaded", "truck", "rocks"))]
    items += [(f"policy_{_rule_name(r)}", "positive", partial(_b_policy_conflict, rule=r))
              for r in BRIDGE_RULES[:3]]
    return templates_balanced("pat_syn_semantic_conflict", items)


def _axiom_templates():
    return templates_balanced("pat_syn_axiom", [
        (_rule_name(r), "positive", partial(_b_axiom, rule=r, scope=SCOPES[i % len(SCOPES)]))
        for i, r in enumerate(BRIDGE_RULES)])


def _syllogism_templates():
    return templates_balanced("pat_syn_syllogism", [
        (_rule_name(r), "positive", partial(_b_syllogism, rule=r, tail=BRIDGE_TAILS[i % len(BRIDGE_TAILS)]))
        for i, r in enumerate(BRIDGE_RULES)])


def _uncertainty_templates():
    return templates_balanced("pat_syn_uncertainty", [
        (f.replace(" ", "_"), "positive", partial(_b_uncertainty, fact=f)) for f in BRIDGE_FACTS])


def _provenance_templates():
    return templates_balanced("pat_syn_provenance", [
        (f.replace(" ", "_"), "positive", partial(_b_provenance, fact=f, src=SOURCES[i % len(SOURCES)]))
        for i, f in enumerate(BRIDGE_FACTS)])


def _tool_templates():
    intents = ("read", "verify", "write", "irreversible")
    return templates_balanced("pat_syn_tool", [
        (f"{intent}_{j}", "positive", partial(_b_tool, intent=intent, fact=BRIDGE_FACTS[(i * 2 + j) % len(BRIDGE_FACTS)]))
        for i, intent in enumerate(intents) for j in range(2)])


def _schema_templates():
    return templates_balanced("pat_syn_schema", [
        (f"{_rule_name(r)}_{'valid' if i % 2 == 0 else 'invalid'}", "positive" if i % 2 == 0 else "negative",
         partial(_b_schema, rule=r, valid=(i % 2 == 0)))
        for i, r in enumerate(BRIDGE_RULES)])


def _idk_templates():
    actions = ("needs_evidence", "ask_clarification", "needs_verification")
    return templates_balanced("pat_syn_idk", [
        (f"{action}_{j}", "positive", partial(_b_idk, action=action, fact=BRIDGE_FACTS[(i * 2 + j) % len(BRIDGE_FACTS)]))
        for i, action in enumerate(actions) for j in range(2)])


# ---------------------------------------------------------------------------
# External-style augmentation (--external-aug).
#
# Entity-predicate records derived FROM the external distribution (FOLIO/
# ProofWriter style) but with identifiable, context-bearing argument spans and
# full support coverage -- including the classes the real external slice lacks
# (FOLIO has no belief/hypothesis/no_progress; neither source has hypothesis or
# no_progress; see scripts/audit_external_origin.py). Real external records are
# NEVER modified; these are a separate origin bucket (source=synthetic_derived_
# from_external, provenance.mix_origin="synthetic_ext"). The universal rule states
# the property; the specific instance (subject + property) is the gold span, so it
# is unique and multi-token. Requires --diversify-surfaces for the support clause.
# ---------------------------------------------------------------------------
EXT_SUBJECTS = ("rina", "the dog", "the bear", "alice", "the cat", "the squirrel",
                "dave", "the rabbit", "mia", "the engineer", "the auditor", "the tenant")
# Each case carries the universal-rule verb phrases AND distinct noun-phrase
# *topics* for the instance paraphrase. The gold span is built from the topic +
# a varied boundary noun, so the instance surface differs from the rule and the
# span END token is never shared with the universal rule (avoids both the old
# atom-recurrence END failure and a single-meta-noun cue shortcut).
EXT_CASES = (  # (ante_verb, cons_verb, relation, tail_verb, ante_topic, cons_topic, tail_topic)
    ("drinks coffee daily", "is dependent on caffeine", "depends_on", "sleeps poorly",
     "morning coffee", "caffeine reliance", "restless sleep"),
    ("visits the archive", "can read the old records", "enables", "cites the source",
     "archive visit", "record-reading", "source citation"),
    ("owns a valid license", "may deploy the service", "permits", "is billed monthly",
     "valid licensing", "deployment", "monthly billing"),
    ("passed the screening", "is cleared for access", "clears", "enters the lab",
     "screening pass", "access clearance", "lab entry"),
    ("signed the shared ledger", "is accountable for it", "binds", "reviews the totals",
     "ledger signing", "accountability", "total review"),
    ("trained on the dataset", "knows the safety policy", "implies", "passes the quiz",
     "dataset training", "safety-policy", "quiz result"),
    ("holds the active lease", "controls the resource", "grants", "pays the rent",
     "active lease", "resource control", "rent payment"),
    ("rotated the signing key", "secured the channel", "secures", "rotates again soon",
     "key rotation", "channel security", "next rotation"),
    ("filed the report", "is on the record", "records", "is audited later",
     "report filing", "on-record", "later audit"),
    ("joined the cohort", "follows the schedule", "obliges", "meets the mentor",
     "cohort joining", "schedule-following", "mentor meeting"),
)
# Varied, natural boundary nouns; the span ends in one of these (never in the
# rule), but the choice varies per record so it is not a single global cue.
EXT_BOUNDARIES = ("status", "state", "condition", "standing", "profile", "record")
EXT_SOURCE = "synthetic_derived_from_external"
EXT_PROV = {"source_dataset": "synthetic_ext", "derived_from_external": True,
            "binding": "derived", "mix_origin": "synthetic_ext"}


def _ext_prop(rng: random.Random, subj: str, topic: str) -> str:
    """Identifiable instance mention: subject + paraphrased topic + a varied
    boundary noun. The END token is one of EXT_BOUNDARIES (never shared with the
    universal rule); the topic paraphrases the predicate so the model must read
    external-style content, not a fixed cue."""
    return f"{subj}'s {topic} {rng.choice(EXT_BOUNDARIES)}"


def _atomize(prop: str) -> str:
    return prop.replace(" ", "_")


def _ext(**kw) -> dict[str, Any]:
    return assemble(source=EXT_SOURCE, provenance=EXT_PROV, **kw)


def _ext_modus_ponens(rng, *, template_id, case, certified) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, cons, rel, _, at, ct, _ = case
    ev = _evidence_id(rng)
    support = "proof" if certified else "belief"
    a0, a1 = _ext_prop(rng, subj, at), _ext_prop(rng, subj, ct)
    text = _with_support(rng, f"anyone who {ante} {cons}; evidence {ev} confirms {a0}, so {a1} follows", support)
    return _ext(family="ext_aug_modus_ponens", template_id=template_id, primitive_class="modus_ponens",
        support_status=support, tool_intent="none", idk_action="answer", verifier_accept=certified,
        schema_validity=True, text=text, target=f"<support:{support}>",
        evidence=[{"id": ev, "source": "theory", "text": ante, "reliability": 0.95 if certified else 0.8, "verified": certified}],
        formula=f"<atom> {_atomize(ante)} </atom> IMPLIES <atom> {_atomize(cons)} </atom>",
        event_graph=_bridge_event(rel,
            [_arg("<arg:0>", a0, ["<role:source>", "<role:causes_change>"]),
             _arg("<arg:1>", a1, ["<role:goal>", "<role:undergoes_change>"])],
            "<prim:modus_ponens>", f"<support:{support}>"),
        evidence_ids=[ev], bridge_binding="<arg:0>", rationale="entity antecedent observed -> consequent")


def _ext_abduction(rng, *, template_id, case) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, cons, rel, _, at, ct, _ = case
    ev = _evidence_id(rng)
    a0, a1 = _ext_prop(rng, subj, ct), _ext_prop(rng, subj, at)
    text = _with_support(rng, f"anyone who {ante} {cons}; evidence {ev} reports {a0}, maybe {a1} explains it", "hypothesis")
    return _ext(family="ext_aug_abduction", template_id=template_id, primitive_class="abduction",
        support_status="hypothesis", tool_intent="none", idk_action="needs_verification", verifier_accept=False,
        schema_validity=True, text=text, target="<support:hypothesis>",
        evidence=[{"id": ev, "source": "theory", "text": cons, "reliability": 0.6, "verified": False}],
        formula=f"<atom> {_atomize(ante)} </atom> IMPLIES <atom> {_atomize(cons)} </atom>",
        event_graph=_bridge_event(rel,
            [_arg("<arg:0>", a0, ["<role:undergoes_change>", "<role:affected>"]),
             _arg("<arg:1>", a1, ["<role:source>"])],
            "<prim:abduction>", "<support:hypothesis>"),
        evidence_ids=[ev], bridge_binding="<arg:0>", rationale="consequent observed -> antecedent is hypothesis")


def _ext_syllogism(rng, *, template_id, case) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, cons, rel, tail, at, ct, tt = case
    a0, a1, a2 = _ext_prop(rng, subj, ct), _ext_prop(rng, subj, at), _ext_prop(rng, subj, tt)
    text = _with_support(rng, f"anyone who {ante} {cons}, and anyone who {cons} {tail}; given {a1}, "
                              f"it follows that {a0} and then {a2}", "proof")
    return _ext(family="ext_aug_syllogism", template_id=template_id, primitive_class="syllogism",
        support_status="proof", tool_intent="none", idk_action="answer", verifier_accept=True,
        schema_validity=True, text=text, target="<support:proof>",
        formula=f"( <atom> {_atomize(ante)} </atom> IMPLIES <atom> {_atomize(cons)} </atom> ) AND "
                f"( <atom> {_atomize(cons)} </atom> IMPLIES <atom> {_atomize(tail)} </atom> )",
        event_graph=_bridge_event("chain",
            [_arg("<arg:0>", a0, ["<role:source>", "<role:path>"]),
             _arg("<arg:1>", a1, ["<role:source>"]),
             _arg("<arg:2>", a2, ["<role:goal>"])],
            "<prim:syllogism>", "<support:proof>"),
        bridge_binding="<arg:0>", rationale="shared middle property licenses the chain")


def _ext_contradiction(rng, *, template_id, case) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, at = case[0], case[4]
    e1, e2 = _evidence_id(rng), _evidence_id(rng)
    a0 = _ext_prop(rng, subj, at)
    a1 = f"the denied {at} {rng.choice(EXT_BOUNDARIES)}"
    text = _with_support(rng, f"evidence {e1} asserts {a0}, while evidence {e2} reports {a1}", "conflict")
    return _ext(family="ext_aug_contradiction", template_id=template_id, primitive_class="contradiction",
        support_status="conflict", tool_intent="none", idk_action="needs_verification", verifier_accept=False,
        schema_validity=True, text=text, target="<conflict>",
        evidence=[{"id": e1, "source": "theory", "text": ante, "reliability": 0.9, "verified": True},
                  {"id": e2, "source": "theory", "text": f"not {ante}", "reliability": 0.9, "verified": True}],
        formula=f"<atom> {_atomize(ante)} </atom> AND NOT <atom> {_atomize(ante)} </atom>",
        event_graph=_bridge_event("contradict",
            [_arg("<arg:0>", a0, ["<role:exists_independently>"]),
             _arg("<arg:1>", a1, ["<role:exists_independently>"])],
            "<prim:contradiction>", "<support:conflict>"),
        evidence_ids=[e1, e2], bridge_binding="<arg:1>", rationale="a claim and its negation are both asserted")


def _ext_contingency(rng, *, template_id, case) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, cons, at, ct = case[0], case[1], case[4], case[5]
    a0, a1 = _ext_prop(rng, subj, at), _ext_prop(rng, subj, ct)
    text = _with_support(rng, f"whether {a1} holds depends on whether {a0} is settled, which no evidence resolves", "unknown")
    return _ext(family="ext_aug_contingency", template_id=template_id, primitive_class="contingency",
        support_status="unknown", tool_intent="none", idk_action="needs_evidence", verifier_accept=False,
        schema_validity=True, text=text, target="<needs_evidence>", evidence=[],
        formula=f"<atom> {_atomize(ante)} </atom> IMPLIES <atom> {_atomize(cons)} </atom>",
        event_graph=_bridge_event("determine",
            [_arg("<arg:0>", a0, ["<role:undergoes_change>"]),
             _arg("<arg:1>", a1, ["<role:goal>"])],
            "<prim:contingency>", "<support:unknown>"),
        evidence_ids=[], bridge_binding="<arg:0>", rationale="missing premise leaves the conclusion contingent")


def _ext_observation(rng, *, template_id, case) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, at = case[0], case[4]
    ev = _evidence_id(rng)
    a0 = _ext_prop(rng, subj, at)
    text = _with_support(rng, f"evidence {ev} directly records {a0}, taken as the claim", "belief")
    return _ext(family="ext_aug_observation", template_id=template_id, primitive_class="observation",
        support_status="belief", tool_intent="none", idk_action="answer", verifier_accept=True,
        schema_validity=True, text=text, target="<support:belief>",
        evidence=[{"id": ev, "source": "theory", "text": ante, "reliability": 0.96, "verified": True}],
        event_graph=_bridge_event("observe",
            [_arg("<arg:0>", a0, ["<role:undergoes_change>", "<role:affected>"])],
            "<prim:observation>", "<support:belief>"),
        evidence_ids=[ev], bridge_binding="<arg:0>", rationale="claim directly supported by evidence")


def _ext_tautology(rng, *, template_id, case) -> dict[str, Any]:
    subj = rng.choice(EXT_SUBJECTS)
    ante, at = case[0], case[4]
    a0 = _ext_prop(rng, subj, at)
    text = _with_support(rng, f"either {a0} resolves or it does not, for the {ante} question", "no_progress")
    return _ext(family="ext_aug_tautology", template_id=template_id, primitive_class="tautology",
        support_status="no_progress", tool_intent="none", idk_action="answer", verifier_accept=False,
        schema_validity=True, text=text, target="<no_progress>",
        formula=f"<atom> {_atomize(ante)} </atom> OR NOT <atom> {_atomize(ante)} </atom>",
        event_graph=_bridge_event("disjoin",
            [_arg("<arg:0>", a0, ["<role:exists_independently>"])],
            "<prim:tautology>", "<support:no_progress>"),
        bridge_binding="<arg:0>", rationale="excluded middle adds no information")


def _ext_family(name: str, builder, kinds_by_index=None) -> Callable[[], list[TemplateSpec]]:
    def factory() -> list[TemplateSpec]:
        specs = []
        for i, case in enumerate(EXT_CASES):
            tid = f"ext_aug_{name}:{_atomize(case[0])}"
            kind = "negative" if (kinds_by_index and kinds_by_index(i)) else "positive"
            specs.append(TemplateSpec(tid, kind, partial(builder, template_id=tid, case=case)))
        return specs
    return factory


def _ext_mp_factory() -> list[TemplateSpec]:
    specs = []
    for i, case in enumerate(EXT_CASES):
        tid = f"ext_aug_modus_ponens:{_atomize(case[0])}"
        specs.append(TemplateSpec(tid, "positive", partial(_ext_modus_ponens, template_id=tid, case=case, certified=(i % 2 == 0))))
    return specs


EXTERNAL_AUG_FAMILY_TEMPLATES: dict[str, Callable[[], list[TemplateSpec]]] = {
    "ext_aug_modus_ponens": _ext_mp_factory,
    "ext_aug_abduction": _ext_family("abduction", _ext_abduction),
    "ext_aug_syllogism": _ext_family("syllogism", _ext_syllogism),
    "ext_aug_contradiction": _ext_family("contradiction", _ext_contradiction),
    "ext_aug_contingency": _ext_family("contingency", _ext_contingency),
    "ext_aug_observation": _ext_family("observation", _ext_observation),
    "ext_aug_tautology": _ext_family("tautology", _ext_tautology),
}


BALANCED_FAMILY_TEMPLATES: dict[str, Callable[[], list[TemplateSpec]]] = {
    "pat_syn_modus_ponens": _mp_templates,
    "pat_syn_abduction": _abduction_templates,
    "pat_syn_contradiction": _contradiction_templates,
    "pat_syn_tautology": _tautology_templates,
    "pat_syn_contingency": _contingency_templates,
    "pat_syn_observation": _observation_templates,
    "pat_syn_semantic_conflict": _semantic_conflict_templates,
    "pat_syn_axiom": _axiom_templates,
    "pat_syn_syllogism": _syllogism_templates,
    "pat_syn_uncertainty": _uncertainty_templates,
    "pat_syn_provenance": _provenance_templates,
    "pat_syn_tool": _tool_templates,
    "pat_syn_schema": _schema_templates,
    "pat_syn_idk": _idk_templates,
}


# ---------------------------------------------------------------------------
# Split assignment + record allocation.
# ---------------------------------------------------------------------------
def _assign_split(template_ids: Sequence[str], val_ratio: float, test_ratio: float) -> dict[str, str]:
    """Assign whole templates to splits. >=1 per split when there are >=3."""

    ordered = sorted(template_ids)
    n = len(ordered)
    if n == 0:
        return {}
    if n == 1:
        return {ordered[0]: "train"}
    if n == 2:
        return {ordered[0]: "train", ordered[1]: "test"}
    n_test = max(1, round(test_ratio * n))
    n_val = max(1, round(val_ratio * n))
    n_val = min(n_val, n - n_test - 1)
    out: dict[str, str] = {}
    for idx, tid in enumerate(ordered):
        if idx < n_test:
            out[tid] = "test"
        elif idx < n_test + n_val:
            out[tid] = "val"
        else:
            out[tid] = "train"
    return out


def generate_family(specs: Sequence[TemplateSpec], rng: random.Random, n: int, val_ratio: float, test_ratio: float) -> list[dict[str, Any]]:
    # Stratify split assignment by kind so positives and hard negatives each
    # appear across train/val/test.
    split_map: dict[str, str] = {}
    for kind in ("positive", "negative"):
        ids = [s.template_id for s in specs if s.kind == kind]
        split_map.update(_assign_split(ids, val_ratio, test_ratio))

    specs_by_split: dict[str, list[TemplateSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_split[split_map[spec.template_id]].append(spec)

    n_test = round(test_ratio * n)
    n_val = round(val_ratio * n)
    n_train = n - n_val - n_test
    targets = {"train": n_train, "val": n_val, "test": n_test}

    records: list[dict[str, Any]] = []
    for split in SPLITS:
        count = targets[split]
        pool = specs_by_split.get(split) or specs_by_split["train"]
        pool = sorted(pool, key=lambda s: s.template_id)
        for i in range(count):
            spec = pool[i % len(pool)]
            record = spec.build(rng)
            record["split"] = split
            records.append(record)
    return records


def build_dataset(records_per_family: int, seed: int, val_ratio: float, test_ratio: float,
                  balanced_primitives: bool = False, diversify_surfaces: bool = False,
                  support_coverage: bool = False, external_aug: bool = False) -> dict[str, list[dict[str, Any]]]:
    # Balanced mode generates one family per primitive_class, so primitive_class
    # is balanced across the 14 classes; default mode keeps the prior families.
    # diversify_surfaces toggles the natural, context-bearing argument mentions
    # (gold spans); support_coverage adds the varied support-licensing clause
    # (honest records only). external_aug builds the external-style augmentation
    # families instead (it implies diversified surfaces + support coverage). All
    # flags off => byte-identical baseline.
    global _DIVERSIFY, _SUPPORT_COVERAGE
    _DIVERSIFY = bool(diversify_surfaces) or bool(external_aug)
    _SUPPORT_COVERAGE = bool(support_coverage) or bool(external_aug)
    if external_aug:
        registry = EXTERNAL_AUG_FAMILY_TEMPLATES
    elif balanced_primitives:
        registry = BALANCED_FAMILY_TEMPLATES
    else:
        registry = FAMILY_TEMPLATES
    families: dict[str, list[dict[str, Any]]] = {}
    for family, templates_fn in registry.items():
        rng = random.Random(f"{seed}:{family}")
        records = generate_family(templates_fn(), rng, records_per_family, val_ratio, test_ratio)
        for index, record in enumerate(records):
            record["id"] = f"{family}_{index:06d}"
        families[family] = records
    return families


def _label_distribution(families: Mapping[str, Sequence[Mapping[str, Any]]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for records in families.values():
        for record in records:
            value = str(record["labels"].get(key))
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _manifest(families: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    total = sum(len(r) for r in families.values())
    per_family = {}
    split_totals: dict[str, int] = {s: 0 for s in SPLITS}
    hard_neg = 0
    for family, records in families.items():
        splits: dict[str, int] = {s: 0 for s in SPLITS}
        negs = 0
        for record in records:
            splits[record["split"]] += 1
            split_totals[record["split"]] += 1
            if record["is_hard_negative"]:
                negs += 1
                hard_neg += 1
        per_family[family] = {"records": len(records), "splits": splits, "hard_negatives": negs}
    return {
        "num_records": total,
        "num_families": len(families),
        "num_hard_negatives": hard_neg,
        "split_totals": split_totals,
        "per_family": per_family,
        "primitive_class_distribution": _label_distribution(families, "primitive_class"),
        "support_status_distribution": _label_distribution(families, "support_status"),
        "tool_intent_distribution": _label_distribution(families, "tool_intent"),
        "idk_action_distribution": _label_distribution(families, "idk_action"),
    }


def write_dataset(output_dir: Path, families: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale family files so the directory reflects the current build mode
    # (default vs balanced families differ). Generated artifacts are git-ignored.
    for stale in output_dir.glob("*.jsonl"):
        stale.unlink()
    paths: dict[str, Path] = {}
    for family, records in families.items():
        path = output_dir / f"{family}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths[family] = path
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(families), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the first CPU-only PAT-ER synthetic dataset (deterministic).")
    parser.add_argument("--records-per-family", type=int, default=100, help="Records generated per family.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Template-level validation fraction.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Template-level test fraction.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--balanced-primitives", action="store_true",
                        help="Generate 14 primitive-class-balanced bridge families instead of the default families.")
    parser.add_argument("--diversify-surfaces", action="store_true",
                        help="Render argument mentions (gold spans) as natural, context-bearing phrases so the "
                             "span pointer has a rule-agnostic cue (formula/rule stays canonical).")
    parser.add_argument("--support-coverage", action="store_true",
                        help="Add a varied, rule-agnostic support-licensing clause to honest records (composes with "
                             "--diversify-surfaces) so support is read from a stated reason, not a surface shortcut.")
    parser.add_argument("--external-aug", action="store_true",
                        help="Build external-style augmentation families (entity-predicate, identifiable spans, full "
                             "support coverage) tagged as a derived origin bucket. Implies diversify + support coverage.")
    args = parser.parse_args()

    families = build_dataset(
        records_per_family=args.records_per_family,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        balanced_primitives=args.balanced_primitives,
        diversify_surfaces=args.diversify_surfaces,
        support_coverage=args.support_coverage,
        external_aug=args.external_aug,
    )
    paths = write_dataset(args.output_dir, families)
    manifest = _manifest(families)

    print(f"PAT-ER synthetic dataset built (balanced_primitives={args.balanced_primitives})")
    print(f"records: {manifest['num_records']} across {manifest['num_families']} families "
          f"(hard negatives={manifest['num_hard_negatives']})")
    print(f"split totals: {manifest['split_totals']}")
    print(f"primitive_class distribution: {manifest['primitive_class_distribution']}")
    print(f"tool_intent distribution: {manifest['tool_intent_distribution']}")
    print("per family (records | train/val/test | hard_neg):")
    for family, info in manifest["per_family"].items():
        s = info["splits"]
        print(f"  {family:<32} {info['records']:>4} | {s['train']:>4}/{s['val']:>3}/{s['test']:>3} | {info['hard_negatives']:>3}")
    print("artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
