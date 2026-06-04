#!/usr/bin/env python3
"""Build a representative PAT-ER tokenizer corpus.

This generates serialized PAT-ER records (via ``pat_er.serialization``) that
exercise the full surface a production tokenizer must carry without fragmenting:

- evidence blocks;
- formulas with ``<atom>`` and formula operators;
- event graphs with event/argument slots and proto-roles;
- OpenAI-style tool schemas;
- Hermes ``<tool_call>`` blocks;
- IDK / abstention control tokens;
- hard-negative cases:
  abduction-as-proof, fake-evidence identifiers, stale evidence, plausible but
  unsupported claims, role reversal, with-PP ambiguity, contradiction hidden in
  event roles, fluent-but-unsupported tool actions, tautology / no-progress.

The records are *training/inference serialization*, not labels embedded in free
text: every primitive, role, support, and control token comes from
``pat_er.tokenizer_spec`` so the tokenizer evaluation can measure atomicity and
fragmentation against the real interface.

This script is CPU-only and deterministic. It does not train, does not load any
model, and does not download anything. Outputs are written under
``artifacts/tokenizer_corpus/`` and are intentionally git-ignored.

Reusable entry point: ``build_corpus()`` returns ``(records, dynamic_ids)`` so
``eval_tokenizer_fragmentation.py`` can consume the corpus in-memory even if the
on-disk artifacts have not been written yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er.serialization import (  # noqa: E402  (path setup must precede import)
    build_reference_release_tool,
    canonical_json,
    render_event_graph,
    render_evidence_block,
    render_formula,
    render_hermes_tool_call,
    render_pater_prompt,
)

DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "tokenizer_corpus"

# ---------------------------------------------------------------------------
# Deterministic value vocabularies. Dynamic identifiers live here so we know
# exactly which IDs appear in the corpus when emitting dynamic_ids.json. Per the
# tokenizer spec these must NOT become special tokens; the evaluator measures how
# badly a real tokenizer fragments them.
# ---------------------------------------------------------------------------
SERVICES = ["service_a", "service_b", "payments_api", "auth_gateway", "billing_worker"]
ATOMS = ["tests_passed", "release_ok", "build_green", "deploy_done", "rollback_safe", "cache_warmed"]
SUBJECTS = ["alice", "bob", "carol", "dave", "erin", "frank"]
SYMMETRIC_VERBS = ["approved", "reviewed", "messaged", "called", "paid"]

EVIDENCE_IDS = [f"e_{n}" for n in (1, 2, 3, 7, 12, 17, 22, 31, 44, 58)]
CLAIM_IDS = [f"claim_{n}" for n in (1, 4, 9, 16, 25, 42)]
RUN_IDS = [f"r_{n}" for n in (11, 17, 22, 33, 47)]
SOURCE_IDS = ["ci_log", "tool_output", "pager_alert", "audit_log", "human_note"]
TOOL_NAMES = ["verify_release", "check_evidence", "query_logs", "deploy_release", "rollback_release"]
# Adversarial dynamic IDs called out in NOTES[tokenizer].md as things that must
# stay structured text rather than special tokens.
ADVERSARIAL_IDS = ["e_17", "claim_42", "tool_abc", "obs_2026_05_30_001", "e_prod_xj_17_a_993", "r_2026_05_30_0001"]


def _cycle(seq: Sequence[Any], i: int) -> Any:
    return seq[i % len(seq)]


def make_record(
    *,
    category: str,
    text: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    formula: str | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    controls: Sequence[str] | None = None,
    hermes_calls: Sequence[str] | None = None,
    is_negative: bool = False,
    hard_negative_type: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Render one PAT-ER record and capture isolated fragments for evaluation."""

    controls = list(controls or [])
    hermes_calls = list(hermes_calls or [])
    output_prelude = "\n".join([*controls, *hermes_calls])
    record_text = render_pater_prompt(
        text=text,
        evidence=evidence,
        formula=formula,
        events=events,
        tools=tools,
        output_prelude=output_prelude,
    )

    fragments: dict[str, list[str]] = {}
    if evidence:
        fragments["evidence"] = [render_evidence_block(evidence)]
    if formula:
        fragments["formula"] = [render_formula(formula)]
    if events:
        fragments["event_graph"] = [render_event_graph(events)]
    if tools:
        # One fragment per tool schema, matching the <tool> wrapper used inside
        # render_tools_block, so "tokens per tool schema" is per-schema.
        fragments["tool_schema"] = [f"<tool>\n{canonical_json(tool)}\n</tool>" for tool in tools]
    if hermes_calls:
        fragments["hermes_tool_call"] = list(hermes_calls)

    return {
        "category": category,
        "is_negative": bool(is_negative),
        "hard_negative_type": hard_negative_type,
        "note": note,
        "text": record_text,
        "fragments": fragments,
    }


def _arg(slot: str, span: str, roles: Sequence[str]) -> dict[str, Any]:
    return {"slot": slot, "span": span, "proto_roles": list(roles)}


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


VERIFY_RELEASE_TOOL = build_reference_release_tool()
CHECK_EVIDENCE_TOOL = _tool_schema(
    "check_evidence",
    {"evidence_id": "string", "claim_id": "string"},
    ["evidence_id", "claim_id"],
    "Check whether an evidence id supports a claim id.",
)
QUERY_LOGS_TOOL = _tool_schema(
    "query_logs",
    {"service": "string", "since": "string"},
    ["service"],
    "Query recent logs for a service.",
)


# ---------------------------------------------------------------------------
# Positive primitive scenarios.
# ---------------------------------------------------------------------------
def gen_observation() -> list[dict[str, Any]]:
    records = []
    for i, service in enumerate(SERVICES):
        ev_id = _cycle(EVIDENCE_IDS, i)
        src = _cycle(SOURCE_IDS, i)
        records.append(
            make_record(
                category="observation",
                text=f"evidence {ev_id} log shows {service} restarted",
                evidence=[{"id": ev_id, "source": src, "text": f"{service} restarted", "reliability": 0.95}],
                events=[
                    {
                        "event": "<evt:0>",
                        "predicate": "restart",
                        "arguments": [_arg("<arg:0>", service, ["<role:undergoes_change>", "<role:affected>"])],
                        "primitive": "<prim:observation>",
                        "support": "<support:belief>",
                    }
                ],
                controls=["<needs_verification>"],
                note="grounded observation from a single evidence item",
            )
        )
    return records


def gen_modus_ponens() -> list[dict[str, Any]]:
    records = []
    for i in range(6):
        premise = _cycle(ATOMS, i)
        conclusion = _cycle(ATOMS, i + 1)
        ev_id = _cycle(EVIDENCE_IDS, i + 1)
        records.append(
            make_record(
                category="modus_ponens",
                text=f"rule if {premise} implies {conclusion} evidence {ev_id} {premise}",
                evidence=[{"id": ev_id, "source": "ci_log", "text": premise, "reliability": 0.97}],
                formula=f"<atom> {premise} </atom> IMPLIES <atom> {conclusion} </atom>",
                events=[
                    {
                        "event": "<evt:0>",
                        "predicate": "imply",
                        "arguments": [
                            _arg("<arg:0>", premise, ["<role:source>", "<role:causes_change>"]),
                            _arg("<arg:1>", conclusion, ["<role:goal>", "<role:undergoes_change>"]),
                        ],
                        "primitive": "<prim:modus_ponens>",
                        "support": "<support:proof>",
                    }
                ],
                controls=["<support:proof>"],
                note="valid modus ponens with grounded premise",
            )
        )
    return records


def gen_syllogism() -> list[dict[str, Any]]:
    return [
        make_record(
            category="syllogism",
            text="all humans are mortal and socrates is human therefore socrates is mortal",
            formula="FORALL x ( <atom> human_x </atom> IMPLIES <atom> mortal_x </atom> ) AND <atom> human_socrates </atom>",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "entail",
                    "arguments": [
                        _arg("<arg:0>", "human_socrates", ["<role:source>"]),
                        _arg("<arg:1>", "mortal_socrates", ["<role:goal>"]),
                    ],
                    "primitive": "<prim:syllogism>",
                    "support": "<support:proof>",
                }
            ],
            controls=["<support:proof>"],
            note="universal-instantiation syllogism",
        ),
        make_record(
            category="syllogism",
            text="every release that is build_green and tests_passed is release_ok and r_17 is build_green and tests_passed",
            formula=(
                "FORALL x ( ( <atom> build_green </atom> AND <atom> tests_passed </atom> ) "
                "IMPLIES <atom> release_ok </atom> )"
            ),
            evidence=[
                {"id": "e_12", "source": "ci_log", "text": "build_green", "reliability": 0.98},
                {"id": "e_7", "source": "ci_log", "text": "tests_passed", "reliability": 0.97},
            ],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "entail",
                    "arguments": [_arg("<arg:0>", "r_17", ["<role:source>"])],
                    "primitive": "<prim:syllogism>",
                    "support": "<support:proof>",
                }
            ],
            note="two-premise syllogism grounded in two evidence ids",
        ),
    ]


def gen_abduction() -> list[dict[str, Any]]:
    records = []
    for i in range(3):
        observed = _cycle(["file_deleted", "service_a restarted", "cache_warmed"], i)
        cause = _cycle(["cleanup_job ran", "deploy_done", "prefetch ran"], i)
        ev_id = _cycle(EVIDENCE_IDS, i + 2)
        # Positive: abduction kept hypothesis-like, not proof.
        records.append(
            make_record(
                category="abduction",
                text=f"rule if {cause} implies {observed} evidence {ev_id} {observed} maybe {cause}",
                evidence=[{"id": ev_id, "source": "audit_log", "text": observed, "reliability": 0.6}],
                formula=f"<atom> {cause.replace(' ', '_')} </atom> IMPLIES <atom> {observed.replace(' ', '_')} </atom>",
                events=[
                    {
                        "event": "<evt:0>",
                        "predicate": "abduce",
                        "arguments": [
                            _arg("<arg:0>", cause, ["<role:source>", "<role:causes_change>"]),
                            _arg("<arg:1>", observed, ["<role:undergoes_change>", "<role:affected>"]),
                        ],
                        "primitive": "<prim:abduction>",
                        "support": "<support:hypothesis>",
                    }
                ],
                controls=["<support:hypothesis>", "<needs_verification>"],
                note="abduction proposed as hypothesis, not proof",
            )
        )
        # Hard negative: the SAME abduction mislabeled as proof.
        records.append(
            make_record(
                category="abduction_trap",
                text=f"rule if {cause} implies {observed} evidence {ev_id} {observed} therefore {cause}",
                evidence=[{"id": ev_id, "source": "audit_log", "text": observed, "reliability": 0.6}],
                formula=f"<atom> {cause.replace(' ', '_')} </atom> IMPLIES <atom> {observed.replace(' ', '_')} </atom>",
                events=[
                    {
                        "event": "<evt:0>",
                        "predicate": "abduce",
                        "arguments": [
                            _arg("<arg:0>", cause, ["<role:source>", "<role:causes_change>"]),
                            _arg("<arg:1>", observed, ["<role:undergoes_change>"]),
                        ],
                        "primitive": "<prim:abduction>",
                        "support": "<support:proof>",
                    }
                ],
                controls=["<support:proof>"],
                is_negative=True,
                hard_negative_type="abduction_as_proof",
                note="affirming-the-consequent: abduction asserted as proof",
            )
        )
    return records


def gen_role_reversal() -> list[dict[str, Any]]:
    records = []
    for i in range(4):
        agent = _cycle(SUBJECTS, i)
        patient = _cycle(SUBJECTS, i + 1)
        verb = _cycle(SYMMETRIC_VERBS, i)
        forward = {
            "event": "<evt:0>",
            "predicate": verb,
            "arguments": [
                _arg("<arg:0>", agent, ["<role:volition>", "<role:sentience>", "<role:causes_change>"]),
                _arg("<arg:1>", patient, ["<role:affected>", "<role:undergoes_change>"]),
            ],
            "primitive": "<prim:observation>",
            "support": "<support:belief>",
        }
        records.append(
            make_record(
                category="role_reversal",
                text=f"{agent} {verb} {patient}",
                events=[forward],
                note="canonical agent/patient binding",
            )
        )
        # Hard negative: text says agent->patient but the event graph swaps the
        # bindings and claims equivalence via <role:role_reversal>.
        swapped = {
            "event": "<evt:0>",
            "predicate": verb,
            "arguments": [
                _arg("<arg:0>", patient, ["<role:volition>", "<role:causes_change>"]),
                _arg("<arg:1>", agent, ["<role:affected>"]),
            ],
            "primitive": "<prim:observation>",
            "support": "<support:belief>",
        }
        records.append(
            make_record(
                category="role_reversal",
                text=f"{agent} {verb} {patient}",
                events=[swapped],
                controls=["<role:role_reversal>", "<unsupported>"],
                is_negative=True,
                hard_negative_type="role_reversal",
                note="patient/agent bindings reversed relative to the text",
            )
        )
    return records


def gen_with_pp_ambiguity() -> list[dict[str, Any]]:
    cases = [
        ("john cut meat with knife", "knife", "<role:instrument>", "instrument reading"),
        ("john burgled house with accomplice", "accomplice", "<role:comitative>", "comitative reading"),
        ("john loaded truck with rocks", "rocks", "<role:incremental_theme>", "incremental-theme reading"),
        ("mary measured cloth with ruler", "ruler", "<role:instrument>", "instrument reading"),
    ]
    records = []
    for text, pp, role, note in cases:
        records.append(
            make_record(
                category="with_pp_ambiguity",
                text=text,
                events=[
                    {
                        "event": "<evt:0>",
                        "predicate": text.split()[1],
                        "arguments": [
                            _arg("<arg:0>", text.split()[0], ["<role:volition>", "<role:causes_change>"]),
                            _arg("<arg:1>", text.split()[2], ["<role:affected>"]),
                            _arg("<arg:adjunct>", pp, [role]),
                        ],
                        "primitive": "<prim:observation>",
                        "support": "<support:belief>",
                    }
                ],
                note=f"correctly disambiguated PP ({note})",
            )
        )
    # Hard negative: instrument PP mislabeled as comitative companion.
    records.append(
        make_record(
            category="with_pp_ambiguity",
            text="john cut meat with knife",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "cut",
                    "arguments": [
                        _arg("<arg:0>", "john", ["<role:volition>"]),
                        _arg("<arg:1>", "meat", ["<role:affected>"]),
                        _arg("<arg:adjunct>", "knife", ["<role:comitative>"]),
                    ],
                    "primitive": "<prim:observation>",
                    "support": "<support:belief>",
                }
            ],
            controls=["<role:ambiguity>", "<unsupported>"],
            is_negative=True,
            hard_negative_type="with_pp_ambiguity",
            note="instrument PP wrongly attached as comitative",
        )
    )
    return records


def gen_contradiction() -> list[dict[str, Any]]:
    records = [
        make_record(
            category="contradiction",
            text="evidence e_22 service_a is up and evidence e_31 service_a is down",
            evidence=[
                {"id": "e_22", "source": "pager_alert", "text": "service_a up", "reliability": 0.9},
                {"id": "e_31", "source": "pager_alert", "text": "service_a down", "reliability": 0.9},
            ],
            formula="<atom> service_a_up </atom> CONTRADICTS <atom> service_a_down </atom>",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "contradict",
                    "arguments": [
                        _arg("<arg:0>", "service_a up", ["<role:exists_independently>"]),
                        _arg("<arg:1>", "service_a down", ["<role:exists_independently>"]),
                    ],
                    "primitive": "<prim:contradiction>",
                    "support": "<support:conflict>",
                }
            ],
            controls=["<conflict>"],
            note="explicit contradiction between two evidence items",
        ),
        make_record(
            category="semantic_conflict",
            text="the bachelor is married",
            formula="<atom> bachelor_x </atom> CONTRADICTS <atom> married_x </atom>",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "be",
                    "arguments": [_arg("<arg:0>", "bachelor", ["<role:exists_independently>"])],
                    "primitive": "<prim:semantic_conflict>",
                    "support": "<support:conflict>",
                }
            ],
            controls=["<conflict>"],
            note="lexical/semantic conflict, not evidence conflict",
        ),
        # Hard negative: contradiction hidden inside the role bindings while the
        # surface text reads as coherent progress.
        make_record(
            category="contradiction_in_roles",
            text="the rock moved itself to the summit and stayed where it started",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "move",
                    "arguments": [
                        _arg("<arg:0>", "rock", ["<role:moves>", "<role:stationary>"]),
                        _arg("<arg:1>", "summit", ["<role:goal>", "<role:source>"]),
                    ],
                    "primitive": "<prim:contradiction>",
                    "support": "<support:conflict>",
                }
            ],
            controls=["<conflict>", "<unsupported>"],
            is_negative=True,
            hard_negative_type="contradiction_in_roles",
            note="same arg holds moves+stationary and source==goal",
        ),
    ]
    return records


def gen_idk() -> list[dict[str, Any]]:
    return [
        make_record(
            category="idk",
            text="did the database migration complete on payments_api",
            evidence=[],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "complete",
                    "arguments": [_arg("<arg:0>", "migration", ["<role:undergoes_change>"])],
                    "primitive": "<prim:contingency>",
                    "support": "<support:unknown>",
                }
            ],
            controls=["<IDK>", "<needs_evidence>"],
            note="no evidence available; abstain with IDK",
        ),
        make_record(
            category="idk",
            text="which on-call engineer triggered the rollback last night",
            controls=["<ask_clarification>", "<needs_evidence>"],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "trigger",
                    "arguments": [_arg("<arg:0>", "engineer", ["<role:volition>", "<role:sentience>"])],
                    "primitive": "<prim:contingency>",
                    "support": "<support:unknown>",
                }
            ],
            note="underspecified question; ask for clarification",
        ),
    ]


def gen_tautology() -> list[dict[str, Any]]:
    return [
        make_record(
            category="tautology",
            text="either the deploy is rollback_safe or it is not rollback_safe",
            formula="<atom> rollback_safe </atom> OR NOT <atom> rollback_safe </atom>",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "disjoin",
                    "arguments": [_arg("<arg:0>", "rollback_safe", ["<role:exists_independently>"])],
                    "primitive": "<prim:tautology>",
                    "support": "<support:no_progress>",
                }
            ],
            controls=["<no_progress>"],
            note="law of excluded middle, correctly flagged as no progress",
        ),
        # Hard negative: a tautology dressed up as a real conclusion.
        make_record(
            category="tautology",
            text="tests_passed implies tests_passed therefore the release is proven",
            formula="<atom> tests_passed </atom> IMPLIES <atom> tests_passed </atom>",
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "imply",
                    "arguments": [_arg("<arg:0>", "tests_passed", ["<role:source>", "<role:goal>"])],
                    "primitive": "<prim:tautology>",
                    "support": "<support:proof>",
                }
            ],
            controls=["<support:proof>"],
            is_negative=True,
            hard_negative_type="no_progress",
            note="p IMPLIES p sold as progress/proof",
        ),
    ]


def gen_fake_evidence() -> list[dict[str, Any]]:
    records = []
    # Fake evidence id: cited id is absent from the evidence block.
    records.append(
        make_record(
            category="fake_evidence",
            text="release_ok because tests_passed per evidence e_404",
            evidence=[{"id": "e_2", "source": "ci_log", "text": "tests_passed", "reliability": 0.97}],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "support",
                    "arguments": [_arg("<arg:0>", "release_ok", ["<role:goal>"])],
                    "primitive": "<prim:observation>",
                    "support": "<support:belief>",
                    "evidence_ids": ["e_404"],
                }
            ],
            controls=["<unsupported>", "<needs_evidence>"],
            is_negative=True,
            hard_negative_type="fake_evidence",
            note="claim cites evidence id e_404 not present in the block",
        )
    )
    # Stale evidence: evidence observed long before the claim window.
    records.append(
        make_record(
            category="fake_evidence",
            text="deploy_done now because deploy_done was observed earlier",
            evidence=[
                {
                    "id": "e_58",
                    "source": "ci_log",
                    "text": "deploy_done",
                    "reliability": 0.8,
                    "observed_at": "2021-01-01",
                    "claim_window": "2026-05-30",
                }
            ],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "support",
                    "arguments": [_arg("<arg:0>", "deploy_done", ["<role:goal>"])],
                    "primitive": "<prim:observation>",
                    "support": "<support:belief>",
                    "evidence_ids": ["e_58"],
                }
            ],
            controls=["<unsupported>", "<needs_verification>"],
            is_negative=True,
            hard_negative_type="stale_evidence",
            note="evidence from 2021 used to support a 2026 claim",
        )
    )
    # Plausible unsupported claim: no evidence at all but asserted as belief.
    records.append(
        make_record(
            category="fake_evidence",
            text="the payments_api is fully GDPR compliant",
            evidence=[],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "be",
                    "arguments": [_arg("<arg:0>", "payments_api", ["<role:exists_independently>"])],
                    "primitive": "<prim:contingency>",
                    "support": "<support:unknown>",
                }
            ],
            controls=["<unsupported>", "<needs_evidence>"],
            is_negative=True,
            hard_negative_type="unsupported_claim",
            note="fluent claim with no supporting evidence",
        )
    )
    return records


def gen_tool_intent() -> list[dict[str, Any]]:
    records = []
    # Grounded positive: arguments come from evidence in the same record.
    records.append(
        make_record(
            category="tool_intent",
            text="verify the release candidate r_17 whose tests_passed",
            evidence=[
                {"id": "e_17", "source": "ci_log", "text": "tests_passed", "reliability": 0.97},
                {"id": "e_22", "source": "ci_log", "text": "run_id r_17", "reliability": 0.99},
            ],
            events=[
                {
                    "event": "<evt:0>",
                    "predicate": "verify_release",
                    "arguments": [
                        _arg("<arg:0>", "r_17", ["<role:undergoes_change>"]),
                        _arg("<arg:1>", "tests_passed", ["<role:source>"]),
                    ],
                    "primitive": "<prim:modus_ponens>",
                    "support": "<support:belief>",
                }
            ],
            tools=[VERIFY_RELEASE_TOOL],
            controls=["<needs_verification>"],
            hermes_calls=[
                render_hermes_tool_call("verify_release", {"run_id": "r_17", "requires": "tests_passed"})
            ],
            note="grounded tool call: every argument traces to evidence",
        )
    )
    records.append(
        make_record(
            category="tool_intent",
            text="check whether evidence e_12 supports claim_9",
            evidence=[{"id": "e_12", "source": "ci_log", "text": "build_green", "reliability": 0.98}],
            tools=[CHECK_EVIDENCE_TOOL],
            controls=["<needs_verification>"],
            hermes_calls=[render_hermes_tool_call("check_evidence", {"evidence_id": "e_12", "claim_id": "claim_9"})],
            note="grounded evidence-check tool call",
        )
    )
    records.append(
        make_record(
            category="tool_intent",
            text="pull recent logs for billing_worker",
            tools=[QUERY_LOGS_TOOL],
            hermes_calls=[render_hermes_tool_call("query_logs", {"service": "billing_worker", "since": "1h"})],
            note="grounded log-query tool call",
        )
    )
    # Hard negative: fluent tool call that is not supported by schema/evidence.
    records.append(
        make_record(
            category="unsupported_tool_action",
            text="ship the release to production right now",
            evidence=[{"id": "e_2", "source": "ci_log", "text": "tests_passed", "reliability": 0.97}],
            tools=[VERIFY_RELEASE_TOOL],  # only verify_release is offered
            controls=["<unsupported>"],
            hermes_calls=[
                # deploy_release is NOT in the offered tools, and prod target is
                # ungrounded: fluent but unsupported action.
                render_hermes_tool_call("deploy_release", {"run_id": "r_17", "target": "production"})
            ],
            is_negative=True,
            hard_negative_type="unsupported_tool_action",
            note="calls a tool absent from <tools> with ungrounded arguments",
        )
    )
    records.append(
        make_record(
            category="unsupported_tool_action",
            text="verify the release",
            evidence=[],  # no evidence for the required argument
            tools=[VERIFY_RELEASE_TOOL],
            controls=["<needs_evidence>", "<unsupported>"],
            hermes_calls=[
                render_hermes_tool_call("verify_release", {"run_id": "r_999", "requires": "tests_passed"})
            ],
            is_negative=True,
            hard_negative_type="unsupported_tool_action",
            note="required argument run_id r_999 is not grounded in any evidence",
        )
    )
    return records


GENERATORS = [
    gen_observation,
    gen_modus_ponens,
    gen_syllogism,
    gen_abduction,
    gen_role_reversal,
    gen_with_pp_ambiguity,
    gen_contradiction,
    gen_idk,
    gen_tautology,
    gen_fake_evidence,
    gen_tool_intent,
]


def build_corpus() -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(records, dynamic_ids)`` deterministically.

    ``records`` each carry ``id``, ``category``, negative flags, the full
    serialized ``text``, and isolated ``fragments`` for per-construct metrics.
    ``dynamic_ids`` is the set of identifiers that must remain structured text
    (never special tokens) for the evaluator's fragmentation check.
    """

    records: list[dict[str, Any]] = []
    for generator in GENERATORS:
        records.extend(generator())
    for i, record in enumerate(records):
        record["id"] = f"rec_{i:04d}"

    dynamic_ids = list(
        dict.fromkeys(
            EVIDENCE_IDS + CLAIM_IDS + RUN_IDS + SOURCE_IDS + TOOL_NAMES + ADVERSARIAL_IDS
        )
    )
    return records, dynamic_ids


def _manifest(records: Sequence[Mapping[str, Any]], dynamic_ids: Sequence[str]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    by_hard_negative: dict[str, int] = {}
    fragment_counts: dict[str, int] = {}
    negatives = 0
    for record in records:
        by_category[record["category"]] = by_category.get(record["category"], 0) + 1
        if record["is_negative"]:
            negatives += 1
            key = record["hard_negative_type"] or "unspecified"
            by_hard_negative[key] = by_hard_negative.get(key, 0) + 1
        for frag_type, frags in record["fragments"].items():
            fragment_counts[frag_type] = fragment_counts.get(frag_type, 0) + len(frags)
    return {
        "num_records": len(records),
        "num_negatives": negatives,
        "num_positives": len(records) - negatives,
        "records_by_category": dict(sorted(by_category.items())),
        "hard_negatives_by_type": dict(sorted(by_hard_negative.items())),
        "fragment_counts": dict(sorted(fragment_counts.items())),
        "num_dynamic_ids": len(dynamic_ids),
    }


def write_corpus(output_dir: Path, records: Sequence[Mapping[str, Any]], dynamic_ids: Sequence[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus": output_dir / "corpus.jsonl",
        "dynamic_ids": output_dir / "dynamic_ids.json",
        "manifest": output_dir / "manifest.json",
        "preview": output_dir / "corpus_preview.txt",
    }

    with paths["corpus"].open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    paths["dynamic_ids"].write_text(json.dumps(list(dynamic_ids), ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = _manifest(records, dynamic_ids)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable preview: one record per category for quick eyeballing.
    seen_categories: set[str] = set()
    preview_chunks: list[str] = []
    for record in records:
        if record["category"] in seen_categories:
            continue
        seen_categories.add(record["category"])
        header = f"# {record['id']} | category={record['category']}"
        if record["is_negative"]:
            header += f" | HARD NEGATIVE={record['hard_negative_type']}"
        preview_chunks.append(f"{header}\n{record['text']}")
    paths["preview"].write_text("\n\n".join(preview_chunks) + "\n", encoding="utf-8")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PAT-ER tokenizer corpus (CPU-only, deterministic).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write corpus artifacts.")
    args = parser.parse_args()

    records, dynamic_ids = build_corpus()
    paths = write_corpus(args.output_dir, records, dynamic_ids)
    manifest = _manifest(records, dynamic_ids)

    print("PAT-ER tokenizer corpus built")
    print(f"records: {manifest['num_records']} (positives={manifest['num_positives']}, negatives={manifest['num_negatives']})")
    print("records by category:")
    for category, count in manifest["records_by_category"].items():
        print(f"  {category}: {count}")
    print("hard negatives by type:")
    for kind, count in manifest["hard_negatives_by_type"].items():
        print(f"  {kind}: {count}")
    print("fragment counts:")
    for frag_type, count in manifest["fragment_counts"].items():
        print(f"  {frag_type}: {count}")
    print(f"dynamic ids: {manifest['num_dynamic_ids']}")
    print("artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
