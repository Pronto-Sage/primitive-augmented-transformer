#!/usr/bin/env python3
"""Validate PAT-ER synthetic dataset JSONL files.

Checks each record against the canonical schema (NOTES[dataset].md section 1) and
the label vocabularies (section 5), plus PAT-ER-specific integrity:

- required canonical fields present with correct types;
- label values inside their controlled vocabularies;
- event-graph event/arg/role/primitive/support tokens are valid atomic special
  tokens from the tokenizer spec;
- target control tokens are valid special tokens;
- label evidence_ids resolve to evidence-block ids (except where a fake/stale
  evidence hard negative makes a dangling id intentional);
- Hermes tool calls in model_text parse back to a name + arguments object;
- model_text reproduces render_pater_prompt over the structured fields;
- ids are globally unique;
- each template_id belongs to exactly one split (family/template holdout).

Errors fail the run (non-zero exit). Softer consistency findings are reported as
warnings and do not fail the run.

CPU-only, no model, no downloads.

Usage:
    python3 scripts/validate_pater_dataset.py artifacts/datasets/pater_synthetic/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pat_er.sample_data import tokenize_with_pater_spec  # noqa: E402
from pat_er.serialization import parse_hermes_tool_calls  # noqa: E402
from pat_er.tokenizer_spec import build_pater_tokenizer_spec  # noqa: E402

import build_synthetic_pater_dataset as builder  # noqa: E402

SPEC = build_pater_tokenizer_spec(include_schema_key_candidates=True)
SPECIAL_TOKENS = set(SPEC.special_tokens)
EVENT_SLOTS = {"<evt:0>", "<evt:1>", "<evt:2>", "<evt:3>"}
ARG_SLOTS = {"<arg:0>", "<arg:1>", "<arg:2>", "<arg:3>", "<arg:4>", "<arg:adjunct>"}

REQUIRED_FIELDS = ("id", "split", "split_group", "source", "task_family", "text", "evidence", "event_graph", "tools", "target", "labels")
REQUIRED_LABELS = ("primitive_class", "support_status", "tool_intent", "idk_action", "schema_validity", "verifier_accept", "evidence_ids")
ALLOWED_HARD_NEGATIVE_TYPES = {
    "abduction_as_proof",
    "no_progress",
    "role_reversal",
    "with_pp_ambiguity",
    "unsupported_tool_action",
    "fake_evidence",
    "stale_evidence",
    "missing_evidence",
    "irrelevant_evidence",
    "unsupported_claim",
    "contradiction_in_roles",
    "symmetric_ambiguity",
}
# Hard negatives where a dangling/stale evidence id is intentional, so label
# evidence_ids are allowed to point outside the evidence block.
DANGLING_EVIDENCE_OK = {"fake_evidence", "stale_evidence"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_token_span(where: str, span: Any, expected_string: str, content_tokens: list[str], report: Report) -> None:
    """A token span must be null, or [start, end] inside content_tokens whose
    tokens exactly reproduce the span string (so labels can never drift)."""

    if span is None:
        return
    if (not isinstance(span, list)) or len(span) != 2 or not all(isinstance(x, int) for x in span):
        report.error(where, f"token span {span!r} must be null or [start, end] ints")
        return
    start, end = span
    if not (0 <= start <= end < len(content_tokens)):
        report.error(where, f"token span {span} out of bounds (len={len(content_tokens)})")
        return
    if content_tokens[start : end + 1] != tokenize_with_pater_spec(expected_string, SPEC):
        report.error(where, f"token span {span} does not match tokenized {expected_string!r}")


def validate_event_graph(where: str, event_graph: Any, content_tokens: list[str], report: Report) -> None:
    if not isinstance(event_graph, list):
        report.error(where, "event_graph must be a list")
        return
    for i, event in enumerate(event_graph):
        loc = f"{where} event_graph[{i}]"
        if not isinstance(event, dict):
            report.error(loc, "event must be an object")
            continue
        if event.get("event") not in EVENT_SLOTS:
            report.error(loc, f"event slot {event.get('event')!r} not in {sorted(EVENT_SLOTS)}")
        if not isinstance(event.get("predicate"), str) or not event.get("predicate"):
            report.error(loc, "predicate must be a non-empty string")
        if event.get("primitive") not in (builder.PRIMITIVE_TOKENS | builder.REG_PRIMITIVE_TOKENS):
            report.error(loc, f"primitive {event.get('primitive')!r} is not a valid <prim:*>/<reg:*> token")
        if event.get("support") not in builder.SUPPORT_TOKENS:
            report.error(loc, f"support {event.get('support')!r} is not a valid support token")
        if "predicate_span" in event and isinstance(event.get("predicate"), str):
            _check_token_span(f"{loc} predicate_span", event["predicate_span"], event["predicate"], content_tokens, report)
        arguments = event.get("arguments")
        if not isinstance(arguments, list):
            report.error(loc, "arguments must be a list")
            continue
        for j, arg in enumerate(arguments):
            aloc = f"{loc} arg[{j}]"
            if not isinstance(arg, dict):
                report.error(aloc, "argument must be an object")
                continue
            if arg.get("slot") not in ARG_SLOTS:
                report.error(aloc, f"slot {arg.get('slot')!r} not in {sorted(ARG_SLOTS)}")
            if not isinstance(arg.get("span"), str) or not arg.get("span"):
                report.error(aloc, "span must be a non-empty string")
            elif "token_span" in arg:
                _check_token_span(f"{aloc} token_span", arg["token_span"], arg["span"], content_tokens, report)
            roles = arg.get("proto_roles")
            if not isinstance(roles, list) or not roles:
                report.error(aloc, "proto_roles must be a non-empty list")
            else:
                for role in roles:
                    if role not in builder.ROLE_TOKENS:
                        report.error(aloc, f"proto_role {role!r} is not a valid <role:*> token")


def validate_target_tokens(where: str, target: Any, report: Report) -> None:
    if not isinstance(target, str):
        report.error(where, "target must be a string")
        return
    for line in target.splitlines():
        token = line.strip()
        # A bare control token line looks like "<...>" with no spaces/braces.
        if token.startswith("<") and token.endswith(">") and " " not in token and "{" not in token:
            if token not in SPECIAL_TOKENS:
                report.error(where, f"target control token {token!r} is not a valid special token")


def validate_tools_and_calls(where: str, record: dict[str, Any], report: Report) -> None:
    tools = record.get("tools")
    offered_names: set[str] = set()
    if not isinstance(tools, list):
        report.error(where, "tools must be a list")
    else:
        for i, tool in enumerate(tools):
            tloc = f"{where} tools[{i}]"
            if not isinstance(tool, dict):
                report.error(tloc, "tool must be an object")
                continue
            function = tool.get("function", tool)
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name:
                report.error(tloc, "tool must have a string function.name")
            else:
                offered_names.add(name)
            try:
                json.dumps(tool)
            except (TypeError, ValueError):
                report.error(tloc, "tool is not JSON-serializable")

    model_text = record.get("model_text")
    if isinstance(model_text, str) and "<tool_call>" in model_text:
        try:
            calls = parse_hermes_tool_calls(model_text)
        except Exception as exc:  # noqa: BLE001
            report.error(where, f"Hermes tool call failed to parse: {exc}")
            return
        if not calls:
            report.error(where, "model_text has <tool_call> but no parseable call")
        for call in calls:
            if not isinstance(call.name, str) or not call.name:
                report.error(where, "parsed tool call missing a name")
            if not isinstance(call.arguments, dict):
                report.error(where, "parsed tool call arguments must be an object")
            # Consistency: a schema-valid call should name an offered tool.
            if record.get("labels", {}).get("schema_validity") is True and offered_names and call.name not in offered_names:
                report.warn(where, f"schema_validity=true but tool call {call.name!r} is not in offered tools {sorted(offered_names)}")


def validate_record(where: str, record: Any, report: Report) -> None:
    if not isinstance(record, dict):
        report.error(where, "record must be a JSON object")
        return

    for field in REQUIRED_FIELDS:
        if field not in record:
            report.error(where, f"missing required field {field!r}")

    if record.get("split") not in builder.SPLITS:
        report.error(where, f"split {record.get('split')!r} not in {builder.SPLITS}")
    if not isinstance(record.get("split_group"), str) or not record.get("split_group"):
        report.error(where, "split_group must be a non-empty string")
    if not isinstance(record.get("text"), str) or not record.get("text"):
        report.error(where, "text must be a non-empty string")
    if "formula" in record and record["formula"] is not None and not isinstance(record["formula"], str):
        report.error(where, "formula must be a string or null")

    # Evidence block.
    evidence = record.get("evidence")
    evidence_block_ids: set[str] = set()
    if not isinstance(evidence, list):
        report.error(where, "evidence must be a list")
    else:
        for i, item in enumerate(evidence):
            eloc = f"{where} evidence[{i}]"
            if not isinstance(item, dict):
                report.error(eloc, "evidence item must be an object")
                continue
            for key in ("id", "source", "text"):
                if not isinstance(item.get(key), str) or not item.get(key):
                    report.error(eloc, f"evidence.{key} must be a non-empty string")
            if isinstance(item.get("id"), str):
                evidence_block_ids.add(item["id"])
            if "reliability" in item and not (_is_number(item["reliability"]) and 0.0 <= item["reliability"] <= 1.0):
                report.error(eloc, "evidence.reliability must be a number in [0, 1]")
            if "verified" in item and not isinstance(item["verified"], bool):
                report.error(eloc, "evidence.verified must be a bool")

    # Token-index span labels are checked against the model_text tokenization.
    model_text = record.get("model_text")
    content_tokens = tokenize_with_pater_spec(model_text, SPEC) if isinstance(model_text, str) else []
    validate_event_graph(where, record.get("event_graph"), content_tokens, report)
    validate_target_tokens(where, record.get("target"), report)
    validate_tools_and_calls(where, record, report)

    # Labels.
    labels = record.get("labels")
    if not isinstance(labels, dict):
        report.error(where, "labels must be an object")
    else:
        for key in REQUIRED_LABELS:
            if key not in labels:
                report.error(where, f"labels missing {key!r}")
        if labels.get("primitive_class") not in builder.PRIMITIVE_CLASSES:
            report.error(where, f"primitive_class {labels.get('primitive_class')!r} invalid")
        if labels.get("support_status") not in builder.SUPPORT_STATUSES:
            report.error(where, f"support_status {labels.get('support_status')!r} invalid")
        if labels.get("tool_intent") not in builder.TOOL_INTENTS:
            report.error(where, f"tool_intent {labels.get('tool_intent')!r} invalid")
        if labels.get("idk_action") not in builder.IDK_ACTIONS:
            report.error(where, f"idk_action {labels.get('idk_action')!r} invalid")
        if "event_type" in labels and labels.get("event_type") not in builder.EVENT_TYPES:
            report.error(where, f"event_type {labels.get('event_type')!r} invalid")
        if "role_ambiguity" in labels and labels.get("role_ambiguity") not in builder.ROLE_AMBIGUITY_CLASSES:
            report.error(where, f"role_ambiguity {labels.get('role_ambiguity')!r} invalid")
        if "is_hard_negative" in labels and labels.get("is_hard_negative") != bool(record.get("is_hard_negative")):
            report.error(where, "labels.is_hard_negative does not mirror top-level is_hard_negative")
        if "bridge_binding" in labels and labels.get("bridge_binding") is not None \
                and labels.get("bridge_binding") not in (ARG_SLOTS | EVENT_SLOTS):
            report.error(where, f"bridge_binding {labels.get('bridge_binding')!r} is not a valid arg/event slot")
        if not isinstance(labels.get("schema_validity"), bool):
            report.error(where, "labels.schema_validity must be a bool")
        if not isinstance(labels.get("verifier_accept"), bool):
            report.error(where, "labels.verifier_accept must be a bool")
        label_evidence_ids = labels.get("evidence_ids")
        if not isinstance(label_evidence_ids, list) or not all(isinstance(x, str) for x in label_evidence_ids):
            report.error(where, "labels.evidence_ids must be a list of strings")
        else:
            hn_type = record.get("hard_negative_type")
            if hn_type not in DANGLING_EVIDENCE_OK:
                for ev_id in label_evidence_ids:
                    if ev_id not in evidence_block_ids:
                        report.error(where, f"labels.evidence_ids references {ev_id!r} not in evidence block")
        # Consistency warning: event support should mirror support_status here.
        support_status = labels.get("support_status")
        for event in record.get("event_graph") or []:
            if isinstance(event, dict) and event.get("support") and event["support"] != f"<support:{support_status}>":
                report.warn(where, f"event support {event['support']} != <support:{support_status}>")
                break

    # Hard-negative flag consistency.
    is_hn = record.get("is_hard_negative", False)
    hn_type = record.get("hard_negative_type")
    if not isinstance(is_hn, bool):
        report.error(where, "is_hard_negative must be a bool")
    if hn_type is not None and hn_type not in ALLOWED_HARD_NEGATIVE_TYPES:
        report.error(where, f"hard_negative_type {hn_type!r} not allowed")
    if bool(is_hn) != (hn_type is not None):
        report.error(where, f"is_hard_negative={is_hn} inconsistent with hard_negative_type={hn_type!r}")

    # model_text reproducibility.
    model_text = record.get("model_text")
    if model_text is not None:
        try:
            expected = builder.render_model_text(record)
        except Exception as exc:  # noqa: BLE001
            report.error(where, f"could not re-render model_text: {exc}")
        else:
            if model_text != expected:
                report.error(where, "model_text does not match render_pater_prompt over the fields")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PAT-ER synthetic dataset JSONL files.")
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL files to validate.")
    parser.add_argument("--max-show", type=int, default=20, help="Max errors/warnings to print.")
    args = parser.parse_args()

    report = Report()
    seen_ids: dict[str, str] = {}
    template_splits: dict[str, set[str]] = defaultdict(set)
    total_records = 0
    per_file: dict[str, int] = {}

    files = [p for p in args.paths if p.is_file()]
    if not files:
        print("no input files matched", file=sys.stderr)
        sys.exit(2)

    for path in files:
        count = 0
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report.error(where, f"invalid JSON: {exc}")
                continue
            count += 1
            total_records += 1
            validate_record(where, record, report)

            rec_id = record.get("id")
            if isinstance(rec_id, str):
                if rec_id in seen_ids:
                    report.error(where, f"duplicate id {rec_id!r} (also in {seen_ids[rec_id]})")
                else:
                    seen_ids[rec_id] = where
            tid = record.get("template_id")
            split = record.get("split")
            if isinstance(tid, str) and isinstance(split, str):
                template_splits[tid].add(split)
        per_file[path.name] = count

    # Global: template/family holdout integrity.
    for tid, splits in sorted(template_splits.items()):
        if len(splits) > 1:
            report.error("split-integrity", f"template {tid!r} appears in multiple splits {sorted(splits)}")

    print("PAT-ER dataset validation")
    print(f"files: {len(files)}  records: {total_records}")
    for name, count in per_file.items():
        print(f"  {name}: {count}")
    print(f"distinct ids: {len(seen_ids)}  distinct templates: {len(template_splits)}")
    print(f"errors: {len(report.errors)}  warnings: {len(report.warnings)}")

    if report.warnings:
        print("\nwarnings (first shown):")
        for msg in report.warnings[: args.max_show]:
            print(f"  WARN {msg}")
    if report.errors:
        print("\nerrors (first shown):")
        for msg in report.errors[: args.max_show]:
            print(f"  ERR  {msg}")
        print(f"\nFAIL: {len(report.errors)} error(s)")
        sys.exit(1)
    print("\nPASS: all records valid")


if __name__ == "__main__":
    main()
