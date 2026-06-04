#!/usr/bin/env python3
"""Shared external-dataset -> PAT-ER conversion helpers (offline).

Both convert_proofwriter.py and convert_folio.py reduce an external logic example
to the canonical PAT-ER record by reusing ``build_synthetic_pater_dataset.assemble``
(which renders model_text, computes token spans, and fills the canonical schema),
then attaching provenance. This guarantees the output validates under
``validate_pater_dataset.py``.

Design rules (docs/datasets_aug.md sections 6 and 9):
- external labels are mapped, never copied raw, into PAT-ER primitive/support/idk;
- provenance (source dataset/split/id/license/conversion version) is preserved;
- no hidden chain-of-thought is emitted: the proof is used only to build the
  structured event_graph/formula, never as a free-text target.

No downloads, no training. Outputs go under artifacts/datasets/external/ which is
git-ignored; the small input fixtures under fixtures/external/ are committed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_synthetic_pater_dataset as builder  # noqa: E402  (reuse assemble + span logic)

CONVERSION_VERSION = "v1"
SPLIT_MAP = {"train": "train", "test": "test", "dev": "val", "val": "val",
             "valid": "val", "validation": "val"}


def arg(slot: str, span: str, roles: Sequence[str]) -> dict[str, Any]:
    return {"slot": slot, "span": span, "proto_roles": list(roles)}


def single_event(predicate: str, args: list[dict[str, Any]], prim_token: str, support_token: str) -> list[dict[str, Any]]:
    return [{"event": "<evt:0>", "predicate": predicate, "arguments": args,
             "primitive": prim_token, "support": support_token}]


def evidence_items(source_id: str, facts: Sequence[str]) -> list[dict[str, Any]]:
    """Turn given facts/premises into PAT-ER evidence items (they are given, so verified)."""

    return [{"id": f"{source_id}_f{i}", "source": "theory", "text": fact, "reliability": 1.0, "verified": True}
            for i, fact in enumerate(facts)]


def to_pater_record(
    *,
    source_dataset: str,
    source_split: str,
    source_id: str,
    theory_id: str,
    license_note: str,
    family: str,
    primitive_class: str,
    support_status: str,
    idk_action: str,
    verifier_accept: bool,
    text: str,
    target: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    formula: str | None = None,
    event_graph: Sequence[Mapping[str, Any]] | None = None,
    bridge_binding: str | None = None,
    is_hard_negative: bool = False,
    hard_negative_type: str | None = None,
    rationale: str = "",
    binding: str = "approximate",
    source_payload: Mapping[str, Any] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical PAT-ER record from a mapped external example.

    split_group = "{dataset}:{theory_id}" so all questions from one theory/example
    hold out together (no renamed-variable leakage across splits).
    """

    split = SPLIT_MAP.get(source_split, "train")
    template_id = f"{source_dataset}:{theory_id}"
    record = builder.assemble(
        family=family, template_id=template_id, primitive_class=primitive_class,
        support_status=support_status, tool_intent="none", idk_action=idk_action,
        verifier_accept=verifier_accept, schema_validity=True, text=text, target=target,
        evidence=evidence, formula=formula, event_graph=event_graph,
        is_hard_negative=is_hard_negative, hard_negative_type=hard_negative_type,
        rationale=rationale, bridge_binding=bridge_binding,
    )
    record["id"] = f"{source_dataset}_{source_id}"
    record["split"] = split
    record["source"] = source_dataset
    record["provenance"] = {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_id": source_id,
        "theory_id": theory_id,
        "license_note": license_note,
        "conversion_version": CONVERSION_VERSION,
        "binding": binding,  # "exact" | "approximate" event-role binding
    }
    # Source-specific provenance (e.g. ProofWriter release/depth/world/subset) that
    # must survive for later audit; never rendered into model_text.
    if extra_provenance:
        record["provenance"].update(dict(extra_provenance))
    # Full upstream fields are preserved here for audit but are NOT forced into
    # model_text (which stays compact); this field is not rendered.
    if source_payload is not None:
        record["source_payload"] = dict(source_payload)
    return record


def read_fixture_dir(input_dir: Path) -> list[dict[str, Any]]:
    """Read every *.jsonl line under a fixture directory as one external example."""

    import json

    examples: list[dict[str, Any]] = []
    paths = sorted(input_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl fixtures under {input_dir}")
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path.name}:{lineno}: invalid JSON fixture: {exc}") from exc
    return examples


def write_jsonl(output_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
