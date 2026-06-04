#!/usr/bin/env python3
"""Convert FOLIO-style first-order-logic entailment examples into PAT-ER records.

Input fixture schema (one JSON object per line under --input):

    {
      "id": "folio_train_0001",
      "theory_id": "f_0001",           # optional; defaults to id
      "split": "train",
      "license": "...",
      "premises": ["All dogs are mammals.", "Rex is a dog."],
      "premises_fol": ["∀x (Dog(x) → Mammal(x))", "Dog(rex)"],   # optional
      "conclusion": "Rex is a mammal.",
      "conclusion_fol": "Mammal(rex)",                                     # optional
      "label": "True",                 # True | False | Uncertain
      "focus": "Rex",                  # optional salient entity for the event arg
      "hard_negative": "swapped_entities"   # optional
    }

Mapping (by entailment label, not raw copy):
- True      -> modus_ponens (1 premise) or syllogism (>=2) ; support proof ;
- False     -> contradiction ; support conflict ;
- Uncertain -> contingency ; support unknown (needs_evidence) ;
- a swapped-entities item is a plausible-but-unsupported hard negative
  (semantic_conflict, is_hard_negative).

FOL is normalized into the formula field; no chain-of-thought is emitted.
No downloads, no training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import external_common as ec

FAMILY = "ext_folio"
FOLIO_LICENSE_NOTE = ("Yale-LILY/FOLIO v0.0 (GitHub @main): CC-BY-SA-4.0; "
                      "HF yale-nlp/FOLIO lists MIT (conflict recorded); cite Han et al. 2022 arXiv:2209.00840")


def infer_split(filename: str) -> str | None:
    name = filename.lower()
    if "train" in name:
        return "train"
    if "val" in name or "dev" in name:
        return "val"
    if "test" in name:
        return "test"
    return None


def normalize_folio(example: dict[str, Any], split_hint: str | None, index: int) -> dict[str, Any]:
    """Map an upstream FOLIO record into the converter's example schema. Our own
    fixture records (which carry id + license) pass through unchanged. Upstream
    FOLIO files vary: some carry story_id/example_id, some only premises/
    conclusion/conclusion-FOL/label; a running index keeps ids unique either way."""

    if "id" in example and "license" in example:
        normalized = dict(example)
        normalized.setdefault("split", split_hint or "train")
        return normalized
    split = split_hint or "train"
    premises = example.get("premises") or []
    if isinstance(premises, str):
        premises = re.split(r"[\n]+", premises)
    premises = [p.strip() for p in premises if isinstance(p, str) and p.strip()]
    example_id = example.get("example_id")
    story_id = example.get("story_id")
    source_id = str(example_id) if example_id is not None else f"{split}_{index}"
    theory_id = f"folio_story_{story_id}" if story_id is not None else f"folio_{source_id}"
    fol = example.get("conclusion-FOL") or example.get("conclusion_fol")
    if not fol:
        premises_fol = example.get("premises-FOL") or example.get("premises_fol") or []
        fol = " AND ".join(premises_fol) if premises_fol else None
    return {
        "id": f"folio_{source_id}",
        "theory_id": theory_id,
        "split": split,
        "license": FOLIO_LICENSE_NOTE,
        "premises": premises,
        "conclusion": example.get("conclusion"),
        "conclusion_fol": fol,
        "label": example.get("label"),
    }

# First-order-logic symbols -> PAT-ER formula operators (normal added tokens).
FOL_SYMBOLS = {"∀": " FORALL ", "∃": " EXISTS ", "→": " IMPLIES ", "↔": " IFF ",
               "∧": " AND ", "∨": " OR ", "¬": " NOT ", "⊕": " XOR ",
               "->": " IMPLIES ", "<->": " IFF ", "&": " AND ", "|": " OR ", "~": " NOT "}


def normalize_fol(fol: str | None) -> str | None:
    if not fol:
        return None
    text = fol
    for symbol, token in FOL_SYMBOLS.items():
        text = text.replace(symbol, token)
    return re.sub(r"\s+", " ", text).strip()


def render_text(premises: list[str], conclusion: str) -> str:
    return f"premises {' '.join(premises)} conclusion {conclusion}"


def _focus_arg(focus: str | None, conclusion: str) -> list[dict[str, Any]]:
    # arg0 = the salient entity/term that licenses the inference; arg1 = conclusion.
    head = focus or conclusion.split()[0]
    return [ec.arg("<arg:0>", head, ["<role:source>"]),
            ec.arg("<arg:1>", conclusion, ["<role:goal>"])]


def convert_example(example: dict[str, Any], render_mode: str = "full") -> tuple[dict[str, Any], str]:
    premises = list(example.get("premises", []))
    conclusion = example["conclusion"]
    label = str(example.get("label", "")).lower()
    source_id = example["id"]
    theory_id = example.get("theory_id", source_id)
    license_note = example.get("license", "unknown; verify source license before redistribution")
    focus = example.get("focus")
    # Compact: put the conclusion first (so it survives truncation); premises go
    # into the evidence block (after the query). Full: premises then conclusion.
    text = f"conclusion {conclusion}" if render_mode == "compact" else render_text(premises, conclusion)
    formula = normalize_fol(example.get("conclusion_fol"))
    evidence = ec.evidence_items(source_id, premises)
    common = dict(source_dataset="folio", source_split=example.get("split", "train"),
                  source_id=source_id, theory_id=theory_id, license_note=license_note, family=FAMILY,
                  text=text, evidence=evidence, formula=formula, binding="approximate",
                  source_payload={"premises": premises, "conclusion": conclusion, "label": example.get("label")})

    if example.get("hard_negative") == "swapped_entities":
        record = ec.to_pater_record(
            **common, primitive_class="semantic_conflict", support_status="conflict",
            idk_action="needs_verification", verifier_accept=False, target="<conflict>",
            event_graph=ec.single_event("relate", _focus_arg(focus, conclusion),
                                        "<prim:semantic_conflict>", "<support:conflict>"),
            bridge_binding="<arg:0>", is_hard_negative=True, hard_negative_type="unsupported_claim",
            rationale="entities swapped: conclusion is plausible but not entailed")
        return record, "semantic_conflict(hard-neg)"

    if label == "true":
        primitive = "modus_ponens" if len(premises) <= 1 else "syllogism"
        prim_token = "<prim:modus_ponens>" if primitive == "modus_ponens" else "<prim:syllogism>"
        record = ec.to_pater_record(
            **common, primitive_class=primitive, support_status="proof", idk_action="answer",
            verifier_accept=True, target="<support:proof>",
            event_graph=ec.single_event("entail", _focus_arg(focus, conclusion), prim_token, "<support:proof>"),
            bridge_binding="<arg:0>", rationale="conclusion is entailed by the premises")
        return record, primitive

    if label == "false":
        record = ec.to_pater_record(
            **common, primitive_class="contradiction", support_status="conflict",
            idk_action="needs_verification", verifier_accept=False, target="<conflict>",
            event_graph=ec.single_event("contradict", _focus_arg(focus, conclusion),
                                        "<prim:contradiction>", "<support:conflict>"),
            bridge_binding="<arg:0>", rationale="conclusion contradicts the premises")
        return record, "contradiction"

    # Uncertain / not-enough-info.
    record = ec.to_pater_record(
        **common, primitive_class="contingency", support_status="unknown",
        idk_action="needs_evidence", verifier_accept=False, target="<needs_evidence>",
        event_graph=ec.single_event("determine", _focus_arg(focus, conclusion),
                                    "<prim:contingency>", "<support:unknown>"),
        bridge_binding="<arg:0>", rationale="conclusion is undetermined by the premises")
    return record, "contingency"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FOLIO-style fixtures to PAT-ER JSONL (offline).")
    parser.add_argument("--input", type=Path, required=True, help="Directory of *.jsonl fixtures.")
    parser.add_argument("--output", type=Path, required=True, help="Output PAT-ER JSONL path.")
    parser.add_argument("--render-mode", choices=["full", "compact"], default="full",
                        help="compact = conclusion-first text + premises as evidence (fits a small context).")
    args = parser.parse_args()

    paths = sorted(Path(args.input).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl under {args.input}")
    records = []
    counts: dict[str, int] = {}
    skipped = 0
    index = 0
    for path in paths:
        split = infer_split(path.name)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            example = normalize_folio(json.loads(line), split, index)
            index += 1
            if not example.get("conclusion") or not example.get("label"):
                skipped += 1
                continue
            record, primitive = convert_example(example, args.render_mode)
            records.append(record)
            counts[primitive] = counts.get(primitive, 0) + 1

    ec.write_jsonl(args.output, records)
    print(f"converted {len(records)} FOLIO examples -> {args.output} (skipped {skipped})")
    print(f"primitive mapping: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
