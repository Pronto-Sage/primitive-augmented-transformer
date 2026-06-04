#!/usr/bin/env python3
"""Convert NLI (e-SNLI / SNLI / ANLI schema) examples into PAT-ER records (offline).

The motivation is the contradiction boundary audit
(docs/results/contradiction_boundary_audit.md): ProofWriter contradiction is
reasoning-hard (refuted vs provable share surface). NLI gives **explicit-conflict**
contradiction (premise + hypothesis that overtly clash), a learnable surface
signal -- like the synthetic contradiction (recall 1.00), unlike ProofWriter's.

Input fixture schema (one JSON object per line under --input):

    {
      "id": "nli_0001",
      "split": "train",                 # train | dev/val | test
      "label": "contradiction",         # entailment | neutral | contradiction (or 0/1/2)
      "premise": "...",                 # or "context" (ANLI)
      "hypothesis": "...",
      "license": "..."                  # source license note
    }

e-SNLI explanation fields are IGNORED (no chain-of-thought is emitted).

Mapping (NLI label -> PAT-ER), per the explicit-conflict gate:
  - contradiction -> contradiction (conflict): hypothesis overtly clashes with premise;
  - neutral / not-enough-info -> uncertainty (unknown): premise does not settle it;
  - entailment / supported -> observation (belief): hypothesis is supported.

The hypothesis is the bound argument (arg0); the premise is the grounding context.
No proof/derivation text, no explanations -- only the premise/hypothesis surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import external_common as ec

FAMILY = "ext_nli"
DEFAULT_LICENSE = "NLI source; verify dataset license (e-SNLI/SNLI: CC-BY-SA-4.0 prior; ANLI: likely CC-BY-NC) before redistribution"

# Accept string or HF integer labels (0=entailment, 1=neutral, 2=contradiction).
_LABEL_NORM = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction",
               "e": "entailment", "n": "neutral", "c": "contradiction",
               "0": "entailment", "1": "neutral", "2": "contradiction",
               "supports": "entailment", "refutes": "contradiction", "nei": "neutral",
               "not enough info": "neutral"}


def infer_split(filename: str) -> str | None:
    name = filename.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return None


def _theory_id(premise: str) -> str:
    """Hold out by premise so all hypotheses of one premise share a split."""
    return "nli_" + hashlib.sha1(premise.strip().encode("utf-8")).hexdigest()[:12]


def convert_example(example: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    premise = (example.get("premise") or example.get("context") or "").strip()
    hypothesis = (example.get("hypothesis") or "").strip()
    raw = example.get("label")
    label = _LABEL_NORM.get(str(raw).strip().lower()) if raw is not None else None
    if not premise or not hypothesis or label is None:
        return None  # skip malformed / unlabeled (-1) rows

    source_id = example["id"]
    theory_id = example.get("theory_id") or _theory_id(premise)
    license_note = example.get("license", DEFAULT_LICENSE)
    text = f"premise {premise} hypothesis {hypothesis}"
    evidence = ec.evidence_items(source_id, [premise])
    common = dict(source_dataset=example.get("source", "nli"), source_split=example.get("split", "train"),
                  source_id=source_id, theory_id=theory_id, license_note=license_note, family=FAMILY,
                  text=text, evidence=evidence,
                  extra_provenance={"nli_label": label, "premise_holdout": theory_id})

    if label == "contradiction":
        # explicit surface conflict: hypothesis clashes with the premise.
        args = [ec.arg("<arg:0>", hypothesis, ["<role:exists_independently>"]),
                ec.arg("<arg:1>", premise, ["<role:exists_independently>"])]
        record = ec.to_pater_record(
            **common, primitive_class="contradiction", support_status="conflict",
            idk_action="needs_verification", verifier_accept=False, target="<conflict>",
            formula=f"<atom> {hypothesis} </atom> CONTRADICTS <atom> {premise} </atom>",
            event_graph=ec.single_event("contradict", args, "<prim:contradiction>", "<support:conflict>"),
            bridge_binding="<arg:0>", binding="approximate",
            rationale="hypothesis (arg0) overtly contradicts the premise (arg1)")
        return record, "contradiction"

    if label == "neutral":
        args = [ec.arg("<arg:0>", hypothesis, ["<role:undergoes_change>", "<role:affected>"])]
        record = ec.to_pater_record(
            **common, primitive_class="uncertainty", support_status="unknown",
            idk_action="needs_verification", verifier_accept=False, target="<needs_verification>",
            event_graph=ec.single_event("estimate", args, "<reg:uncertainty>", "<support:unknown>"),
            bridge_binding="<arg:0>", binding="approximate",
            rationale="premise does not settle the hypothesis (arg0) -> uncertain")
        return record, "uncertainty"

    # entailment
    args = [ec.arg("<arg:0>", hypothesis, ["<role:undergoes_change>", "<role:affected>"])]
    record = ec.to_pater_record(
        **common, primitive_class="observation", support_status="belief",
        idk_action="answer", verifier_accept=True, target="<support:belief>",
        event_graph=ec.single_event("observe", args, "<prim:observation>", "<support:belief>"),
        bridge_binding="<arg:0>", binding="approximate",
        rationale="hypothesis (arg0) is supported by the premise")
    return record, "observation"


def _iter_examples(input_dir: Path):
    for path in sorted(input_dir.glob("*.jsonl")):
        split = infer_split(path.name)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec.setdefault("split", split or "train")
            yield rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NLI (e-SNLI/SNLI/ANLI) to PAT-ER JSONL (offline).")
    parser.add_argument("--input", type=Path, required=True, help="Directory of *.jsonl files.")
    parser.add_argument("--output", type=Path, required=True, help="Output PAT-ER JSONL path.")
    parser.add_argument("--max-records", type=int, default=None, help="Cap total converted records.")
    args = parser.parse_args()

    records, counts = [], {}
    for example in _iter_examples(Path(args.input)):
        out = convert_example(example)
        if out is None:
            continue
        record, primitive = out
        records.append(record)
        counts[primitive] = counts.get(primitive, 0) + 1
        if args.max_records and len(records) >= args.max_records:
            break

    ec.write_jsonl(args.output, records)
    print(f"converted {len(records)} NLI examples -> {args.output}")
    print(f"primitive mapping: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
