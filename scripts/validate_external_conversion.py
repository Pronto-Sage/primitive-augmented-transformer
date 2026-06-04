#!/usr/bin/env python3
"""External-conversion invariants for PAT-ER records (complements validate_pater_dataset.py).

validate_pater_dataset.py checks the canonical PAT-ER schema. This adds the
external-ingestion-specific gates from docs/datasets_aug.md section 9:

- every record carries complete provenance (source_dataset, source_split,
  source_id, license_note, conversion_version);
- no hidden chain-of-thought: the target is only control tokens and/or Hermes
  tool-call JSON, never free-text reasoning;
- the primitive label is a mapped PAT-ER class (not a verbatim copy of the source
  answer/label).

Exit non-zero on any violation. CPU-only, no downloads.

Usage:
    python3 scripts/validate_external_conversion.py artifacts/datasets/external/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er.tokenizer_spec import build_pater_tokenizer_spec  # noqa: E402

SPECIAL_TOKENS = set(build_pater_tokenizer_spec(include_schema_key_candidates=True).special_tokens)
REQUIRED_PROVENANCE = ("source_dataset", "source_split", "source_id", "license_note", "conversion_version")
PRIMITIVE_CLASSES = (
    "axiom", "observation", "contingency", "contradiction", "tautology", "modus_ponens",
    "syllogism", "abduction", "semantic_conflict", "uncertainty", "provenance", "tool", "schema", "idk",
)


def _target_is_clean(target: str) -> str | None:
    """A target line must be a control token, or part of a tool-call JSON block.
    Returns an error string if a free-text (chain-of-thought) line is found."""

    for line in target.splitlines():
        token = line.strip()
        if not token:
            continue
        if token.startswith("<") and token.endswith(">") and " " not in token:
            if token not in SPECIAL_TOKENS:
                return f"target control token {token!r} is not a special token"
            continue
        # JSON / tool-call payload lines are allowed.
        if token[0] in "{}[]\"" or '"name"' in token or '"arguments"' in token or token.endswith(",}"):
            continue
        if token.startswith("{") or token.endswith("}"):
            continue
        return f"target line looks like free-text chain-of-thought: {token[:60]!r}"
    return None


def validate_record(where: str, record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{where}: missing provenance object")
    else:
        for key in REQUIRED_PROVENANCE:
            if not provenance.get(key):
                errors.append(f"{where}: provenance.{key} missing/empty")

    target = record.get("target")
    if not isinstance(target, str):
        errors.append(f"{where}: target must be a string")
    else:
        problem = _target_is_clean(target)
        if problem:
            errors.append(f"{where}: {problem}")

    primitive = record.get("labels", {}).get("primitive_class")
    if primitive not in PRIMITIVE_CLASSES:
        errors.append(f"{where}: primitive_class {primitive!r} not a mapped PAT-ER class")
    # A raw source label should never appear verbatim as the primitive class.
    raw = str((record.get("provenance") or {}).get("source_answer", "")).lower()
    if raw and raw == str(primitive).lower():
        errors.append(f"{where}: primitive_class copies the raw source label {raw!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate external-conversion invariants for PAT-ER records.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    files = [p for p in args.paths if p.is_file()]
    if not files:
        print("no input files matched", file=sys.stderr)
        sys.exit(2)

    errors: list[str] = []
    total = 0
    by_dataset: dict[str, int] = {}
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            ds = (record.get("provenance") or {}).get("source_dataset", "?")
            by_dataset[ds] = by_dataset.get(ds, 0) + 1
            errors.extend(validate_record(f"{path.name}:{lineno}", record))

    print(f"external conversion validation: {total} records across {len(files)} files")
    print(f"by source_dataset: {dict(sorted(by_dataset.items()))}")
    if errors:
        for msg in errors[:20]:
            print(f"  ERR {msg}")
        print(f"\nFAIL: {len(errors)} error(s)")
        sys.exit(1)
    print("PASS: provenance complete, no chain-of-thought targets, labels mapped")


if __name__ == "__main__":
    main()
