#!/usr/bin/env python3
"""Summarize a PAT-ER synthetic dataset.

Descriptive statistics over one or more dataset JSONL files: family/split counts,
label distributions, hard-negative breakdown, evidence/tool coverage, Hermes
parse rate, and reference-tokenizer token statistics over the rendered
``model_text``. This does not enforce correctness (use validate_pater_dataset.py
for that); it reports what the dataset contains.

CPU-only, no model training, no downloads.

Usage:
    python3 scripts/summarize_pater_dataset.py artifacts/datasets/pater_synthetic/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

from pat_er.sample_data import build_reference_tokenizer  # noqa: E402
from pat_er.serialization import parse_hermes_tool_calls  # noqa: E402
from pat_er.tokenizer_spec import build_pater_tokenizer_spec  # noqa: E402


def _summary(counts: Sequence[int]) -> dict[str, Any]:
    if not counts:
        return {"n": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0, "min": 0, "total": 0}
    ordered = sorted(counts)
    n = len(ordered)

    def pct(p: float) -> int:
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return ordered[k]

    return {
        "n": n,
        "mean": round(sum(ordered) / n, 2),
        "p50": pct(50),
        "p95": pct(95),
        "max": ordered[-1],
        "min": ordered[0],
        "total": sum(ordered),
    }


def _print_distribution(title: str, counter: Counter, total: int) -> None:
    print(f"{title}:")
    for key, count in counter.most_common():
        pct = round(100.0 * count / total, 1) if total else 0.0
        print(f"  {key:<22} {count:>5}  ({pct}%)")


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PAT-ER synthetic dataset.")
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL files to summarize.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write the summary JSON.")
    args = parser.parse_args()

    records = load_records(args.paths)
    if not records:
        print("no records found", file=sys.stderr)
        sys.exit(2)

    total = len(records)
    by_family: Counter = Counter(r.get("task_family") for r in records)
    by_split: Counter = Counter(r.get("split") for r in records)
    family_split: dict[Any, Counter] = defaultdict(Counter)
    primitive_class: Counter = Counter()
    support_status: Counter = Counter()
    tool_intent: Counter = Counter()
    idk_action: Counter = Counter()
    hard_neg: Counter = Counter()
    source_primitive: dict[Any, Counter] = defaultdict(Counter)
    verifier_accept = 0
    schema_valid = 0

    template_splits: dict[str, set[str]] = defaultdict(set)
    evidence_counts: list[int] = []
    records_with_tools = 0
    records_with_calls = 0
    parsed_calls_ok = 0

    model_texts: list[str] = []

    for r in records:
        family_split[r.get("task_family")][r.get("split")] += 1
        labels = r.get("labels", {})
        primitive_class[labels.get("primitive_class")] += 1
        source_primitive[r.get("source")][labels.get("primitive_class")] += 1
        support_status[labels.get("support_status")] += 1
        tool_intent[labels.get("tool_intent")] += 1
        idk_action[labels.get("idk_action")] += 1
        verifier_accept += 1 if labels.get("verifier_accept") else 0
        schema_valid += 1 if labels.get("schema_validity") else 0
        if r.get("is_hard_negative"):
            hard_neg[r.get("hard_negative_type")] += 1
        tid, split = r.get("template_id"), r.get("split")
        if isinstance(tid, str) and isinstance(split, str):
            template_splits[tid].add(split)
        evidence_counts.append(len(r.get("evidence") or []))
        if r.get("tools"):
            records_with_tools += 1
        mt = r.get("model_text")
        if isinstance(mt, str):
            model_texts.append(mt)
            if "<tool_call>" in mt:
                records_with_calls += 1
                try:
                    calls = parse_hermes_tool_calls(mt)
                    if calls and all(c.name and isinstance(c.arguments, dict) for c in calls):
                        parsed_calls_ok += 1
                except Exception:
                    pass

    # Reference-tokenizer token statistics over model_text.
    token_summary = {"n": 0}
    atomic_rate = 1.0
    if model_texts:
        tokenizer = build_reference_tokenizer(extra_texts=model_texts)
        token_counts = [len(tokenizer.encode(mt, add_special_tokens=False)) for mt in model_texts]
        token_summary = _summary(token_counts)
        spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
        unk = tokenizer.token_to_id[tokenizer.unk_token]
        atomic = 0
        controls = spec.special_tokens
        for token in controls:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1 and ids[0] != unk:
                atomic += 1
        atomic_rate = round(atomic / len(controls), 4)

    leakage = sorted(t for t, s in template_splits.items() if len(s) > 1)

    # ---- print ----
    print("PAT-ER synthetic dataset summary")
    print(f"records: {total}  families: {len(by_family)}  templates: {len(template_splits)}")
    print(f"splits: {dict(by_split)}")
    print()
    print("per family (records | train/val/test):")
    for family in sorted(by_family):
        s = family_split[family]
        print(f"  {family:<32} {by_family[family]:>4} | {s['train']:>4}/{s['val']:>3}/{s['test']:>3}")
    print()
    _print_distribution("primitive_class", primitive_class, total)
    _print_distribution("support_status", support_status, total)
    _print_distribution("tool_intent", tool_intent, total)
    _print_distribution("idk_action", idk_action, total)
    print()
    if len(source_primitive) > 1:
        all_prims = sorted({p for c in source_primitive.values() for p in c})
        print("primitive_class by source:")
        for src in sorted(source_primitive, key=lambda s: str(s)):
            cells = " ".join(f"{p}={source_primitive[src][p]}" for p in all_prims if source_primitive[src][p])
            print(f"  {src}: {cells}")
        print()
    print(f"hard negatives: {sum(hard_neg.values())}")
    for kind, count in hard_neg.most_common():
        print(f"  {kind:<24} {count:>4}")
    print()
    print(f"verifier_accept=true: {verifier_accept}/{total}  schema_validity=true: {schema_valid}/{total}")
    ev = _summary(evidence_counts)
    print(f"evidence items per record: mean={ev['mean']} p95={ev['p95']} max={ev['max']} (records with >=1: {sum(1 for c in evidence_counts if c)})")
    call_rate = round(parsed_calls_ok / records_with_calls, 4) if records_with_calls else 1.0
    print(f"tools: records_with_tools={records_with_tools}  records_with_tool_calls={records_with_calls}  hermes_parse_rate={parsed_calls_ok}/{records_with_calls} ({call_rate})")
    print()
    print(f"reference-tokenizer model_text tokens: mean={token_summary['mean']} p50={token_summary.get('p50')} "
          f"p95={token_summary.get('p95')} max={token_summary.get('max')} total={token_summary.get('total')}")
    print(f"control-token atomicity (reference tokenizer): {atomic_rate}")
    print(f"template/split holdout integrity: {'OK (no leakage)' if not leakage else f'LEAKAGE in {leakage}'}")

    if args.json_out:
        summary = {
            "records": total,
            "families": dict(by_family),
            "splits": dict(by_split),
            "per_family_splits": {f: dict(s) for f, s in family_split.items()},
            "primitive_class": dict(primitive_class),
            "support_status": dict(support_status),
            "tool_intent": dict(tool_intent),
            "idk_action": dict(idk_action),
            "hard_negatives": dict(hard_neg),
            "verifier_accept_true": verifier_accept,
            "schema_validity_true": schema_valid,
            "evidence_per_record": ev,
            "records_with_tools": records_with_tools,
            "records_with_tool_calls": records_with_calls,
            "hermes_parse_ok": parsed_calls_ok,
            "model_text_tokens": token_summary,
            "control_atomicity": atomic_rate,
            "leakage": leakage,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote summary: {args.json_out}")


if __name__ == "__main__":
    main()
