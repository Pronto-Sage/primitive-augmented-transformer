#!/usr/bin/env python3
"""Mix external (real logic) and synthetic (balanced primitive/bridge) PAT-ER data.

The repaired external data is a usable *component*, not standalone training data:
it carries real entailment signal for the event-role/span heads but covers only
5 primitives with no hard negatives. Synthetic data fills the 14-class primitive
coverage, hard negatives, and tool/IDK balance. This mixer combines them at a
controllable external fraction, optionally balancing a label, while preserving
provenance and the family/template holdout.

Sampling is per output split (80/10/10) and within each split's records, so a
record never crosses splits (no leakage). Oversampled records get fresh unique
ids (original id kept in provenance); the output filename is used as an id prefix
so several mixes can be validated together.

CPU-only, no downloads, no training. Output is git-ignored.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SYNTHETIC_SOURCES = {"synthetic_primitive"}


def _load(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _balance_key(args: argparse.Namespace) -> str | None:
    if args.balance_primitive:
        return "primitive_class"
    if args.balance_support:
        return "support_status"
    if args.balance_tool_intent:
        return "tool_intent"
    return None


def balanced_sample(pool: list[dict[str, Any]], quota: int, key: str, rng: random.Random) -> list[dict[str, Any]]:
    """Round-robin over the label's classes (oversampling rare classes by wrap)."""

    if quota <= 0 or not pool:
        return []
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for record in pool:
        buckets.setdefault(record["labels"].get(key), []).append(record)
    classes = sorted(buckets, key=lambda c: str(c))
    for cls in classes:
        rng.shuffle(buckets[cls])
    cursors = {cls: 0 for cls in classes}
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < quota:
        cls = classes[i % len(classes)]
        i += 1
        bucket = buckets[cls]
        if cursors[cls] >= len(bucket):
            rng.shuffle(bucket)
            cursors[cls] = 0
        out.append(bucket[cursors[cls]])
        cursors[cls] += 1
    return out


def plain_sample(pool: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    if quota <= 0 or not pool:
        return []
    out: list[dict[str, Any]] = []
    while len(out) < quota:
        order = list(range(len(pool)))
        rng.shuffle(order)
        out.extend(pool[i] for i in order[: quota - len(out)])
    return out


DERIVED_SOURCES = {"synthetic_derived_from_external"}


def _origin(record: dict[str, Any]) -> str:
    src = record.get("source")
    if src in DERIVED_SOURCES:
        return "synthetic_ext"  # external-style augmentation, a distinct bucket
    return "synthetic" if src in SYNTHETIC_SOURCES else "external"


def mix(external: list[dict[str, Any]], synthetic: list[dict[str, Any]], *, total: int, external_fraction: float,
        balance_key: str | None, rng: random.Random, id_prefix: str,
        derived: list[dict[str, Any]] | None = None, derived_fraction: float = 0.0) -> list[dict[str, Any]]:
    derived = derived or []
    out: list[dict[str, Any]] = []
    for split, split_frac in SPLIT_FRACTIONS.items():
        total_s = round(total * split_frac)
        n_ext = round(external_fraction * total_s)
        n_der = round(derived_fraction * total_s)
        n_syn = total_s - n_ext - n_der
        ext_pool = [r for r in external if r.get("split") == split]
        syn_pool = [r for r in synthetic if r.get("split") == split]
        der_pool = [r for r in derived if r.get("split") == split]
        if balance_key:
            picked = (balanced_sample(ext_pool, n_ext, balance_key, rng)
                      + balanced_sample(der_pool, n_der, balance_key, rng)
                      + balanced_sample(syn_pool, n_syn, balance_key, rng))
        else:
            picked = (plain_sample(ext_pool, n_ext, rng) + plain_sample(der_pool, n_der, rng)
                      + plain_sample(syn_pool, n_syn, rng))
        out.extend(picked)
    rng.shuffle(out)

    mixed: list[dict[str, Any]] = []
    for index, record in enumerate(out):
        clone = copy.deepcopy(record)
        provenance = dict(clone.get("provenance") or {})
        provenance["mix_origin"] = _origin(record)
        provenance["original_id"] = clone.get("id")
        provenance["original_source"] = clone.get("source")
        clone["provenance"] = provenance
        clone["id"] = f"{id_prefix}_{index:06d}"
        mixed.append(clone)
    return mixed


def manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_origin = Counter(r["provenance"]["mix_origin"] for r in records)
    by_split = Counter(r["split"] for r in records)
    prim = Counter(r["labels"]["primitive_class"] for r in records)
    # external vs synthetic contribution per primitive
    contrib: dict[str, dict[str, int]] = {}
    for r in records:
        cls = r["labels"]["primitive_class"]
        origin = r["provenance"]["mix_origin"]
        contrib.setdefault(cls, {})[origin] = contrib.setdefault(cls, {}).get(origin, 0) + 1
    return {
        "records": len(records),
        "by_origin": dict(by_origin),
        "by_split": dict(by_split),
        "primitive_class": dict(prim.most_common()),
        "support_status": dict(Counter(r["labels"]["support_status"] for r in records).most_common()),
        "idk_action": dict(Counter(r["labels"]["idk_action"] for r in records).most_common()),
        "tool_intent": dict(Counter(r["labels"]["tool_intent"] for r in records).most_common()),
        "hard_negatives": sum(1 for r in records if r.get("is_hard_negative")),
        "primitive_by_source": {k: contrib[k] for k in sorted(contrib)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix external + synthetic PAT-ER datasets (CPU-only).")
    parser.add_argument("--external", type=Path, nargs="+", required=True, help="External JSONL file(s).")
    parser.add_argument("--synthetic", type=Path, nargs="+", required=True, help="Synthetic JSONL file(s).")
    parser.add_argument("--derived", type=Path, nargs="*", default=None,
                        help="External-style augmentation JSONL file(s) (synthetic_ext origin bucket).")
    parser.add_argument("--derived-fraction", type=float, default=0.0, help="Fraction of records drawn from derived.")
    parser.add_argument("--output", type=Path, required=True, help="Output mixed JSONL path.")
    parser.add_argument("--external-fraction", type=float, default=0.5, help="Fraction of records drawn from external.")
    parser.add_argument("--total", type=int, default=2000, help="Target total records.")
    parser.add_argument("--balance-primitive", action="store_true", help="Balance each source over primitive_class.")
    parser.add_argument("--balance-support", action="store_true", help="Balance each source over support_status.")
    parser.add_argument("--balance-tool-intent", action="store_true", help="Balance each source over tool_intent.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    external = _load(args.external)
    synthetic = _load(args.synthetic)
    derived = _load(args.derived) if args.derived else []
    balance_key = _balance_key(args)
    id_prefix = args.output.stem

    records = mix(external, synthetic, total=args.total, external_fraction=args.external_fraction,
                  balance_key=balance_key, rng=rng, id_prefix=id_prefix,
                  derived=derived, derived_fraction=args.derived_fraction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    info = manifest(records)
    (args.output.parent / f"{id_prefix}_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"mixed {info['records']} records -> {args.output} "
          f"(external_fraction={args.external_fraction}, balance={balance_key or 'none'})")
    print(f"by_origin: {info['by_origin']}  by_split: {info['by_split']}")
    print(f"primitive_class: {info['primitive_class']}")
    print(f"hard_negatives: {info['hard_negatives']}  tool_intent: {info['tool_intent']}")


if __name__ == "__main__":
    main()
