#!/usr/bin/env python3
"""External-origin audit (FOLIO / ProofWriter) -- the Step-1 diagnostic for the
external-coverage gate.

For external-derived records only, reports per source dataset:
  - argument-span ambiguity (occurrences of the gold span text in model_text);
  - argument-span width distribution (subword tokens);
  - primitive_class and support_status distribution;
  - identifiability by width bucket.

Real records are read as-is; nothing is modified. CPU-only, offline.

Usage:
    python3 scripts/audit_external_origin.py \
        --dataset artifacts/datasets/external/folio_pater.jsonl \
                  artifacts/datasets/external/proofwriter_pater.jsonl \
        --tokenizer artifacts/tokenizers/qwen_pater_extended
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_lm_aux as T  # noqa: E402
import pater_hf_tokenizer as H  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizers" / "qwen_pater_extended"
RENDER_MODE = "input_only"  # the model is trained input-only; audit the same text


def _width_bucket(w: int) -> str:
    return "w1" if w <= 1 else "w2-4" if w <= 4 else "w5-15" if w <= 15 else "w16+"


def _source(rec: dict) -> str:
    return (rec.get("provenance") or {}).get("source_dataset") or rec.get("source") or "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="External-origin (FOLIO/ProofWriter) audit.")
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "external_origin_audit.json")
    args = p.parse_args()

    tok = H.load_pater_hf_tokenizer(args.tokenizer)
    records: list[dict] = []
    for f in args.dataset:
        records += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]

    # per-span rows; occurrences/width measured on the input-only render (what the
    # model actually sees -- the stored model_text also embeds the event_graph copy
    # of every span, which would spuriously inflate ambiguity to ~100%).
    rows: list[dict] = []
    for rec in records:
        src = _source(rec)
        rendered = T.render_for_mode(rec, RENDER_MODE)
        enc = T.encode_for_targets(tok, rec, 1024, RENDER_MODE)
        use_offsets = getattr(enc, "offsets", None) is not None
        for ev in (rec.get("event_graph") or [])[:1]:
            for arg in ev.get("arguments") or []:
                span = arg.get("span")
                if not isinstance(span, str) or not span:
                    continue
                if use_offsets:
                    hit = H.locate_text_span(enc.text, enc.offsets, span)
                else:
                    sub = tuple(tok.encode(span, add_special_tokens=False))
                    hit = T._find_subseq(enc.ids, sub) if sub else None
                width = (hit[1] - hit[0] + 1) if hit else 0
                occ = rendered.count(span)
                rows.append({"source": src, "primitive": rec["labels"]["primitive_class"],
                             "width": width, "width_bucket": _width_bucket(width) if hit else "unlocated",
                             "occ": occ, "ambiguous": occ > 1, "located": hit is not None})

    def block(rs: list[dict], key: str) -> list[dict]:
        by = defaultdict(lambda: {"spans": 0, "ambiguous": 0, "located": 0})
        for r in rs:
            b = by[r[key]]
            b["spans"] += 1
            b["ambiguous"] += int(r["ambiguous"])
            b["located"] += int(r["located"])
        out = [{key: k, "spans": b["spans"], "ambiguous_rate": round(b["ambiguous"] / b["spans"], 3),
                "located_rate": round(b["located"] / b["spans"], 3)} for k, b in by.items()]
        return sorted(out, key=lambda d: (-d["ambiguous_rate"], -d["spans"]))

    report: dict[str, Any] = {"records": len(records), "spans": len(rows), "by_source": {}}
    for src in sorted({r["source"] for r in rows}):
        srows = [r for r in rows if r["source"] == src]
        srecs = [r for r in records if _source(r) == src]
        amb = sum(r["ambiguous"] for r in srows)
        report["by_source"][src] = {
            "records": len(srecs), "spans": len(srows),
            "ambiguous_rate": round(amb / len(srows), 3) if srows else 0.0,
            "uniquely_identifiable": len(srows) - amb,
            "mean_width": round(sum(r["width"] for r in srows) / len(srows), 2) if srows else 0.0,
            "by_width_bucket": block(srows, "width_bucket"),
            "by_primitive": block(srows, "primitive"),
            "primitive_distribution": dict(Counter(r["labels"]["primitive_class"] for r in srecs)),
            "support_distribution": dict(Counter(r["labels"]["support_status"] for r in srecs)),
        }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"external-origin audit  records={report['records']}  spans={report['spans']}")
    for src, b in report["by_source"].items():
        print(f"\n=== {src}: {b['records']} records, {b['spans']} spans, "
              f"ambiguous={b['ambiguous_rate']:.3f}, mean_width={b['mean_width']} ===")
        print("  by width:   " + "  ".join(f"{w['width_bucket']}={w['spans']}({w['ambiguous_rate']:.2f}amb)"
                                           for w in b["by_width_bucket"]))
        print("  by primitive: " + "  ".join(f"{w['primitive']}={w['spans']}({w['ambiguous_rate']:.2f})"
                                              for w in b["by_primitive"]))
        print(f"  primitive_dist: {b['primitive_distribution']}")
        print(f"  support_dist:   {b['support_distribution']}")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
