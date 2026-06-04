#!/usr/bin/env python3
"""Audit argument-span identifiability in a PAT-ER dataset.

The 450M span gate showed the pointer fails on SHORT SYNTHETIC argument spans.
The hypothesis: the gold span text occurs multiple times in the input the model
sees, so a content-based pointer is asked to pick one occurrence with no local
evidence -- a non-identifiable label, not a model-capacity problem.

For every argument-span label this reports:
  - gold span text;
  - number of exact occurrences in the rendered (input-only) text the pointer sees;
  - number of exact occurrences in model_text;
  - whether the gold occurrence is uniquely identifiable (exactly one occurrence);
  - origin (synthetic / external), task_family, template_id, primitive_class;
  - subword width and located start/end token positions.

Then it summarises the ambiguity rate (occurrences > 1) by origin, family,
template, width bucket, and primitive class.

CPU-only, offline. Reads JSONL, writes JSON + console. No model, no training.

Usage:
    python3 scripts/audit_span_ambiguity.py \
        --dataset artifacts/datasets/mixed/pater_mixed_25_75.jsonl \
        --tokenizer artifacts/tokenizers/qwen_pater_extended
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

import train_lm_aux as T  # noqa: E402
import pater_hf_tokenizer as H  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizers" / "qwen_pater_extended"


def _width_bucket(width: int) -> str:
    if width <= 1:
        return "w1"
    if width <= 4:
        return "w2-4"
    if width <= 15:
        return "w5-15"
    return "w16+"


def _rate_block(rows: list[dict], key: str, top: int | None = None) -> list[dict]:
    by = defaultdict(lambda: {"spans": 0, "ambiguous": 0, "located": 0})
    for r in rows:
        b = by[r[key]]
        b["spans"] += 1
        b["ambiguous"] += int(r["occ_render"] > 1)
        b["located"] += int(r["located"])
    out = []
    for k, b in by.items():
        out.append({key: k, "spans": b["spans"], "ambiguous": b["ambiguous"],
                    "ambiguous_rate": round(b["ambiguous"] / b["spans"], 3) if b["spans"] else 0.0,
                    "located": b["located"]})
    out.sort(key=lambda d: (-d["ambiguous_rate"], -d["spans"]))
    return out[:top] if top else out


def main() -> None:
    p = argparse.ArgumentParser(description="Argument-span identifiability audit (CPU-only, offline).")
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--render-mode", choices=list(T.RENDER_MODES), default="input_only")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "span_ambiguity_audit.json")
    p.add_argument("--rows-out", type=Path, default=None, help="Optional JSONL dump of every per-span row.")
    args = p.parse_args()

    split = None if args.split == "all" else args.split
    records = T.load_record_files(args.dataset, split=split)
    if not records:
        print("no records", file=sys.stderr)
        sys.exit(2)
    tok = H.load_pater_hf_tokenizer(args.tokenizer)

    rows: list[dict] = []
    for record in records:
        origin = (record.get("provenance") or {}).get("mix_origin") or "unknown"
        family = record.get("task_family") or "?"
        template = record.get("template_id") or "?"
        primitive = (record.get("labels") or {}).get("primitive_class") or "?"
        rendered = T.render_for_mode(record, args.render_mode)
        model_text = record.get("model_text") if isinstance(record.get("model_text"), str) else ""
        enc = T.encode_for_targets(tok, record, args.max_len, args.render_mode)
        use_offsets = getattr(enc, "offsets", None) is not None
        for event in (record.get("event_graph") or [])[:1]:
            for arg in (event.get("arguments") or []):
                span = arg.get("span")
                if not isinstance(span, str) or not span:
                    continue
                if use_offsets:
                    hit = H.locate_text_span(enc.text, enc.offsets, span)
                else:
                    sub = tuple(tok.encode(span, add_special_tokens=False))
                    hit = T._find_subseq(enc.ids, sub) if sub else None
                width = (hit[1] - hit[0] + 1) if hit else 0
                occ_render = rendered.count(span)
                rows.append({
                    "id": record.get("id"), "origin": origin, "family": family,
                    "template": template, "primitive": primitive, "span": span,
                    "occ_render": occ_render, "occ_model_text": model_text.count(span) if model_text else 0,
                    "uniquely_identifiable": occ_render == 1,
                    "located": hit is not None, "width": width,
                    "width_bucket": _width_bucket(width) if hit else "unlocated",
                    "start": hit[0] if hit else None, "end": hit[1] if hit else None,
                })

    n = len(rows)
    located = sum(r["located"] for r in rows)
    ambiguous = sum(r["occ_render"] > 1 for r in rows)
    unique = sum(r["uniquely_identifiable"] for r in rows)
    summary = {
        "dataset": [str(x) for x in args.dataset], "split": args.split, "render_mode": args.render_mode,
        "spans": n, "located": located, "uniquely_identifiable": unique,
        "ambiguous_repeated": ambiguous,
        "ambiguous_rate": round(ambiguous / n, 4) if n else 0.0,
        "by_origin": _rate_block(rows, "origin"),
        "by_width_bucket": _rate_block(rows, "width_bucket"),
        "by_primitive": _rate_block(rows, "primitive"),
        "by_family": _rate_block(rows, "family"),
        "by_template_worst": _rate_block(rows, "template", top=15),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.rows_out:
        args.rows_out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    print(f"span ambiguity audit  spans={n}  located={located}  "
          f"uniquely_identifiable={unique} ({100*unique/n:.0f}%)  "
          f"ambiguous(>1 occ)={ambiguous} ({100*ambiguous/n:.0f}%)")
    for title, key in (("by origin", "by_origin"), ("by width", "by_width_bucket"),
                       ("by primitive", "by_primitive"), ("by family", "by_family")):
        print(f"\n{title}:")
        for d in summary[key]:
            kname = [v for k, v in d.items() if k not in ("spans", "ambiguous", "ambiguous_rate", "located")][0]
            print(f"  {str(kname):<22} spans={d['spans']:<5} ambiguous={d['ambiguous_rate']:.3f} "
                  f"located={d['located']}")
    print("\nworst templates (by ambiguity rate, min a few spans):")
    for d in summary["by_template_worst"]:
        print(f"  {d['template']:<40} spans={d['spans']:<4} ambiguous={d['ambiguous_rate']:.3f}")
    print(f"\nwrote {args.json_out}" + (f"\nwrote {args.rows_out}" if args.rows_out else ""))


if __name__ == "__main__":
    main()
