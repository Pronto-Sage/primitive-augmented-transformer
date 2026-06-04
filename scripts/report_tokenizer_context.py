#!/usr/bin/env python3
"""Tokenizer + context report for the production-tokenizer migration.

Compares the word-level reference tokenizer with the PAT-ER-extended Qwen
tokenizer on a dataset, over the rendered *input-only* text (what the model
actually trains on):

- token length stats (mean / p50 / p95 / max);
- truncation rate at each context length (512 / 768 / 1024);
- control-token atomicity (98 special + 34 normal);
- Hermes tool-call parse rate after encode -> decode;
- span/evidence label migration (arg / predicate / evidence located, plus
  out-of-bounds and points-to-pad counts) -- the proof that subword tokenization
  preserves argument_start/end, event_token, and evidence-pointer labels.

CPU-only, offline (the extended tokenizer is loaded from a local dir). No model,
no training, no downloads.

Usage:
    python3 scripts/report_tokenizer_context.py \
        --dataset artifacts/datasets/mixed/pater_mixed_25_75.jsonl \
        --tokenizer artifacts/tokenizers/qwen_pater_extended --contexts 512,768,1024
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

from pat_er.serialization import parse_hermes_tool_calls  # noqa: E402
from pat_er.tokenizer_spec import build_pater_tokenizer_spec  # noqa: E402
import train_lm_aux as T  # noqa: E402
import pater_hf_tokenizer as H  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizers" / "qwen_pater_extended"


def _summary(counts: Sequence[int]) -> dict[str, Any]:
    if not counts:
        return {"n": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(counts)
    n = len(ordered)

    def pct(p: float) -> int:
        return ordered[max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))]

    return {"n": n, "mean": round(sum(ordered) / n, 2), "p50": pct(50), "p95": pct(95), "max": ordered[-1]}


def _atomicity(tokenizer: Any, tokens: Sequence[str]) -> dict[str, Any]:
    """Atomicity via the tokenizer's own encode (works for reference + HF)."""

    bad = []
    for tok in tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            bad.append(tok)
    total = len(tokens)
    return {"total": total, "atomic": total - len(bad), "rate": round((total - len(bad)) / total, 4) if total else 1.0,
            "failures": bad[:8]}


def _hermes_parse_rate(tokenizer: Any, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    calls = [r["model_text"] for r in records if isinstance(r.get("model_text"), str) and "<tool_call>" in r["model_text"]]
    ok = 0
    for text in calls:
        try:
            decoded = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), skip_special_tokens=False)
            parsed = parse_hermes_tool_calls(decoded)
        except Exception:
            continue
        if parsed and all(c.name and isinstance(c.arguments, dict) for c in parsed):
            ok += 1
    return {"records_with_calls": len(calls), "parsed": ok, "parse_rate": round(ok / len(calls), 4) if calls else 1.0}


def _length_block(tokenizer: Any, records: Sequence[dict[str, Any]], render_mode: str,
                  contexts: Sequence[int]) -> dict[str, Any]:
    counts = [len(tokenizer.encode(T.render_for_mode(r, render_mode), add_special_tokens=False)) for r in records]
    summary = _summary(counts)
    summary["truncation_rate"] = {str(c): round(sum(1 for n in counts if n > c) / len(counts), 4) for c in contexts}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer + context report (CPU-only, offline).")
    parser.add_argument("--dataset", type=Path, nargs="+", required=True, help="Dataset JSONL file(s).")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Extended HF tokenizer dir.")
    parser.add_argument("--contexts", type=str, default="512,768,1024", help="Comma-separated context lengths.")
    parser.add_argument("--render-mode", choices=list(T.RENDER_MODES), default="input_only",
                        help="Render mode for length/truncation/migration stats (training uses input_only).")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "tokenizer_context_report.json")
    args = parser.parse_args()

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    split = None if args.split == "all" else args.split
    records = T.load_record_files(args.dataset, split=split)
    if not records:
        print("no records", file=sys.stderr)
        sys.exit(2)

    spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
    reference = T.build_tokenizer(records)
    extended = H.load_pater_hf_tokenizer(args.tokenizer)
    # Migration stats need a config for register dims + a max context.
    from pat_er import PATERConfig
    dims = T.dims_from_config(PATERConfig())  # default register counts match all tiny configs
    max_ctx = max(contexts)

    report: dict[str, Any] = {"dataset": [str(p) for p in args.dataset], "records": len(records),
                              "render_mode": args.render_mode, "contexts": contexts, "tokenizers": {}}

    for name, tok in (("reference", reference), ("extended_qwen", extended)):
        block = {
            "vocab_size": tok.vocab_size,
            "lengths": _length_block(tok, records, args.render_mode, contexts),
            "atomicity": {"special": _atomicity(tok, spec.special_tokens), "normal": _atomicity(tok, spec.normal_tokens)},
            "hermes": _hermes_parse_rate(tok, records),
            "span_migration": T.span_label_stats(tok, records, max_ctx, dims, args.render_mode),
        }
        report["tokenizers"][name] = block

    # ---- print ----
    print(f"tokenizer + context report  records={len(records)}  render={args.render_mode}")
    print(f"dataset: {', '.join(p.name for p in args.dataset)}\n")
    for name in ("reference", "extended_qwen"):
        b = report["tokenizers"][name]
        ln = b["lengths"]
        sp, nm = b["atomicity"]["special"], b["atomicity"]["normal"]
        sm = b["span_migration"]
        print(f"=== {name}  vocab={b['vocab_size']} ===")
        print(f"  length: mean={ln['mean']} p50={ln['p50']} p95={ln['p95']} max={ln['max']}")
        print(f"  truncation: " + "  ".join(f"@{c}={ln['truncation_rate'][str(c)]}" for c in contexts))
        print(f"  atomicity: special {sp['atomic']}/{sp['total']} (rate={sp['rate']})  "
              f"normal {nm['atomic']}/{nm['total']} (rate={nm['rate']})")
        print(f"  hermes parse: {b['hermes']['parsed']}/{b['hermes']['records_with_calls']} "
              f"(rate={b['hermes']['parse_rate']})")
        print(f"  span migration: arg {sm['located']['arg']}/{sm['source']['arg']} (rate={sm['located_rate']['arg']}), "
              f"predicate {sm['located']['predicate']}/{sm['source']['predicate']} (rate={sm['located_rate']['predicate']}), "
              f"evidence {sm['located']['evidence']}/{sm['source']['evidence']} (rate={sm['located_rate']['evidence']})")
        print(f"  span bounds: out_of_bounds={sm['out_of_bounds']} point_to_pad={sm['point_to_pad']}")
        print()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.json_out}")

    ext = report["tokenizers"]["extended_qwen"]
    sm = ext["span_migration"]
    ok = (ext["atomicity"]["special"]["rate"] == 1.0 and ext["atomicity"]["normal"]["rate"] == 1.0
          and sm["out_of_bounds"] == 0 and sm["point_to_pad"] == 0)
    print("PASS: extended tokenizer atomic + span labels in-bounds/non-pad" if ok
          else "WARN: check atomicity / span-bounds above")


if __name__ == "__main__":
    main()
