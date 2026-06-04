#!/usr/bin/env python3
"""Aggregate the 450M span-recovery gate across seeds and compare to tiny-subword.

Reads the per-seed val eval JSONs produced by eval_lm_aux.py for the 450M run,
computes mean/std per metric, and diffs against the tiny extended-Qwen baseline
(same tokenizer/context, 64-dim model). Emits the gate verdict:

  - span pointer recovers materially over tiny subword;
  - primitive / role-to-primitive bridge stays near or above tiny;
  - no tokenizer/context regression (evidence_pointer / event_token stay ~1.0).

CPU-only, offline; reads JSON, writes JSON + Markdown. No model, no training.

Usage:
    python3 scripts/report_450m_gate.py \
        --seeds artifacts/reports/qwen_450m_eval_s0.json ... \
        --baseline artifacts/reports/qwen_tiny_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# metric label -> (json key under "metrics", field) ; "lm_loss" is top-level.
METRICS: list[tuple[str, str, str]] = [
    ("primitive_acc", "primitive", "accuracy"),
    ("primitive_macro_f1", "primitive", "macro_f1"),
    ("role_to_primitive_acc", "role_to_primitive", "accuracy"),
    ("role_to_primitive_macro_f1", "role_to_primitive", "macro_f1"),
    ("predicate_event", "predicate_event", "accuracy"),
    ("argument_start", "argument_start", "accuracy"),
    ("argument_end", "argument_end", "accuracy"),
    ("arg_span_exact", "arg_span_exact", "accuracy"),
    ("event_token", "event_token", "accuracy"),
    ("event_arg", "event_arg", "accuracy"),
    ("evidence_pointer", "evidence_pointer", "accuracy"),
    ("proto_role_f1", "proto_role", "micro_f1"),
    ("arg_role_f1", "arg_role", "micro_f1"),
    ("support", "support", "accuracy"),
    ("idk", "idk", "accuracy"),
    ("verifier", "verifier", "accuracy"),
    ("tool_intent", "tool_intent", "accuracy"),
    ("schema", "schema", "accuracy"),
]

# Gate buckets.
SPAN_KEYS = ["argument_start", "argument_end", "arg_span_exact"]
BRIDGE_KEYS = ["primitive_macro_f1", "role_to_primitive_macro_f1"]
NO_REGRESS_KEYS = ["evidence_pointer", "event_token", "event_arg"]


def _val(report: dict[str, Any], mkey: str, field: str) -> float | None:
    if mkey == "lm_loss":
        return float(report.get("lm_loss")) if report.get("lm_loss") is not None else None
    node = report.get("metrics", {}).get(mkey)
    if not isinstance(node, dict) or field not in node:
        return None
    return float(node[field])


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n if n > 1 else 0.0
    return m, var ** 0.5


def main() -> None:
    p = argparse.ArgumentParser(description="450M span-recovery gate aggregation (CPU-only).")
    p.add_argument("--seeds", type=Path, nargs="+", required=True, help="Per-seed 450M eval JSONs.")
    p.add_argument("--baseline", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_tiny_eval.json",
                   help="Tiny extended-Qwen eval JSON (same tokenizer/context).")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_gate.md")
    p.add_argument("--material-delta", type=float, default=0.05,
                   help="Min mean improvement to call a span metric 'materially recovered'.")
    args = p.parse_args()

    seed_reports = [json.loads(Path(s).read_text()) for s in args.seeds]
    base = json.loads(args.baseline.read_text())

    rows: list[dict[str, Any]] = []
    agg: dict[str, dict[str, float]] = {}
    for label, mkey, field in [("lm_loss", "lm_loss", "")] + METRICS:
        vals = [v for v in (_val(r, mkey, field) for r in seed_reports) if v is not None]
        mean, std = _mean_std(vals)
        bval = _val(base, mkey, field)
        delta = (mean - bval) if (bval is not None and vals) else None
        agg[label] = {"mean": mean, "std": std, "n": len(vals),
                      "baseline_tiny": bval, "delta_vs_tiny": delta}
        rows.append({"metric": label, "mean": mean, "std": std,
                     "baseline_tiny": bval, "delta": delta})

    def bucket_ok(keys: list[str], require_recover: bool) -> tuple[bool, list[str]]:
        notes = []
        ok = True
        for k in keys:
            d = agg[k]["delta_vs_tiny"]
            if d is None:
                continue
            if require_recover:
                passed = d >= args.material_delta
            else:
                passed = d >= -0.02  # "near or above": allow tiny slack
            ok = ok and passed
            notes.append(f"{k} Δ={d:+.3f}{' OK' if passed else ' MISS'}")
        return ok, notes

    span_ok, span_notes = bucket_ok(SPAN_KEYS, require_recover=True)
    bridge_ok, bridge_notes = bucket_ok(BRIDGE_KEYS, require_recover=False)
    noregress_ok, noregress_notes = bucket_ok(NO_REGRESS_KEYS, require_recover=False)
    gate_pass = bool(span_ok and bridge_ok and noregress_ok)

    verdict = {
        "span_recovers_materially": span_ok,
        "bridge_near_or_above_tiny": bridge_ok,
        "no_tokenizer_context_regression": noregress_ok,
        "gate_pass": gate_pass,
        "material_delta_threshold": args.material_delta,
        "notes": {"span": span_notes, "bridge": bridge_notes, "no_regression": noregress_notes},
    }
    out = {"seeds": [str(s) for s in args.seeds], "n_seeds": len(seed_reports),
           "baseline": str(args.baseline), "aggregate": agg, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- Markdown ----
    lines = [f"# 450M span-recovery gate ({len(seed_reports)} seeds)", "",
             f"Baseline: tiny extended-Qwen (`{args.baseline.name}`), same tokenizer + context 1024.", "",
             "| metric | 450M mean±std | tiny-subword | Δ vs tiny |", "|---|---|---|---|"]
    for label, _mk, _f in [("lm_loss", "", "")] + [(m[0], m[1], m[2]) for m in METRICS]:
        a = agg[label]
        b = "—" if a["baseline_tiny"] is None else f"{a['baseline_tiny']:.3f}"
        d = "—" if a["delta_vs_tiny"] is None else f"{a['delta_vs_tiny']:+.3f}"
        lines.append(f"| {label} | {a['mean']:.3f} ± {a['std']:.3f} | {b} | {d} |")
    lines += ["", "## Verdict", "",
              f"- **Gate pass: {gate_pass}**",
              f"- span recovers materially (Δ≥{args.material_delta}): {span_ok} — {'; '.join(span_notes)}",
              f"- bridge near/above tiny: {bridge_ok} — {'; '.join(bridge_notes)}",
              f"- no tokenizer/context regression: {noregress_ok} — {'; '.join(noregress_notes)}"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- console ----
    print(f"450M span-recovery gate  seeds={len(seed_reports)}  baseline={args.baseline.name}")
    for label, _mk, _f in [("lm_loss", "", "")] + [(m[0], m[1], m[2]) for m in METRICS]:
        a = agg[label]
        b = "  n/a " if a["baseline_tiny"] is None else f"{a['baseline_tiny']:.3f}"
        d = "  n/a " if a["delta_vs_tiny"] is None else f"{a['delta_vs_tiny']:+.3f}"
        print(f"  {label:28s} {a['mean']:.3f} ± {a['std']:.3f}   tiny={b}  Δ={d}")
    print(f"\nspan recovers materially: {span_ok}  ({'; '.join(span_notes)})")
    print(f"bridge near/above tiny:   {bridge_ok}  ({'; '.join(bridge_notes)})")
    print(f"no tok/ctx regression:    {noregress_ok}  ({'; '.join(noregress_notes)})")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
