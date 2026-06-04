#!/usr/bin/env python3
"""Set-valued span gate aggregation: 450M + multi-positive (any-occurrence) span
targets vs the prior single-gold 450M baseline, across seeds.

Any-occurrence is the headline metric (repeated logical atoms are co-referent
mentions of the same argument); strict first-occurrence is a diagnostic.

Pass conditions (user-specified):
  - any-occurrence arg_span improves materially over baseline;
  - argument_start / argument_end any-occurrence improve materially;
  - primitive / role_to_primitive macro-F1 do not regress;
  - evidence_pointer remains near 1.0.

CPU-only, offline. Reads JSON, writes JSON + Markdown.

Usage:
    python3 scripts/report_450m_set_gate.py \
        --set artifacts/reports/qwen_450m_set_eval_s{0,1,2}.json \
        --baseline artifacts/reports/qwen_450m_base_eval_s{0,1,2}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# label -> (metrics key, field). Any-occurrence first (headline), then strict, bridge.
METRICS: list[tuple[str, str, str]] = [
    ("arg_span_joint_any", "arg_span_joint_any", "accuracy"),
    ("arg_span_any", "arg_span_any", "accuracy"),
    ("argument_start_any", "argument_start_any", "accuracy"),
    ("argument_end_any", "argument_end_any", "accuracy"),
    ("arg_span_joint_strict", "arg_span_joint", "accuracy"),
    ("arg_span_exact_strict", "arg_span_exact", "accuracy"),
    ("argument_start_strict", "argument_start", "accuracy"),
    ("argument_end_strict", "argument_end", "accuracy"),
    ("primitive_macro_f1", "primitive", "macro_f1"),
    ("role_to_primitive_macro_f1", "role_to_primitive", "macro_f1"),
    ("evidence_pointer", "evidence_pointer", "accuracy"),
    ("proto_role_f1", "proto_role", "micro_f1"),
    ("arg_role_f1", "arg_role", "micro_f1"),
    ("predicate_event", "predicate_event", "accuracy"),
    ("event_arg", "event_arg", "accuracy"),
]

MATERIAL = 0.03
BRIDGE_TOL = 0.03


def _val(rep: dict[str, Any], mkey: str, field: str) -> float | None:
    node = rep.get("metrics", {}).get(mkey)
    if not isinstance(node, dict) or field not in node:
        return None
    return float(node[field])


def _ms(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n if n > 1 else 0.0
    return m, v ** 0.5


def main() -> None:
    p = argparse.ArgumentParser(description="450M set-valued span gate aggregation.")
    p.add_argument("--set", dest="set_", type=Path, nargs="+", required=True, help="Set-valued per-seed eval JSONs.")
    p.add_argument("--baseline", type=Path, nargs="+", required=True, help="Single-gold 450M per-seed eval JSONs.")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_set_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_set_gate.md")
    args = p.parse_args()

    sset = [json.loads(Path(s).read_text()) for s in args.set_]
    base = [json.loads(Path(b).read_text()) for b in args.baseline]

    agg: dict[str, dict[str, Any]] = {}
    for label, mkey, field in METRICS:
        sv = [v for v in (_val(r, mkey, field) for r in sset) if v is not None]
        bv = [v for v in (_val(r, mkey, field) for r in base) if v is not None]
        sm, ss = _ms(sv)
        bm, bs = _ms(bv)
        agg[label] = {"set_mean": sm, "set_std": ss, "base_mean": bm, "base_std": bs,
                      "delta": (sm - bm) if (sv and bv) else None}

    def d(label: str) -> float | None:
        return agg[label]["delta"]

    cond_span = (d("arg_span_joint_any") or -1) >= MATERIAL
    cond_start = (d("argument_start_any") or -1) >= MATERIAL
    cond_end = (d("argument_end_any") or -1) >= MATERIAL
    prim_d, r2p_d = d("primitive_macro_f1"), d("role_to_primitive_macro_f1")
    cond_bridge = (prim_d is None or prim_d >= -BRIDGE_TOL) and (r2p_d is None or r2p_d >= -BRIDGE_TOL)
    cond_evi = agg["evidence_pointer"]["set_mean"] >= 0.98
    gate_pass = bool(cond_span and cond_start and cond_end and cond_bridge and cond_evi)

    verdict = {
        "arg_span_any_material": cond_span, "argument_start_any_material": cond_start,
        "argument_end_any_material": cond_end, "bridge_held": cond_bridge,
        "evidence_near_one": cond_evi, "gate_pass": gate_pass,
        "thresholds": {"material": MATERIAL, "bridge_tol": BRIDGE_TOL},
    }
    out = {"set_seeds": [str(s) for s in args.set_], "baseline_seeds": [str(b) for b in args.baseline],
           "aggregate": agg, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [f"# 450M set-valued span gate ({len(sset)} seeds)", "",
             "Multi-positive (any-occurrence) span targets vs single-gold 450M baseline.", "",
             "| metric | set-valued mean±std | baseline mean±std | Δ |", "|---|---|---|---|"]
    for label, _m, _f in METRICS:
        a = agg[label]
        dv = "—" if a["delta"] is None else f"{a['delta']:+.3f}"
        lines.append(f"| {label} | {a['set_mean']:.3f} ± {a['set_std']:.3f} | "
                     f"{a['base_mean']:.3f} ± {a['base_std']:.3f} | {dv} |")
    lines += ["", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- arg_span_joint_any material (≥{MATERIAL}): {cond_span} (Δ={d('arg_span_joint_any')})",
              f"- argument_start_any material: {cond_start} (Δ={d('argument_start_any')})",
              f"- argument_end_any material: {cond_end} (Δ={d('argument_end_any')})",
              f"- bridge held: {cond_bridge} (primitive Δ={prim_d}, role_to_primitive Δ={r2p_d})",
              f"- evidence_pointer ≥0.98: {cond_evi} (mean {agg['evidence_pointer']['set_mean']:.3f})"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M set-valued span gate  set_seeds={len(sset)} baseline_seeds={len(base)}")
    for label, _m, _f in METRICS:
        a = agg[label]
        dv = "  n/a" if a["delta"] is None else f"{a['delta']:+.3f}"
        print(f"  {label:30s} set={a['set_mean']:.3f}±{a['set_std']:.3f}  base={a['base_mean']:.3f}  Δ={dv}")
    print(f"\narg_span_any material: {cond_span}  start_any: {cond_start}  end_any: {cond_end}  "
          f"bridge held: {cond_bridge}  evidence~1: {cond_evi}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
