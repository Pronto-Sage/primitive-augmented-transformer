#!/usr/bin/env python3
"""Span-END repair gate aggregation: 450M + joint span loss + span sampler vs the
prior 450M baseline, across seeds. Reports overall span/bridge metrics with
mean/std and the per-(origin/width/subword) span breakdown, and the verdict.

Pass conditions (user-specified):
  - argument_end improves materially over baseline 0.576;
  - arg_span_exact improves materially over baseline 0.477;
  - primitive / role_to_primitive macro-F1 stay near baseline (no material drop);
  - evidence_pointer remains ~1.0.

CPU-only, offline; reads JSON, writes JSON + Markdown.

Usage:
    python3 scripts/report_450m_span_gate.py \
        --span artifacts/reports/qwen_450m_span_eval_s{0,1,2}.json \
        --baseline artifacts/reports/qwen_450m_eval_s{0,1,2}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

METRICS: list[tuple[str, str, str]] = [
    ("primitive_macro_f1", "primitive", "macro_f1"),
    ("role_to_primitive_macro_f1", "role_to_primitive", "macro_f1"),
    ("argument_start", "argument_start", "accuracy"),
    ("argument_end", "argument_end", "accuracy"),
    ("arg_span_exact", "arg_span_exact", "accuracy"),
    ("arg_span_joint", "arg_span_joint", "accuracy"),
    ("evidence_pointer", "evidence_pointer", "accuracy"),
    ("proto_role_f1", "proto_role", "micro_f1"),
    ("arg_role_f1", "arg_role", "micro_f1"),
    ("predicate_event", "predicate_event", "accuracy"),
    ("event_arg", "event_arg", "accuracy"),
]

# Pass thresholds.
ARG_END_BASE = 0.576
ARG_SPAN_BASE = 0.477
MATERIAL = 0.03           # min improvement to call "material"
BRIDGE_TOL = 0.03         # max allowed bridge drop


def _val(rep: dict[str, Any], mkey: str, field: str) -> float | None:
    if mkey == "lm_loss":
        return rep.get("lm_loss")
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


def _agg_breakdown(reps: list[dict]) -> dict:
    """Average per-bucket start/end/exact-indep/exact-joint rates across seeds."""

    out: dict[str, dict[str, dict[str, float]]] = {}
    for group in ("by_origin", "by_width", "by_subword"):
        keys: set[str] = set()
        for r in reps:
            keys |= set((r.get("span_breakdown") or {}).get(group, {}).keys())
        out[group] = {}
        for key in sorted(keys):
            accs = {"start": [], "end": [], "exact_indep": [], "exact_joint": [], "n": []}
            for r in reps:
                c = (r.get("span_breakdown") or {}).get(group, {}).get(key)
                if not c or not c.get("n"):
                    continue
                n = c["n"]
                for f in ("start", "end", "exact_indep", "exact_joint"):
                    accs[f].append(c[f] / n)
                accs["n"].append(n)
            if accs["n"]:
                out[group][key] = {f: round(sum(accs[f]) / len(accs[f]), 3)
                                   for f in ("start", "end", "exact_indep", "exact_joint")}
                out[group][key]["n_mean"] = round(sum(accs["n"]) / len(accs["n"]), 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="450M span-END repair gate aggregation.")
    p.add_argument("--span", type=Path, nargs="+", required=True, help="Span-repair per-seed eval JSONs.")
    p.add_argument("--baseline", type=Path, nargs="+", required=True, help="Baseline-450M per-seed eval JSONs.")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_span_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_span_gate.md")
    args = p.parse_args()

    span = [json.loads(Path(s).read_text()) for s in args.span]
    base = [json.loads(Path(b).read_text()) for b in args.baseline]

    agg: dict[str, dict[str, Any]] = {}
    for label, mkey, field in METRICS:
        sv = [v for v in (_val(r, mkey, field) for r in span) if v is not None]
        bv = [v for v in (_val(r, mkey, field) for r in base) if v is not None]
        sm, ss = _ms(sv)
        bm, bs = _ms(bv)
        agg[label] = {"span_mean": sm, "span_std": ss, "base_mean": bm, "base_std": bs,
                      "delta": (sm - bm) if (sv and bv) else None}

    arg_end = agg["argument_end"]["span_mean"]
    arg_span = agg["arg_span_exact"]["span_mean"]
    prim = agg["primitive_macro_f1"]
    r2p = agg["role_to_primitive_macro_f1"]
    evi = agg["evidence_pointer"]["span_mean"]

    cond_end = arg_end >= ARG_END_BASE + MATERIAL
    cond_span = arg_span >= ARG_SPAN_BASE + MATERIAL
    cond_bridge = (prim["delta"] is None or prim["delta"] >= -BRIDGE_TOL) and \
                  (r2p["delta"] is None or r2p["delta"] >= -BRIDGE_TOL)
    cond_evi = evi >= 0.98
    gate_pass = bool(cond_end and cond_span and cond_bridge and cond_evi)

    verdict = {
        "argument_end_material": cond_end,
        "arg_span_exact_material": cond_span,
        "bridge_held": cond_bridge,
        "evidence_near_one": cond_evi,
        "gate_pass": gate_pass,
        "thresholds": {"arg_end_base": ARG_END_BASE, "arg_span_base": ARG_SPAN_BASE,
                       "material": MATERIAL, "bridge_tol": BRIDGE_TOL},
    }
    out = {"span_seeds": [str(s) for s in args.span], "baseline_seeds": [str(b) for b in args.baseline],
           "aggregate": agg, "breakdown_span": _agg_breakdown(span),
           "breakdown_baseline": _agg_breakdown(base), "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- Markdown + console ----
    lines = [f"# 450M span-END repair gate ({len(span)} seeds)", "",
             "Joint boundary-aware span loss + span-focused sampler vs prior 450M baseline.", "",
             "| metric | span-repair mean±std | baseline mean±std | Δ |", "|---|---|---|---|"]
    for label, _m, _f in METRICS:
        a = agg[label]
        d = "—" if a["delta"] is None else f"{a['delta']:+.3f}"
        lines.append(f"| {label} | {a['span_mean']:.3f} ± {a['span_std']:.3f} | "
                     f"{a['base_mean']:.3f} ± {a['base_std']:.3f} | {d} |")
    bd = out["breakdown_span"]
    bb = out["breakdown_baseline"]
    lines += ["", "## Span breakdown (exact-indep, span-repair vs baseline)", "",
              "| bucket | span exact | baseline exact | Δ | n |", "|---|---|---|---|---|"]
    for group in ("by_origin", "by_width", "by_subword"):
        for key, c in bd[group].items():
            cb = bb[group].get(key, {})
            be = cb.get("exact_indep")
            d = "—" if be is None else f"{c['exact_indep'] - be:+.3f}"
            lines.append(f"| {group[3:]}:{key} | {c['exact_indep']:.3f} | "
                         f"{be if be is None else f'{be:.3f}'} | {d} | {c['n_mean']:.0f} |")
    lines += ["", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- argument_end material (≥{ARG_END_BASE}+{MATERIAL}): {cond_end} (mean {arg_end:.3f})",
              f"- arg_span_exact material (≥{ARG_SPAN_BASE}+{MATERIAL}): {cond_span} (mean {arg_span:.3f})",
              f"- bridge held (Δ≥−{BRIDGE_TOL}): {cond_bridge} "
              f"(primitive Δ={prim['delta']:+.3f}, role_to_primitive Δ={r2p['delta']:+.3f})",
              f"- evidence_pointer ≥0.98: {cond_evi} (mean {evi:.3f})"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M span-END repair gate  span_seeds={len(span)} baseline_seeds={len(base)}")
    for label, _m, _f in METRICS:
        a = agg[label]
        d = "  n/a" if a["delta"] is None else f"{a['delta']:+.3f}"
        print(f"  {label:28s} span={a['span_mean']:.3f}±{a['span_std']:.3f}  base={a['base_mean']:.3f}  Δ={d}")
    print("\nbreakdown (exact-indep) span vs baseline:")
    for group in ("by_origin", "by_width", "by_subword"):
        for key, c in bd[group].items():
            be = bb[group].get(key, {}).get("exact_indep")
            ds = "n/a" if be is None else f"{c['exact_indep'] - be:+.3f}"
            print(f"  {group[3:]:>8}:{key:<8} span={c['exact_indep']:.3f} base={be if be is None else f'{be:.3f}'} Δ={ds} n={c['n_mean']:.0f}")
    print(f"\nargument_end material: {cond_end}  arg_span_exact material: {cond_span}  "
          f"bridge held: {cond_bridge}  evidence~1: {cond_evi}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
