#!/usr/bin/env python3
"""450M support-coverage gate aggregation.

Same diversified 450M recipe plus --support-coverage (varied, rule-agnostic
support-licensing clauses + counterfactual support). Compared against the frozen
diversified 450M baseline, whose aggregate means are read from its gate report
(qwen_450m_divrec_gate.json) -- the support gap that baseline left was the target.

Pass conditions (user-specified):
  - support val improves materially from the frozen baseline;
  - support train/val gap shrinks from the frozen +0.152;
  - span / bridge / verifier / evidence stay near (or above) the frozen baseline.

CPU-only, offline. Reads JSON, writes JSON + Markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# label -> (metrics key, field) for the per-seed support-coverage evals.
ACC = [("support", "support"), ("idk", "idk"), ("verifier", "verifier"),
       ("arg_span_joint_any", "arg_span_joint_any"), ("evidence_pointer", "evidence_pointer")]
MF1 = [("primitive_macro_f1", "primitive"), ("role_to_primitive_macro_f1", "role_to_primitive")]

MATERIAL = 0.03
GUARD_TOL = 0.03
FROZEN_SUPPORT_GAP = 0.152


def _ms(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def _acc(rs: list[dict], k: str) -> list[float]:
    return [r["metrics"][k]["accuracy"] for r in rs if r["metrics"].get(k)]


def _mf1(rs: list[dict], k: str) -> list[float]:
    return [r["metrics"][k]["macro_f1"] for r in rs if r["metrics"].get(k)]


def _origin(rs: list[dict], o: str) -> list[float]:
    out = []
    for r in rs:
        c = (r.get("span_breakdown") or {}).get("by_origin", {}).get(o)
        if c and c.get("n"):
            out.append(c["exact_joint"] / c["n"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="450M support-coverage gate aggregation.")
    p.add_argument("--divsup", type=Path, nargs="+", required=True, help="Support-coverage val-split eval JSONs.")
    p.add_argument("--divsup-train", type=Path, nargs="+", required=True, help="Support-coverage train-split eval JSONs.")
    p.add_argument("--frozen-gate", type=Path, required=True,
                   help="Frozen diversified baseline gate JSON (qwen_450m_divrec_gate.json); aggregate.*.divsurf_mean.")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_divsup_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_divsup_gate.md")
    args = p.parse_args()

    ev = [json.loads(Path(s).read_text()) for s in args.divsup]
    tr = [json.loads(Path(s).read_text()) for s in args.divsup_train]
    fb = json.loads(args.frozen_gate.read_text())["aggregate"]

    def fbm(label: str) -> float | None:
        node = fb.get(label)
        return float(node["divsurf_mean"]) if isinstance(node, dict) and "divsurf_mean" in node else None

    agg: dict[str, dict[str, Any]] = {}
    for label, k in ACC:
        sm, ss = _ms(_acc(ev, k))
        bm = fbm(label)
        agg[label] = {"mean": sm, "std": ss, "frozen": bm, "delta": (sm - bm) if bm is not None else None}
    for label, k in MF1:
        sm, ss = _ms(_mf1(ev, k))
        bm = fbm(label)
        agg[label] = {"mean": sm, "std": ss, "frozen": bm, "delta": (sm - bm) if bm is not None else None}
    for o, label in (("synthetic", "span_synthetic"), ("external", "span_external")):
        sm, ss = _ms(_origin(ev, o))
        bm = fbm(label)
        agg[label] = {"mean": sm, "std": ss, "frozen": bm, "delta": (sm - bm) if bm is not None else None}
    lm_m, lm_s = _ms([r["lm_loss"] for r in ev])
    agg["lm_loss"] = {"mean": lm_m, "std": lm_s, "frozen": fbm("lm_loss"),
                      "delta": (lm_m - fbm("lm_loss")) if fbm("lm_loss") is not None else None}

    # support train/val gap (this run).
    sup_tr, _ = _ms(_acc(tr, "support"))
    sup_va = agg["support"]["mean"]
    support_gap = sup_tr - sup_va

    cond_support = (agg["support"]["delta"] or -1) >= MATERIAL
    cond_gap = support_gap < FROZEN_SUPPORT_GAP - 1e-9
    # near-or-above frozen on span/bridge/verifier/evidence
    near = {}
    for label in ("arg_span_joint_any", "primitive_macro_f1", "role_to_primitive_macro_f1", "verifier", "evidence_pointer"):
        d = agg[label]["delta"]
        near[label] = (d is None) or (d >= -GUARD_TOL)
    cond_near = all(near.values())
    gate_pass = bool(cond_support and cond_gap and cond_near)

    verdict = {"support_material": cond_support, "support_delta": agg["support"]["delta"],
               "support_gap": support_gap, "frozen_support_gap": FROZEN_SUPPORT_GAP, "gap_shrinks": cond_gap,
               "span_bridge_verifier_evidence_near": cond_near, "near_detail": near,
               "gate_pass": gate_pass, "thresholds": {"material": MATERIAL, "guard_tol": GUARD_TOL}}
    out = {"divsup_seeds": [str(s) for s in args.divsup], "frozen_gate": str(args.frozen_gate),
           "aggregate": agg, "support_train_val_gap": support_gap, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    order = ["support", "idk", "verifier", "span_synthetic", "span_external", "arg_span_joint_any",
             "primitive_macro_f1", "role_to_primitive_macro_f1", "evidence_pointer", "lm_loss"]
    lines = [f"# 450M support-coverage gate ({len(ev)} seeds)", "",
             "Diversified recipe + --support-coverage vs the frozen diversified 450M baseline.", "",
             "| metric | support-cov mean±std | frozen baseline | Δ |", "|---|---|---|---|"]
    for label in order:
        a = agg[label]
        fb_s = "—" if a["frozen"] is None else f"{a['frozen']:.3f}"
        dl = "—" if a["delta"] is None else f"{a['delta']:+.3f}"
        lines.append(f"| {label} | {a['mean']:.3f} ± {a['std']:.3f} | {fb_s} | {dl} |")
    lines += ["", f"- support train/val gap: **{support_gap:+.3f}** (frozen was +{FROZEN_SUPPORT_GAP:.3f})",
              "", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- support improves materially (≥{MATERIAL}): {cond_support} (Δ={agg['support']['delta']:+.3f})",
              f"- support train/val gap shrinks: {cond_gap} ({support_gap:+.3f} < +{FROZEN_SUPPORT_GAP:.3f})",
              f"- span/bridge/verifier/evidence near-or-above frozen: {cond_near} ({near})"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M support-coverage gate  seeds={len(ev)}")
    for label in order:
        a = agg[label]
        dl = "  n/a" if a["delta"] is None else f"{a['delta']:+.3f}"
        fb_s = "n/a" if a["frozen"] is None else f"{a['frozen']:.3f}"
        print(f"  {label:30s} divsup={a['mean']:.3f}±{a['std']:.3f}  frozen={fb_s}  Δ={dl}")
    print(f"\nsupport train/val gap={support_gap:+.3f} (frozen +{FROZEN_SUPPORT_GAP})")
    print(f"support material: {cond_support}  gap shrinks: {cond_gap}  near: {cond_near} {near}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
