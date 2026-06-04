#!/usr/bin/env python3
"""450M real-external candidate gate (step 6).

Frozen divsup recipe on the real-data candidate mix (existing external +
ProofWriter meta-abduction extension + divsup synthetic; no synthetic_ext) vs the
frozen divsup baseline (qwen_450m_divsup_eval_s*.json).

Headline: real **abduction/hypothesis** coverage now appears in val/test, and the
abduction primitive recall (bridge per-class) is measurable. Guardrails:
external-origin span and the overall heads must stay within tolerance.

Pass conditions (user-specified):
  - hypothesis/abduction coverage appears in val/test;
  - abduction (or hypothesis) metrics improve from absent to measurable;
  - real-external span does not regress materially;
  - overall frozen metrics within tolerance.

CPU-only, offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD_TOL = 0.03


def _ms(xs):
    n = len(xs)
    if not n:
        return float("nan"), 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def _per_class_recall(reps, head, cls):
    """Aggregate 'c/t' per-class recall across seeds -> (recall, n_total)."""
    c = t = 0
    for r in reps:
        node = (r.get("metrics", {}).get(head) or {}).get("per_class_recall") or {}
        cell = node.get(cls)
        if isinstance(cell, str) and "/" in cell:
            a, b = cell.split("/")
            c += int(a); t += int(b)
    return (c / t if t else float("nan")), t


def _acc(reps, k):
    return _ms([r["metrics"][k]["accuracy"] for r in reps if r["metrics"].get(k)])[0]


def _mf1(reps, k):
    return _ms([r["metrics"][k]["macro_f1"] for r in reps if r["metrics"].get(k)])[0]


def _origin_span(reps, o):
    vals = []
    for r in reps:
        c = (r.get("span_breakdown") or {}).get("by_origin", {}).get(o)
        if c and c.get("n"):
            vals.append(c["exact_joint"] / c["n"])
    return _ms(vals)[0]


def main():
    p = argparse.ArgumentParser(description="450M real-external candidate gate.")
    p.add_argument("--realext", type=Path, nargs="+", required=True, help="Candidate val evals.")
    p.add_argument("--realext-test", type=Path, nargs="*", default=[], help="Candidate test evals.")
    p.add_argument("--baseline", type=Path, nargs="+", required=True, help="Frozen divsup val evals.")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_realext_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_realext_gate.md")
    args = p.parse_args()

    ev = [json.loads(Path(s).read_text()) for s in args.realext]
    te = [json.loads(Path(s).read_text()) for s in args.realext_test]
    base = [json.loads(Path(b).read_text()) for b in args.baseline]

    abd_val, abd_val_n = _per_class_recall(ev, "primitive", "abduction")
    abd_test, abd_test_n = _per_class_recall(te, "primitive", "abduction") if te else (float("nan"), 0)
    abd_base, abd_base_n = _per_class_recall(base, "primitive", "abduction")

    heads = {}
    for k in ("arg_span_joint_any", "support", "idk", "verifier", "evidence_pointer"):
        heads[k] = {"realext": _acc(ev, k), "base": _acc(base, k)}
    for k in ("primitive", "role_to_primitive"):
        heads[k + "_macro_f1"] = {"realext": _mf1(ev, k), "base": _mf1(base, k)}
    heads["external_span"] = {"realext": _origin_span(ev, "external"), "base": _origin_span(base, "external")}
    for h in heads.values():
        h["delta"] = h["realext"] - h["base"]

    cond_cover = abd_val_n > 0 and (abd_test_n > 0 if te else True)
    cond_measurable = abd_val_n > 0 and abd_val == abd_val  # not NaN
    cond_ext_span = heads["external_span"]["delta"] >= -GUARD_TOL
    overall_ok = all(heads[k]["delta"] >= -GUARD_TOL for k in
                     ("arg_span_joint_any", "primitive_macro_f1", "role_to_primitive_macro_f1",
                      "support", "idk", "verifier", "evidence_pointer"))
    gate_pass = bool(cond_cover and cond_measurable and cond_ext_span and overall_ok)

    verdict = {"abduction_coverage_val_test": cond_cover, "abduction_recall_measurable": cond_measurable,
               "abduction_recall_val": abd_val, "abduction_n_val": abd_val_n,
               "abduction_recall_test": abd_test, "abduction_n_test": abd_test_n,
               "abduction_recall_base": abd_base, "abduction_n_base": abd_base_n,
               "real_external_span_ok": cond_ext_span, "external_span_delta": heads["external_span"]["delta"],
               "overall_within_tol": overall_ok, "gate_pass": gate_pass, "guard_tol": GUARD_TOL}
    out = {"realext_seeds": [str(s) for s in args.realext], "baseline_seeds": [str(b) for b in args.baseline],
           "abduction": {"val": [abd_val, abd_val_n], "test": [abd_test, abd_test_n], "base": [abd_base, abd_base_n]},
           "heads": heads, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [f"# 450M real-external candidate gate ({len(ev)} seeds)", "",
             "Frozen recipe on the real-data candidate mix (external + ProofWriter meta-abduction "
             "extension + divsup synthetic; no synthetic_ext) vs the divsup baseline.", "",
             "## Abduction (real-data target; bridge per-class recall)", "",
             "| split | recall | n (abduction spans) |", "|---|---|---|",
             f"| candidate val | {abd_val:.3f} | {abd_val_n} |",
             f"| candidate test | {abd_test:.3f} | {abd_test_n} |",
             f"| divsup baseline val | {abd_base:.3f} | {abd_base_n} |", "",
             "## Guardrails (candidate vs baseline)", "", "| metric | candidate | baseline | Δ |", "|---|---|---|---|"]
    for k in ("external_span", "arg_span_joint_any", "primitive_macro_f1", "role_to_primitive_macro_f1",
              "support", "idk", "verifier", "evidence_pointer"):
        h = heads[k]
        lines.append(f"| {k} | {h['realext']:.3f} | {h['base']:.3f} | {h['delta']:+.3f} |")
    lines += ["", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- abduction/hypothesis coverage in val/test: {cond_cover} (val n={abd_val_n}, test n={abd_test_n})",
              f"- abduction recall measurable: {cond_measurable} (val {abd_val:.3f})",
              f"- real-external span no material regress: {cond_ext_span} (Δ={heads['external_span']['delta']:+.3f})",
              f"- overall within tolerance: {overall_ok}"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M real-external candidate gate  seeds={len(ev)}")
    print(f"abduction recall  val={abd_val:.3f} (n={abd_val_n})  test={abd_test:.3f} (n={abd_test_n})  base={abd_base:.3f} (n={abd_base_n})")
    for k in ("external_span", "arg_span_joint_any", "primitive_macro_f1", "role_to_primitive_macro_f1",
              "support", "idk", "verifier", "evidence_pointer"):
        h = heads[k]
        print(f"  {k:28s} cand={h['realext']:.3f}  base={h['base']:.3f}  Δ={h['delta']:+.3f}")
    print(f"\ncoverage:{cond_cover}  measurable:{cond_measurable}  ext-span-ok:{cond_ext_span}  overall-ok:{overall_ok}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
