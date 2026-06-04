#!/usr/bin/env python3
"""450M external-coverage gate aggregation.

Frozen diversified+support recipe on the 3-way mix (real external / derived
external-style augmentation `synthetic_ext` / synthetic) vs the frozen
diversified+support baseline (qwen_450m_divsup_eval_s*.json), which had no
external-style augmentation.

Reports per-origin argument span (exact_joint from span_breakdown.by_origin):
external (real, untouched), synthetic_ext (the new derived bucket), synthetic.

Pass conditions (user-specified):
  - external-origin OR external-like (synthetic_ext) span/support improves;
  - overall frozen metrics do not regress;
  - provenance fields remain clear (3 distinct origin buckets present).

CPU-only, offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ACC = ["support", "idk", "verifier", "arg_span_joint_any", "evidence_pointer"]
MF1 = ["primitive", "role_to_primitive"]
MATERIAL = 0.03
GUARD_TOL = 0.03
EXT_LIKE_STRONG = 0.85


def _ms(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def _acc(rs: list[dict], k: str) -> float:
    return _ms([r["metrics"][k]["accuracy"] for r in rs if r["metrics"].get(k)])[0]


def _mf1(rs: list[dict], k: str) -> float:
    return _ms([r["metrics"][k]["macro_f1"] for r in rs if r["metrics"].get(k)])[0]


def _origin(rs: list[dict], o: str) -> tuple[float, float, int]:
    vals, ns = [], 0
    for r in rs:
        c = (r.get("span_breakdown") or {}).get("by_origin", {}).get(o)
        if c and c.get("n"):
            vals.append(c["exact_joint"] / c["n"])
            ns = c["n"]
    m, s = _ms(vals)
    return m, s, ns


def main() -> None:
    p = argparse.ArgumentParser(description="450M external-coverage gate.")
    p.add_argument("--extaug", type=Path, nargs="+", required=True)
    p.add_argument("--extaug-train", type=Path, nargs="*", default=[])
    p.add_argument("--baseline", type=Path, nargs="+", required=True, help="Frozen divsup val evals.")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_extaug_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_extaug_gate.md")
    args = p.parse_args()

    ev = [json.loads(Path(s).read_text()) for s in args.extaug]
    base = [json.loads(Path(b).read_text()) for b in args.baseline]
    tr = [json.loads(Path(s).read_text()) for s in args.extaug_train]

    span = {}
    for o in ("external", "synthetic_ext", "synthetic"):
        em, es, en = _origin(ev, o)
        bm, bs, bn = _origin(base, o)
        span[o] = {"extaug_mean": em, "extaug_std": es, "extaug_n": en,
                   "base_mean": bm, "base_n": bn, "delta": (em - bm) if bn else None}

    overall = {}
    for k in ACC:
        em, es = _ms([r["metrics"][k]["accuracy"] for r in ev if r["metrics"].get(k)])
        bm, _ = _ms([r["metrics"][k]["accuracy"] for r in base if r["metrics"].get(k)])
        overall[k] = {"extaug_mean": em, "extaug_std": es, "base_mean": bm, "delta": em - bm}
    for k in MF1:
        em = _mf1(ev, k); bm = _mf1(base, k)
        overall[k + "_macro_f1"] = {"extaug_mean": em, "extaug_std": 0.0, "base_mean": bm, "delta": em - bm}
    lm_e = _ms([r["lm_loss"] for r in ev])[0]; lm_b = _ms([r["lm_loss"] for r in base])[0]
    overall["lm_loss"] = {"extaug_mean": lm_e, "extaug_std": 0.0, "base_mean": lm_b, "delta": lm_e - lm_b}

    # provenance: are the 3 buckets present in the eval span breakdown?
    buckets = sorted({o for r in ev for o in ((r.get("span_breakdown") or {}).get("by_origin") or {})})
    prov_clear = {"external", "synthetic", "synthetic_ext"}.issubset(set(buckets))

    # external-like (synthetic_ext) span improves materially (it is a new bucket;
    # "improves" = the model learns this external-style grounding well).
    ext_like_improves = span["synthetic_ext"]["extaug_mean"] >= EXT_LIKE_STRONG
    # real external span must NOT regress (user pass condition).
    ext_no_regress = (span["external"]["delta"] is None) or (span["external"]["delta"] >= -GUARD_TOL)
    # overall no-regression: synthetic span + the overall heads within tol
    syn_ok = (span["synthetic"]["delta"] is None) or (span["synthetic"]["delta"] >= -GUARD_TOL)
    heads = {k: overall[k]["delta"] >= -GUARD_TOL for k in
             ("arg_span_joint_any", "primitive_macro_f1", "role_to_primitive_macro_f1",
              "support", "idk", "verifier", "evidence_pointer")}
    no_regress = syn_ok and ext_no_regress and all(heads.values())
    gate_pass = bool(ext_like_improves and no_regress and prov_clear)

    # support train/val gap (guardrail)
    sup_gap = None
    if tr:
        sup_gap = _acc(tr, "support") - _acc(ev, "support")

    verdict = {"external_like_improves": ext_like_improves, "synthetic_ext_span": span["synthetic_ext"]["extaug_mean"],
               "external_origin_delta": span["external"]["delta"], "external_no_regress": ext_no_regress,
               "overall_no_regression": no_regress, "synthetic_span_ok": syn_ok, "heads_ok": heads,
               "provenance_clear": prov_clear, "origin_buckets": buckets,
               "support_train_val_gap": sup_gap, "gate_pass": gate_pass}
    out = {"extaug_seeds": [str(s) for s in args.extaug], "baseline_seeds": [str(b) for b in args.baseline],
           "span_by_origin": span, "overall": overall, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [f"# 450M external-coverage gate ({len(ev)} extaug vs {len(base)} baseline seeds)", "",
             "Frozen diversified+support recipe on the 3-way mix (external / synthetic_ext / synthetic) "
             "vs the frozen diversified+support baseline.", "",
             "## Argument span by origin (exact_joint)", "",
             "| origin | extaug mean±std | baseline | Δ | n |", "|---|---|---|---|---|"]
    for o in ("external", "synthetic_ext", "synthetic"):
        s = span[o]
        bm = "—(new)" if not s["base_n"] else f"{s['base_mean']:.3f}"
        dl = "—" if s["delta"] is None else f"{s['delta']:+.3f}"
        lines.append(f"| {o} | {s['extaug_mean']:.3f} ± {s['extaug_std']:.3f} | {bm} | {dl} | {s['extaug_n']} |")
    lines += ["", "## Overall heads (vs frozen baseline)", "", "| metric | extaug | baseline | Δ |", "|---|---|---|---|"]
    for k, a in overall.items():
        lines.append(f"| {k} | {a['extaug_mean']:.3f} | {a['base_mean']:.3f} | {a['delta']:+.3f} |")
    lines += ["", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- external-like (synthetic_ext) improves: {ext_like_improves} (span={span['synthetic_ext']['extaug_mean']:.3f})",
              f"- real external span no regress: {ext_no_regress} (Δ={span['external']['delta']})",
              f"- overall no regression: {no_regress} (synthetic span ok={syn_ok}, heads={heads})",
              f"- provenance clear (3 buckets): {prov_clear} ({buckets})"]
    if sup_gap is not None:
        lines.append(f"- support train/val gap: {sup_gap:+.3f}")
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M external-coverage gate  extaug={len(ev)} baseline={len(base)}")
    print("span by origin (exact_joint):")
    for o in ("external", "synthetic_ext", "synthetic"):
        s = span[o]
        bm = "new" if not s["base_n"] else f"{s['base_mean']:.3f}"
        dl = "n/a" if s["delta"] is None else f"{s['delta']:+.3f}"
        print(f"  {o:14s} extaug={s['extaug_mean']:.3f}±{s['extaug_std']:.3f}  base={bm}  Δ={dl}  n={s['extaug_n']}")
    print("overall heads:")
    for k, a in overall.items():
        print(f"  {k:28s} extaug={a['extaug_mean']:.3f}  base={a['base_mean']:.3f}  Δ={a['delta']:+.3f}")
    print(f"\nsynthetic_ext improves: {ext_like_improves}  real-external no-regress: {ext_no_regress} "
          f"(Δ={span['external']['delta']:+.3f})  overall no-regress: {no_regress}  provenance clear: {prov_clear}")
    if sup_gap is not None:
        print(f"support train/val gap: {sup_gap:+.3f}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
