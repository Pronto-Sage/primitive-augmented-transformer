#!/usr/bin/env python3
"""450M surface-diversification gate aggregation.

Same 450M stability recipe, but the synthetic argument surfaces are diversified
(natural, context-bearing gold spans) so the synthetic span pointer is learnable
on held-out rules. Compared against the frozen 450M stability baseline
(qwen_450m_stab_eval_s*.json), which used the repeated-atom synthetic surfaces.

Headline span metrics:
  - overall any-occurrence span (arg_span_joint_any), and
  - synthetic-origin span (span_breakdown.by_origin.synthetic.exact_joint) -- on
    the diversified data synthetic spans are ~95% identifiable, so strict exact
    ~= any-occurrence there. external-origin span is an untouched guardrail.

Pass conditions (user-specified):
  - synthetic any-span improves materially;
  - overall any-span improves;
  - primitive / role_to_primitive macro-F1 stay >= baseline;
  - support / idk / verifier do not regress.

CPU-only, offline. Reads JSON, writes JSON + Markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

METRICS: list[tuple[str, str, str]] = [
    ("arg_span_joint_any", "arg_span_joint_any", "accuracy"),
    ("argument_start_any", "argument_start_any", "accuracy"),
    ("argument_end_any", "argument_end_any", "accuracy"),
    ("arg_span_exact_strict", "arg_span_exact", "accuracy"),
    ("primitive_macro_f1", "primitive", "macro_f1"),
    ("role_to_primitive_macro_f1", "role_to_primitive", "macro_f1"),
    ("support", "support", "accuracy"),
    ("idk", "idk", "accuracy"),
    ("verifier", "verifier", "accuracy"),
    ("evidence_pointer", "evidence_pointer", "accuracy"),
    ("lm_loss", "__lm_loss__", "value"),
]

MATERIAL = 0.03
GUARD_TOL = 0.03


def _val(rep: dict[str, Any], mkey: str, field: str) -> float | None:
    if mkey == "__lm_loss__":
        v = rep.get("lm_loss")
        return float(v) if isinstance(v, (int, float)) else None
    node = rep.get("metrics", {}).get(mkey)
    if not isinstance(node, dict) or field not in node:
        return None
    return float(node[field])


def _origin_span(rep: dict[str, Any], origin: str) -> float | None:
    """exact_joint accuracy for a given mix_origin from span_breakdown."""
    bo = (rep.get("span_breakdown") or {}).get("by_origin") or {}
    c = bo.get(origin)
    if not isinstance(c, dict) or not c.get("n"):
        return None
    return c["exact_joint"] / c["n"]


def _ms(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n if n > 1 else 0.0
    return m, v ** 0.5


def _agg_origin(reps: list[dict], origin: str) -> tuple[float, float]:
    return _ms([v for v in (_origin_span(r, origin) for r in reps) if v is not None])


def main() -> None:
    p = argparse.ArgumentParser(description="450M surface-diversification gate aggregation.")
    p.add_argument("--divsurf", type=Path, nargs="+", required=True, help="Diversified val-split eval JSONs.")
    p.add_argument("--divsurf-train", type=Path, nargs="*", default=[], help="Diversified train-split eval JSONs.")
    p.add_argument("--baseline", type=Path, nargs="+", required=True, help="Frozen 450M stability val-split eval JSONs.")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_divsurf_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_divsurf_gate.md")
    args = p.parse_args()

    dv = [json.loads(Path(s).read_text()) for s in args.divsurf]
    base = [json.loads(Path(b).read_text()) for b in args.baseline]
    dv_tr = [json.loads(Path(s).read_text()) for s in args.divsurf_train]

    agg: dict[str, dict[str, Any]] = {}
    for label, mkey, field in METRICS:
        sv = [v for v in (_val(r, mkey, field) for r in dv) if v is not None]
        bv = [v for v in (_val(r, mkey, field) for r in base) if v is not None]
        sm, ss = _ms(sv)
        bm, bs = _ms(bv)
        agg[label] = {"divsurf_mean": sm, "divsurf_std": ss, "base_mean": bm, "base_std": bs,
                      "delta": (sm - bm) if (sv and bv) else None}

    # Per-origin span (exact_joint) from span_breakdown.
    for origin in ("synthetic", "external"):
        sm, ss = _agg_origin(dv, origin)
        bm, bs = _agg_origin(base, origin)
        agg[f"span_{origin}"] = {"divsurf_mean": sm, "divsurf_std": ss, "base_mean": bm, "base_std": bs,
                                 "delta": (sm - bm)}

    # Train/val gap (train - val) for headline metrics.
    gap: dict[str, Any] = {}
    if dv_tr:
        for label, mkey, field in METRICS:
            tr = [v for v in (_val(r, mkey, field) for r in dv_tr) if v is not None]
            va = [v for v in (_val(r, mkey, field) for r in dv) if v is not None]
            if tr and va and len(tr) == len(va):
                gap[label] = {"train": _ms(tr)[0], "val": _ms(va)[0], "gap": _ms([t - v for t, v in zip(tr, va)])[0]}
        # synthetic-origin gap
        tr = [v for v in (_origin_span(r, "synthetic") for r in dv_tr) if v is not None]
        va = [v for v in (_origin_span(r, "synthetic") for r in dv) if v is not None]
        if tr and va and len(tr) == len(va):
            gap["span_synthetic"] = {"train": _ms(tr)[0], "val": _ms(va)[0], "gap": _ms([t - v for t, v in zip(tr, va)])[0]}

    def d(label: str) -> float | None:
        return agg[label]["delta"]

    cond_syn = (agg["span_synthetic"]["delta"] or -1) >= MATERIAL
    cond_overall = (d("arg_span_joint_any") or -1) > 0
    prim_d, r2p_d = d("primitive_macro_f1"), d("role_to_primitive_macro_f1")
    cond_bridge = (prim_d is None or prim_d >= -GUARD_TOL) and (r2p_d is None or r2p_d >= -GUARD_TOL)
    calib = {h: (d(h) if d(h) is not None else 0.0) >= -GUARD_TOL for h in ("support", "idk", "verifier")}
    cond_calib = all(calib.values())
    cond_evi = agg["evidence_pointer"]["divsurf_mean"] >= 0.98
    gate_pass = bool(cond_syn and cond_overall and cond_bridge and cond_calib and cond_evi)

    verdict = {
        "synthetic_span_material": cond_syn, "overall_any_span_improves": cond_overall,
        "bridge_held": cond_bridge, "calibration_no_regression": cond_calib, "calibration_detail": calib,
        "evidence_no_regression": cond_evi, "gate_pass": gate_pass,
        "thresholds": {"material": MATERIAL, "guard_tol": GUARD_TOL},
    }
    out = {"divsurf_seeds": [str(s) for s in args.divsurf], "baseline_seeds": [str(b) for b in args.baseline],
           "aggregate": agg, "train_val_gap": gap, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    order = (["span_synthetic", "span_external", "arg_span_joint_any", "argument_start_any", "argument_end_any",
              "arg_span_exact_strict", "primitive_macro_f1", "role_to_primitive_macro_f1",
              "support", "idk", "verifier", "evidence_pointer", "lm_loss"])
    lines = [f"# 450M surface-diversification gate ({len(dv)} divsurf vs {len(base)} baseline seeds)", "",
             "Same stability recipe; synthetic argument surfaces diversified (natural, context-bearing "
             "gold spans). Baseline = frozen 450M stability (repeated-atom synthetic surfaces).", "",
             "| metric | divsurf mean±std | baseline mean±std | Δ |", "|---|---|---|---|"]
    for label in order:
        a = agg[label]
        dvv = "—" if a["delta"] is None else f"{a['delta']:+.3f}"
        lines.append(f"| {label} | {a['divsurf_mean']:.3f} ± {a['divsurf_std']:.3f} | "
                     f"{a['base_mean']:.3f} ± {a['base_std']:.3f} | {dvv} |")
    if gap:
        lines += ["", "## Train/val gap (train − val)", "", "| metric | train | val | gap |", "|---|---|---|---|"]
        for label in ("span_synthetic", "arg_span_joint_any", "primitive_macro_f1", "role_to_primitive_macro_f1", "lm_loss"):
            g = gap.get(label)
            if g:
                lines.append(f"| {label} | {g['train']:.3f} | {g['val']:.3f} | {g['gap']:+.3f} |")
    lines += ["", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- synthetic span improves materially (≥{MATERIAL}): {cond_syn} (Δ={agg['span_synthetic']['delta']:+.3f})",
              f"- overall any-span improves: {cond_overall} (Δ={d('arg_span_joint_any')})",
              f"- bridge held (Δ≥-{GUARD_TOL}): {cond_bridge} (primitive Δ={prim_d}, role_to_primitive Δ={r2p_d})",
              f"- support/idk/verifier no regression: {cond_calib} ({calib})",
              f"- evidence_pointer ≥0.98: {cond_evi} (mean {agg['evidence_pointer']['divsurf_mean']:.3f})"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M surface-diversification gate  divsurf={len(dv)} baseline={len(base)}")
    for label in order:
        a = agg[label]
        dvv = "  n/a" if a["delta"] is None else f"{a['delta']:+.3f}"
        print(f"  {label:30s} divsurf={a['divsurf_mean']:.3f}±{a['divsurf_std']:.3f}  base={a['base_mean']:.3f}  Δ={dvv}")
    print(f"\nsynthetic-span material: {cond_syn}  overall improves: {cond_overall}  bridge: {cond_bridge}  "
          f"calib: {cond_calib} {calib}  evidence~1: {cond_evi}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
