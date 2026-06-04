#!/usr/bin/env python3
"""450M stability gate aggregation.

Longer-schedule (set-valued span objective kept) 450M run vs:
  - the short 400-step set-valued baseline (qwen_450m_set_eval_s*.json), and
  - the tiny full-model reference on the same 25/75 mix
    (mixed_25_75_fullmodel_balanced.json), for support/idk/verifier recovery.

The gate is about variance and recovery, not span surgery. Pass conditions
(user-specified):
  - primitive / role_to_primitive macro-F1 stay strong or improve;
  - support / idk / verifier recover toward tiny levels;
  - any-occurrence span variance shrinks (std over seeds drops vs the short run);
  - no evidence-pointer regression.

Also reports the train/val gap from per-seed train-split evals. CPU-only, offline.

Usage:
    python3 scripts/report_450m_stability.py \
        --stab artifacts/reports/qwen_450m_stab_eval_s{0,1,2}.json \
        --stab-train artifacts/reports/qwen_450m_stab_traineval_s{0,1,2}.json \
        --short artifacts/reports/qwen_450m_set_eval_s{0,1,2}.json \
        --tiny artifacts/reports/mixed_25_75_fullmodel_balanced.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# label -> (metrics key, field).
METRICS: list[tuple[str, str, str]] = [
    ("primitive_macro_f1", "primitive", "macro_f1"),
    ("role_to_primitive_macro_f1", "role_to_primitive", "macro_f1"),
    ("support", "support", "accuracy"),
    ("idk", "idk", "accuracy"),
    ("verifier", "verifier", "accuracy"),
    ("evidence_pointer", "evidence_pointer", "accuracy"),
    ("arg_span_joint_any", "arg_span_joint_any", "accuracy"),
    ("arg_span_any", "arg_span_any", "accuracy"),
    ("argument_start_any", "argument_start_any", "accuracy"),
    ("argument_end_any", "argument_end_any", "accuracy"),
    ("arg_span_exact_strict", "arg_span_exact", "accuracy"),
    ("lm_loss", "__lm_loss__", "value"),
]

# Recovery / regression tolerances.
BRIDGE_TOL = 0.03   # bridge "stays strong" band vs short baseline
RECOVER_NEAR = 0.05  # within this of tiny counts as recovered
RECOVER_STEP = 0.01  # or improved over short baseline by at least this


def _val(rep: dict[str, Any], mkey: str, field: str) -> float | None:
    if mkey == "__lm_loss__":
        v = rep.get("lm_loss")
        return float(v) if isinstance(v, (int, float)) else None
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


def _tiny_levels(tiny_path: Path | None) -> dict[str, float]:
    """Read tiny full-model means from the multi-seed aggregate report."""
    if not tiny_path or not tiny_path.exists():
        return {}
    d = json.loads(tiny_path.read_text())
    agg = d.get("aggregate", {})
    variant = next(iter(agg.values()), {}) if agg else {}
    out: dict[str, float] = {}
    for key in ("support", "idk", "verifier", "evidence_pointer",
                "primitive_macro_f1", "role_to_primitive_macro_f1", "lm_loss"):
        node = variant.get(key)
        if isinstance(node, dict) and "mean" in node:
            out[key] = float(node["mean"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="450M stability gate aggregation.")
    p.add_argument("--stab", type=Path, nargs="+", required=True, help="Stability val-split eval JSONs.")
    p.add_argument("--stab-train", type=Path, nargs="*", default=[], help="Stability train-split eval JSONs (for gap).")
    p.add_argument("--short", type=Path, nargs="+", required=True, help="Short 400-step set-valued eval JSONs.")
    p.add_argument("--tiny", type=Path, default=ROOT / "artifacts" / "reports" / "mixed_25_75_fullmodel_balanced.json")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_stability_gate.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "artifacts" / "reports" / "qwen_450m_stability_gate.md")
    args = p.parse_args()

    stab = [json.loads(Path(s).read_text()) for s in args.stab]
    short = [json.loads(Path(s).read_text()) for s in args.short]
    stab_tr = [json.loads(Path(s).read_text()) for s in args.stab_train]
    tiny = _tiny_levels(args.tiny)

    agg: dict[str, dict[str, Any]] = {}
    for label, mkey, field in METRICS:
        sv = [v for v in (_val(r, mkey, field) for r in stab) if v is not None]
        bv = [v for v in (_val(r, mkey, field) for r in short) if v is not None]
        sm, ss = _ms(sv)
        bm, bs = _ms(bv)
        agg[label] = {"stab_mean": sm, "stab_std": ss, "short_mean": bm, "short_std": bs,
                      "delta": (sm - bm) if (sv and bv) else None, "tiny": tiny.get(label)}

    # Train/val gap (train minus val) for a few headline metrics, per seed then mean.
    gap: dict[str, Any] = {}
    if stab_tr:
        for label, mkey, field in METRICS:
            tr = [v for v in (_val(r, mkey, field) for r in stab_tr) if v is not None]
            va = [v for v in (_val(r, mkey, field) for r in stab) if v is not None]
            if tr and va and len(tr) == len(va):
                gaps = [t - v for t, v in zip(tr, va)]
                gm, _ = _ms(gaps)
                gap[label] = {"train_mean": _ms(tr)[0], "val_mean": _ms(va)[0], "gap_mean": gm}

    def d(label: str) -> float | None:
        return agg[label]["delta"]

    # Conditions.
    prim_d, r2p_d = d("primitive_macro_f1"), d("role_to_primitive_macro_f1")
    cond_bridge = (prim_d is None or prim_d >= -BRIDGE_TOL) and (r2p_d is None or r2p_d >= -BRIDGE_TOL)

    def recovered(label: str) -> bool:
        a = agg[label]
        sm, bm, tn = a["stab_mean"], a["short_mean"], a["tiny"]
        if tn is None:
            return False
        # close to tiny, or moved toward tiny vs the short run
        return (sm >= tn - RECOVER_NEAR) or (sm >= bm + RECOVER_STEP and sm <= tn)
    rec = {h: recovered(h) for h in ("support", "idk", "verifier")}
    cond_recover = all(rec.values())

    short_std = agg["arg_span_joint_any"]["short_std"]
    stab_std = agg["arg_span_joint_any"]["stab_std"]
    cond_var = stab_std < short_std

    cond_evi = agg["evidence_pointer"]["stab_mean"] >= 0.98

    gate_pass = bool(cond_bridge and cond_recover and cond_var and cond_evi)

    verdict = {
        "bridge_strong": cond_bridge, "support_idk_verifier_recover": cond_recover,
        "recovery_detail": rec, "span_variance_shrinks": cond_var,
        "span_any_std_short": short_std, "span_any_std_stab": stab_std,
        "evidence_no_regression": cond_evi, "gate_pass": gate_pass,
        "thresholds": {"bridge_tol": BRIDGE_TOL, "recover_near": RECOVER_NEAR, "recover_step": RECOVER_STEP},
    }
    out = {"stab_seeds": [str(s) for s in args.stab], "short_seeds": [str(s) for s in args.short],
           "tiny_ref": str(args.tiny), "aggregate": agg, "train_val_gap": gap, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [f"# 450M stability gate ({len(stab)} stability seeds vs {len(short)} short seeds)", "",
             "Longer schedule, set-valued span objective kept. Tiny = full-model on the same 25/75 mix.", "",
             "| metric | stability mean±std | short 400-step mean±std | Δ | tiny |", "|---|---|---|---|---|"]
    for label, _m, _f in METRICS:
        a = agg[label]
        dv = "—" if a["delta"] is None else f"{a['delta']:+.3f}"
        tn = "—" if a["tiny"] is None else f"{a['tiny']:.3f}"
        lines.append(f"| {label} | {a['stab_mean']:.3f} ± {a['stab_std']:.3f} | "
                     f"{a['short_mean']:.3f} ± {a['short_std']:.3f} | {dv} | {tn} |")
    if gap:
        lines += ["", "## Train/val gap (train − val)", "", "| metric | train | val | gap |", "|---|---|---|---|"]
        for label in ("primitive_macro_f1", "role_to_primitive_macro_f1", "arg_span_joint_any",
                      "support", "idk", "verifier", "lm_loss"):
            g = gap.get(label)
            if g:
                lines.append(f"| {label} | {g['train_mean']:.3f} | {g['val_mean']:.3f} | {g['gap_mean']:+.3f} |")
    lines += ["", "## Verdict", "", f"- **Gate pass: {gate_pass}**",
              f"- bridge stays strong (Δ≥-{BRIDGE_TOL}): {cond_bridge} (primitive Δ={prim_d}, role_to_primitive Δ={r2p_d})",
              f"- support/idk/verifier recover toward tiny: {cond_recover} ({rec})",
              f"- span any-occurrence variance shrinks: {cond_var} (std {short_std:.3f} → {stab_std:.3f})",
              f"- evidence_pointer no regression (≥0.98): {cond_evi} (mean {agg['evidence_pointer']['stab_mean']:.3f})"]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"450M stability gate  stab_seeds={len(stab)} short_seeds={len(short)}")
    for label, _m, _f in METRICS:
        a = agg[label]
        dv = "  n/a" if a["delta"] is None else f"{a['delta']:+.3f}"
        tn = "  n/a" if a["tiny"] is None else f"{a['tiny']:.3f}"
        print(f"  {label:30s} stab={a['stab_mean']:.3f}±{a['stab_std']:.3f}  short={a['short_mean']:.3f}  Δ={dv}  tiny={tn}")
    if gap:
        print("train/val gap (train-val):")
        for label in ("primitive_macro_f1", "role_to_primitive_macro_f1", "arg_span_joint_any", "support", "idk", "verifier", "lm_loss"):
            g = gap.get(label)
            if g:
                print(f"  {label:30s} train={g['train_mean']:.3f} val={g['val_mean']:.3f} gap={g['gap_mean']:+.3f}")
    print(f"\nbridge strong: {cond_bridge}  recover: {cond_recover} {rec}  "
          f"var shrinks: {cond_var} ({short_std:.3f}->{stab_std:.3f})  evidence~1: {cond_evi}")
    print(f"GATE PASS: {gate_pass}")
    print(f"wrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
