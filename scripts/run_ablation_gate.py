#!/usr/bin/env python3
"""PAT-ER ablation gate: does each architectural stream actually matter?

Trains the tiny config under matched conditions (same steps/dataset/split) for a
set of ablation variants across multiple seeds, evaluates each on the held-out
split, and writes a multi-seed comparison report with bootstrap confidence
intervals on the full-minus-ablation deltas.

Variants
--------
Full-text (the model input still serializes the event_graph -> this is a
supervised-plumbing ablation, not final architecture proof):

    full, no_event, no_primitive, no_ffn, no_pressure

Input-only (event_graph/output stripped from the model input -> lower leakage,
stronger architecture evidence):

    full_input_only, no_event_input_only, no_primitive_input_only,
    no_ffn_input_only, no_pressure_input_only          (with --full-input-only-matrix)

The core architecture question: removing the event-role state must hurt
primitive prediction, grounding, or argument quality. Deltas are computed paired
per seed (full[seed] - ablation[seed]) and bootstrapped over seeds; a component
is "supported" when full beats the ablation with a 95% CI that excludes 0 on the
relevant metrics (and, for the streams, under input-only).

Outputs: artifacts/reports/ablation_gate_multiseed.{json,md} (git-ignored).
CPU/GPU-safe; no 770M; no downloads.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_lm_aux as T
import eval_lm_aux as E

DEFAULT_REPORT_DIR = ROOT / "artifacts" / "reports"
DEFAULT_CKPT_ROOT = ROOT / "artifacts" / "checkpoints" / "ablation"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_RNG_SEED = 12345


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# (display name, train overrides, input_only)
FULLTEXT_VARIANTS = [
    ("full", {}, False),
    ("no_event", {"no_event_stream": True}, False),
    ("no_primitive", {"no_primitive_stream": True}, False),
    ("no_ffn", {"no_ffn_adapters": True}, False),
    ("no_pressure", {"no_pressure": True}, False),
]
INPUT_ONLY_FULL = [
    ("full_input_only", {}, True),
    ("no_event_input_only", {"no_event_stream": True}, True),
    ("no_primitive_input_only", {"no_primitive_stream": True}, True),
    ("no_ffn_input_only", {"no_ffn_adapters": True}, True),
    ("no_pressure_input_only", {"no_pressure": True}, True),
]
INPUT_ONLY_DEFAULT = INPUT_ONLY_FULL[:3]  # full / no_event / no_primitive

# Base-name -> variant entry, for explicit --variants selection.
_FULLTEXT_BY_BASE = {v[0]: v for v in FULLTEXT_VARIANTS}
_INPUT_ONLY_BY_BASE = {
    "full": INPUT_ONLY_FULL[0], "no_event": INPUT_ONLY_FULL[1], "no_primitive": INPUT_ONLY_FULL[2],
    "no_ffn": INPUT_ONLY_FULL[3], "no_pressure": INPUT_ONLY_FULL[4],
}

# Reported metrics: (key, kind). kind "acc"/"f1" higher is better; "loss" lower.
METRICS: list[tuple[str, str]] = [
    ("lm_loss", "loss"),
    ("primitive", "acc"),
    ("primitive_macro_f1", "f1"),
    ("role_to_primitive", "acc"),
    ("role_to_primitive_macro_f1", "f1"),
    ("predicate_event", "acc"),
    ("arg_span_exact", "acc"),
    ("argument_start", "acc"),
    ("proto_role_f1", "f1"),
    ("arg_role_f1", "f1"),
    ("support", "acc"),
    ("evidence_pointer", "acc"),
    ("tool_intent", "acc"),
    ("idk", "acc"),
    ("verifier", "acc"),
]
METRIC_KIND = dict(METRICS)
# Eval-report (eval_key, field) for each metric. lm_loss is read at the report
# top level; the bridge heads expose both accuracy and macro_f1; proto/arg-role
# expose micro_f1.
METRIC_SOURCE: dict[str, tuple[str, str]] = {
    "primitive": ("primitive", "accuracy"),
    "primitive_macro_f1": ("primitive", "macro_f1"),
    "role_to_primitive": ("role_to_primitive", "accuracy"),
    "role_to_primitive_macro_f1": ("role_to_primitive", "macro_f1"),
    "predicate_event": ("predicate_event", "accuracy"),
    "arg_span_exact": ("arg_span_exact", "accuracy"),
    "argument_start": ("argument_start", "accuracy"),
    "proto_role_f1": ("proto_role", "micro_f1"),
    "arg_role_f1": ("arg_role", "micro_f1"),
    "support": ("support", "accuracy"),
    "evidence_pointer": ("evidence_pointer", "accuracy"),
    "tool_intent": ("tool_intent", "accuracy"),
    "idk": ("idk", "accuracy"),
    "verifier": ("verifier", "accuracy"),
}

# Which metrics each component is expected to drive (for the verdict). The
# primitive stream's defining heads are the bridge: primitive_class_logits
# (reads only the primitive registers) and role_to_primitive_logits, scored by
# both accuracy and macro-F1. idk/verifier also read pooled token state, so they
# are reported but not used to gate the primitive-stream verdict.
COMPONENT_METRICS = {
    "event": ["primitive", "role_to_primitive", "predicate_event", "arg_span_exact",
              "argument_start", "proto_role_f1", "arg_role_f1", "evidence_pointer"],
    "primitive": ["primitive", "primitive_macro_f1", "role_to_primitive", "role_to_primitive_macro_f1"],
    "ffn": [k for k, _ in METRICS],
    "pressure": ["lm_loss"],
}
# Which ablation variant(s) test each component (full-text, input-only).
COMPONENT_VARIANTS = {
    "event": ("no_event", "no_event_input_only"),
    "primitive": ("no_primitive", "no_primitive_input_only"),
    "ffn": ("no_ffn", "no_ffn_input_only"),
    "pressure": ("no_pressure", "no_pressure_input_only"),
}


def _metric_value(report: dict, key: str) -> float | None:
    if key == "lm_loss":
        return report.get("lm_loss")
    eval_key, field = METRIC_SOURCE[key]
    entry = report.get("metrics", {}).get(eval_key)
    if not isinstance(entry, dict):
        return None
    return entry.get(field)


def run_variant(name: str, overrides: dict, input_only: bool, seed: int, base: argparse.Namespace, log_dir: Path) -> dict:
    targs = T.build_arg_parser().parse_args([])
    targs.steps = base.steps
    targs.batch_size = base.batch_size
    targs.device = base.device
    targs.dtype = base.dtype
    targs.seed = seed
    targs.lr = base.lr
    targs.dataset_dir = base.dataset_dir
    targs.dataset = base.dataset
    targs.balanced_sampler = base.balanced_sampler
    targs.config = base.config
    targs.out_dir = base.ckpt_root / f"{name}_seed{seed}"
    targs.tag = f"{name}_seed{seed}"
    targs.input_only = input_only
    for flag in ("no_event_stream", "no_primitive_stream", "no_ffn_adapters", "no_pressure"):
        setattr(targs, flag, bool(overrides.get(flag, False)))

    eargs = argparse.Namespace(
        checkpoint=targs.out_dir / "latest.pt", dataset_dir=base.dataset_dir, dataset=base.dataset,
        split=base.split, device=base.device, dtype=base.dtype, batch_size=16, seed=seed,
        json_out=None, input_only=input_only, render_mode=None)

    log_path = log_dir / f"{name}_seed{seed}.log"
    with log_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        T.train(targs)
        report = E.evaluate(eargs)
    return {key: _metric_value(report, key) for key, _ in METRICS}


# ---------------------------------------------------------------------------
# Aggregation + bootstrap.
# ---------------------------------------------------------------------------
def _bootstrap_ci(values: list[float], rng: random.Random, samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (clean[0], clean[0])
    means = []
    for _ in range(samples):
        means.append(sum(clean[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * samples)], means[int(0.975 * samples)])


def _agg(values: list[float], rng: random.Random) -> dict:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"mean": None, "std": None, "ci_lo": None, "ci_hi": None, "n": 0, "values": values}
    lo, hi = _bootstrap_ci(clean, rng)
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": len(clean),
        "values": values,
    }


def _delta_favors_full(metric: str, lo: float, hi: float) -> bool:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return False
    return hi < 0 if METRIC_KIND[metric] == "loss" else lo > 0


def _delta_favors_ablation(metric: str, lo: float, hi: float) -> bool:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return False
    return lo > 0 if METRIC_KIND[metric] == "loss" else hi < 0


def compute_deltas(results: dict, seeds: list[int], rng: random.Random) -> dict:
    """For each ablation, paired per-seed delta = full - ablation, bootstrapped."""

    deltas: dict[str, dict] = {}
    for name in results:
        if name in ("full", "full_input_only"):
            continue
        ref = "full_input_only" if name.endswith("input_only") else "full"
        if ref not in results:
            continue
        per_metric = {}
        for key, _ in METRICS:
            paired = []
            for s in seeds:
                rv = results[ref][s].get(key)
                av = results[name][s].get(key)
                if rv is not None and av is not None:
                    paired.append(rv - av)
            stats = _agg(paired, rng)
            stats["favors_full"] = _delta_favors_full(key, stats["ci_lo"], stats["ci_hi"])
            stats["favors_ablation"] = _delta_favors_ablation(key, stats["ci_lo"], stats["ci_hi"])
            stats["excludes_zero"] = stats["favors_full"] or stats["favors_ablation"]
            per_metric[key] = stats
        deltas[name] = per_metric
    return deltas


def verdict(deltas: dict) -> dict:
    """Per-component yes/no/weak from CI-significant full-vs-ablation deltas."""

    out = {}
    for comp, (ft_var, io_var) in COMPONENT_VARIANTS.items():
        metrics = COMPONENT_METRICS[comp]

        def tally(var: str) -> tuple[int, int]:
            d = deltas.get(var, {})
            wins = sum(1 for m in metrics if d.get(m, {}).get("favors_full"))
            losses = sum(1 for m in metrics if d.get(m, {}).get("favors_ablation"))
            return wins, losses

        ft_wins, ft_losses = tally(ft_var)
        io_wins, io_losses = tally(io_var) if io_var in deltas else (None, None)

        if comp in ("event", "primitive"):
            # Streams need significant input-only evidence to be "supported".
            if io_wins is None:
                status = "supported (full-text)" if ft_wins >= 2 and ft_losses == 0 else (
                    "weak/noisy" if ft_wins >= 1 else "not supported")
            elif io_wins >= 2 and io_losses == 0:
                status = "supported"
            elif io_wins >= 1 and io_wins > io_losses:
                status = "weak (input-only signal present but limited)"
            elif ft_wins >= 2 and ft_losses == 0:
                status = "weak (full-text only; input-only not significant -> likely leakage-driven)"
            else:
                status = "not supported"
        else:
            total_wins = ft_wins + (io_wins or 0)
            total_losses = ft_losses + (io_losses or 0)
            status = ("supported" if total_wins >= 2 and total_losses == 0 else
                      "weak/noisy" if total_wins >= 1 else "not supported")

        out[comp] = {
            "status": status,
            "fulltext": {"variant": ft_var, "sig_full_wins": ft_wins, "sig_ablation_wins": ft_losses},
            "input_only": {"variant": io_var, "sig_full_wins": io_wins, "sig_ablation_wins": io_losses},
        }
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _fmt(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{places}f}"


def build_markdown(variant_names: list[str], agg: dict, deltas: dict, verdicts: dict, args: argparse.Namespace,
                   seeds: list[int]) -> str:
    lines = ["# PAT-ER ablation gate (multi-seed)", ""]
    lines.append(f"seeds={seeds} steps={args.steps} batch_size={args.batch_size} device={args.device} "
                 f"dtype={args.dtype} split={args.split} balanced_sampler={bool(getattr(args, 'balanced_sampler', False))}")
    ds = getattr(args, "dataset", None)
    if ds:
        lines.append(f"dataset={', '.join(Path(p).name for p in ds)}")
    lines.append("")
    lines.append("Core variants keep the serialized event_graph in the model input (**supervised-plumbing**). "
                 "`*_input_only` variants strip it and are the stronger architecture evidence. "
                 "Cells are mean±std over seeds; CIs are on the full−ablation deltas below.")
    lines.append("")

    # Mean±std table: metric rows x variant columns.
    lines.append("| metric | " + " | ".join(variant_names) + " |")
    lines.append("|" + "---|" * (len(variant_names) + 1))
    for key, _ in METRICS:
        row = [key]
        for name in variant_names:
            a = agg[name][key]
            row.append("n/a" if a["mean"] is None else f"{a['mean']:.2f}±{a['std']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Delta tables (full - ablation) with bootstrap 95% CI; ** = CI excludes 0 favoring full.
    lines.append("## Deltas: full − ablation (mean [95% CI]); ** favors full (CI excludes 0), ~~ favors ablation")
    lines.append("")
    for name in variant_names:
        if name not in deltas:
            continue
        ref = "full_input_only" if name.endswith("input_only") else "full"
        lines.append(f"### {ref} − {name}")
        lines.append("| metric | delta | 95% CI | significant |")
        lines.append("|---|---|---|---|")
        for key, _ in METRICS:
            d = deltas[name][key]
            mark = "**full**" if d["favors_full"] else ("~~ablation" if d["favors_ablation"] else "ns")
            lines.append(f"| {key} | {_fmt(d['mean'])} | [{_fmt(d['ci_lo'])}, {_fmt(d['ci_hi'])}] | {mark} |")
        lines.append("")

    # Verdict.
    lines.append("## Verdict by component")
    lines.append("")
    lines.append("| component | status | full-text sig (full/abl) | input-only sig (full/abl) |")
    lines.append("|---|---|---|---|")
    label = {"event": "event-role stream", "primitive": "primitive stream",
             "ffn": "role/primitive FFN adapters", "pressure": "pressure logits"}
    for comp in ("event", "primitive", "ffn", "pressure"):
        v = verdicts[comp]
        ft = v["fulltext"]; io = v["input_only"]
        io_str = "n/a" if io["sig_full_wins"] is None else f"{io['sig_full_wins']}/{io['sig_ablation_wins']}"
        lines.append(f"| {label[comp]} | **{v['status']}** | {ft['sig_full_wins']}/{ft['sig_ablation_wins']} | {io_str} |")
    lines.append("")
    lines.append("`sig` counts metrics whose full−ablation 95% CI excludes 0 (full wins / ablation wins) over that "
                 "component's relevant metrics. Streams require input-only significance to be fully \"supported\".")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-seed PAT-ER ablation gate.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "artifacts" / "datasets" / "pater_synthetic")
    parser.add_argument("--dataset", type=Path, nargs="+", default=None,
                        help="Explicit JSONL file(s) to train/eval on; overrides --dataset-dir.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "tiny_smoke.yaml")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--report-name", type=str, default="ablation_gate_multiseed",
                        help="Basename for the saved {json,md} report and per-run checkpoint/log subdir.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2.")
    parser.add_argument("--num-seeds", type=int, default=None, help="Alternative to --seeds: use 0..num-1.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated base variants (full,no_event,no_primitive,no_ffn,no_pressure); "
                             "default runs the full-text + input-only matrix.")
    parser.add_argument("--input-only", action="store_true",
                        help="Run the selected --variants under input-only rendering only (no full-text variants).")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Class-balanced batches + inverse-freq bridge class weights (applied to every variant).")
    parser.add_argument("--full-input-only-matrix", action="store_true",
                        help="Run all 5 ablations under input-only (not just full/no_event/no_primitive).")
    parser.add_argument("--skip-input-only", action="store_true", help="Run only the full-text variants.")
    args = parser.parse_args()

    seeds = list(range(args.num_seeds)) if args.num_seeds else [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    if args.variants:
        bases = [b.strip() for b in args.variants.split(",") if b.strip()]
        table = _INPUT_ONLY_BY_BASE if args.input_only else _FULLTEXT_BY_BASE
        if "full" not in bases:
            bases = ["full"] + bases  # the paired delta reference must be present
        unknown = [b for b in bases if b not in table]
        if unknown:
            raise SystemExit(f"unknown variant(s) {unknown}; choose from {sorted(table)}")
        variants = [table[b] for b in bases]
    else:
        variants = list(FULLTEXT_VARIANTS)
        if not args.skip_input_only:
            variants += INPUT_ONLY_FULL if args.full_input_only_matrix else INPUT_ONLY_DEFAULT
    variant_names = [v[0] for v in variants]

    args.ckpt_root = DEFAULT_CKPT_ROOT / args.report_name
    report_dir = Path(args.report_dir)
    log_dir = report_dir / f"{args.report_name}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"PAT-ER ablation gate (multi-seed): {len(variants)} variants x {len(seeds)} seeds "
          f"= {len(variants) * len(seeds)} runs | steps={args.steps} device={args.device} split={args.split}")

    results: dict[str, dict[int, dict]] = {name: {} for name in variant_names}
    for seed in seeds:
        for name, overrides, input_only in variants:
            print(f"  seed {seed} :: {name} (input_only={input_only}) ...", flush=True)
            results[name][seed] = run_variant(name, overrides, input_only, seed, args, log_dir)

    rng = random.Random(BOOTSTRAP_RNG_SEED)
    agg = {name: {key: _agg([results[name][s].get(key) for s in seeds], rng) for key, _ in METRICS}
           for name in variant_names}
    deltas = compute_deltas(results, seeds, rng)
    verdicts = verdict(deltas)

    report = {
        "config": {"seeds": seeds, "steps": args.steps, "batch_size": args.batch_size, "device": args.device,
                   "dtype": args.dtype, "split": args.split, "bootstrap_samples": BOOTSTRAP_SAMPLES,
                   "balanced_sampler": bool(args.balanced_sampler), "input_only": bool(args.input_only),
                   "dataset": [str(p) for p in args.dataset] if args.dataset else None},
        "metrics_order": [k for k, _ in METRICS],
        "variants": variant_names,
        "aggregate": agg,
        "deltas": deltas,
        "verdict": verdicts,
        "raw": {name: {str(s): results[name][s] for s in seeds} for name in variant_names},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{args.report_name}.json"
    md_path = report_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = build_markdown(variant_names, agg, deltas, verdicts, args, seeds)
    md_path.write_text(md, encoding="utf-8")

    print(f"\nwrote {_rel(json_path)} and {_rel(md_path)}\n")
    print(md)


if __name__ == "__main__":
    main()
