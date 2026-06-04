#!/usr/bin/env python3
"""PAT-ER baseline gate: does true side-state beat simpler non-side-state models?

Compares full PAT-ER against three non-side-state baselines under matched
conditions (same steps/dataset/split, multiple seeds), with bootstrap 95% CIs on
the paired full-minus-baseline deltas.

Variants (input-only is the main, lower-leakage comparison):

    full_pater          input_only render; true event/primitive side-state;
                        aux heads read the side-state registers.
    base_decoder        plain text only; no streams/adapters/pressure;
                        aux heads read pooled token state.
    base_control_vocab  input_only render (PAT-ER control vocabulary) but no
                        side-state; aux heads read pooled token state.
    prefix_register     input_only render + a fixed register-token prefix
                        (compatibility registers as input tokens); no true
                        side-state; aux heads read pooled token state.

The key test is full_pater vs prefix_register on the event-role metrics: if true
side-state does not beat compatibility prefix registers, the side-state
architecture is not yet justified over a simpler register encoding.

Outputs: artifacts/reports/baseline_gate.{json,md} (git-ignored). CPU/GPU-safe;
no 770M; no downloads.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_lm_aux as T
import eval_lm_aux as E
import run_ablation_gate as A  # reuse METRICS, _agg, _bootstrap_ci, _delta_favors_*, _metric_value, _fmt

DEFAULT_REPORT_DIR = ROOT / "artifacts" / "reports"
DEFAULT_CKPT_ROOT = ROOT / "artifacts" / "checkpoints" / "baseline"

EVENT_ROLE_METRICS = A.COMPONENT_METRICS["event"]  # primitive, role_to_primitive, predicate_event,
#                                                    arg_span_exact, proto_role_f1, arg_role_f1, evidence_pointer
BASE_OFF = {"no_event_stream": True, "no_primitive_stream": True, "no_ffn_adapters": True, "no_pressure": True}

# (name, ablation overrides, render_mode, aux_from_token_state)
INPUT_ONLY_VARIANTS = [
    ("full_pater", {}, "input_only", False),
    ("base_decoder", BASE_OFF, "plain", True),
    ("base_control_vocab", BASE_OFF, "input_only", True),
    ("prefix_register", BASE_OFF, "prefix", True),
]
# Weaker full-text counterpart (event_graph leaked in the input) if requested.
FULLTEXT_VARIANTS = [
    ("full_pater_ft", {}, "full", False),
    ("base_decoder_ft", BASE_OFF, "plain", True),
    ("base_control_vocab_ft", BASE_OFF, "full", True),
    ("prefix_register_ft", BASE_OFF, "full", True),  # prefix over full-text is redundant; use full
]
REFERENCE = {"full_pater", "full_pater_ft"}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_variant(name: str, overrides: dict, render_mode: str, aux_from_token_state: bool,
                seed: int, base: argparse.Namespace, log_dir: Path) -> dict:
    targs = T.build_arg_parser().parse_args([])
    targs.steps = base.steps
    targs.batch_size = base.batch_size
    targs.device = base.device
    targs.dtype = base.dtype
    targs.seed = seed
    targs.lr = base.lr
    targs.dataset_dir = base.dataset_dir
    targs.config = base.config
    targs.out_dir = DEFAULT_CKPT_ROOT / f"{name}_seed{seed}"
    targs.tag = f"{name}_seed{seed}"
    targs.render_mode = render_mode
    targs.aux_from_token_state = aux_from_token_state
    for flag in ("no_event_stream", "no_primitive_stream", "no_ffn_adapters", "no_pressure"):
        setattr(targs, flag, bool(overrides.get(flag, False)))

    eargs = argparse.Namespace(
        checkpoint=targs.out_dir / "latest.pt", dataset_dir=base.dataset_dir, split=base.split,
        device=base.device, dtype=base.dtype, batch_size=16, seed=seed, json_out=None)

    log_path = log_dir / f"{name}_seed{seed}.log"
    with log_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        T.train(targs)
        report = E.evaluate(eargs)
    return {key: A._metric_value(report, key) for key, _ in A.METRICS}


def compute_deltas(results: dict, ref_name: str, seeds: list[int], rng: random.Random) -> dict:
    deltas: dict[str, dict] = {}
    for name in results:
        if name == ref_name:
            continue
        per_metric = {}
        for key, _ in A.METRICS:
            paired = []
            for s in seeds:
                rv = results[ref_name][s].get(key)
                av = results[name][s].get(key)
                if rv is not None and av is not None:
                    paired.append(rv - av)
            stats = A._agg(paired, rng)
            stats["favors_full"] = A._delta_favors_full(key, stats["ci_lo"], stats["ci_hi"])
            stats["favors_ablation"] = A._delta_favors_ablation(key, stats["ci_lo"], stats["ci_hi"])
            per_metric[key] = stats
        deltas[name] = per_metric
    return deltas


def verdict(deltas: dict, baseline_names: list[str]) -> dict:
    out = {}
    for name in baseline_names:
        d = deltas.get(name, {})
        wins = [m for m in EVENT_ROLE_METRICS if d.get(m, {}).get("favors_full")]
        losses = [m for m in EVENT_ROLE_METRICS if d.get(m, {}).get("favors_ablation")]
        if len(wins) >= 2 and not losses:
            status = "full beats (CI excludes 0)"
        elif wins and len(wins) > len(losses):
            status = "full ahead (weak)"
        elif not wins and not losses:
            status = "indistinguishable"
        else:
            status = "mixed / not ahead"
        out[name] = {"status": status, "event_role_wins": wins, "event_role_losses": losses}
    return out


def build_markdown(names: list[str], agg: dict, deltas: dict, verdicts: dict, ref_name: str,
                   args: argparse.Namespace, seeds: list[int]) -> str:
    lines = ["# PAT-ER baseline gate (multi-seed)", ""]
    lines.append(f"reference={ref_name} seeds={seeds} steps={args.steps} batch_size={args.batch_size} "
                 f"device={args.device} dtype={args.dtype} split={args.split}")
    lines.append("")
    lines.append("Input-only is the main comparison (event_graph stripped from the model input). Baselines have no "
                 "side-state: their aux heads read pooled token state. `prefix_register` adds a fixed register-token "
                 "prefix as ordinary input tokens. Cells are mean±std over seeds; CIs are on the deltas below.")
    lines.append("")

    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for key, _ in A.METRICS:
        row = [key]
        for name in names:
            a = agg[name][key]
            row.append("n/a" if a["mean"] is None else f"{a['mean']:.2f}±{a['std']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"## Deltas: {ref_name} − baseline (mean [95% CI]); ** favors full, ~~ favors baseline")
    lines.append("")
    for name in names:
        if name not in deltas:
            continue
        lines.append(f"### {ref_name} − {name}")
        lines.append("| metric | delta | 95% CI | significant |")
        lines.append("|---|---|---|---|")
        for key, _ in A.METRICS:
            dd = deltas[name][key]
            mark = "**full**" if dd["favors_full"] else ("~~baseline" if dd["favors_ablation"] else "ns")
            lines.append(f"| {key} | {A._fmt(dd['mean'])} | [{A._fmt(dd['ci_lo'])}, {A._fmt(dd['ci_hi'])}] | {mark} |")
        lines.append("")

    lines.append("## Verdict: does true side-state beat simpler baselines?")
    lines.append("")
    lines.append("| comparison | event-role status | full wins (CI) | full losses (CI) |")
    lines.append("|---|---|---|---|")
    for name in [n for n in names if n != ref_name]:
        v = verdicts[name]
        lines.append(f"| {ref_name} vs {name} | **{v['status']}** | {', '.join(v['event_role_wins']) or '-'} | "
                     f"{', '.join(v['event_role_losses']) or '-'} |")
    lines.append("")
    prefix_key = "prefix_register" if "prefix_register" in verdicts else None
    if prefix_key:
        pv = verdicts[prefix_key]
        justified = pv["status"].startswith("full beats")
        lines.append(f"**Side-state justified over prefix_register?** "
                     f"{'YES — full beats prefix registers on event-role metrics with CI excluding 0.' if justified else 'NOT YET — full does not significantly beat prefix registers on event-role metrics.'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PAT-ER baseline gate (full vs base/control/prefix).")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "artifacts" / "datasets" / "pater_synthetic")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "tiny_smoke.yaml")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--input-only", action="store_true",
                        help="Main comparison: input-only rendering (default behavior; flag accepted for clarity).")
    parser.add_argument("--include-full-text", action="store_true",
                        help="Also run the weaker full-text comparison (event_graph leaked into the input).")
    args = parser.parse_args()

    seeds = list(range(args.num_seeds)) if args.num_seeds else [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    variants = list(INPUT_ONLY_VARIANTS)
    ref_name = "full_pater"
    if args.include_full_text:
        variants = variants + FULLTEXT_VARIANTS
    names = [v[0] for v in variants]

    report_dir = Path(args.report_dir)
    log_dir = report_dir / "baseline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"PAT-ER baseline gate: {len(variants)} variants x {len(seeds)} seeds = {len(variants) * len(seeds)} runs "
          f"| steps={args.steps} device={args.device} split={args.split}")
    results: dict[str, dict[int, dict]] = {name: {} for name in names}
    for seed in seeds:
        for name, overrides, render_mode, aux_pool in variants:
            print(f"  seed {seed} :: {name} (render={render_mode}, aux_from_token_state={aux_pool}) ...", flush=True)
            results[name][seed] = run_variant(name, overrides, render_mode, aux_pool, seed, args, log_dir)

    rng = random.Random(A.BOOTSTRAP_RNG_SEED)
    agg = {name: {key: A._agg([results[name][s].get(key) for s in seeds], rng) for key, _ in A.METRICS}
           for name in names}

    # Input-only deltas reference full_pater; full-text deltas reference full_pater_ft.
    io_names = [v[0] for v in INPUT_ONLY_VARIANTS]
    deltas = compute_deltas({n: results[n] for n in io_names}, "full_pater", seeds, rng)
    if args.include_full_text:
        ft_names = [v[0] for v in FULLTEXT_VARIANTS]
        deltas.update(compute_deltas({n: results[n] for n in ft_names}, "full_pater_ft", seeds, rng))

    baseline_names = [n for n in io_names if n != "full_pater"]
    verdicts = verdict(deltas, baseline_names)

    report = {
        "config": {"seeds": seeds, "steps": args.steps, "batch_size": args.batch_size, "device": args.device,
                   "dtype": args.dtype, "split": args.split, "reference": ref_name},
        "metrics_order": [k for k, _ in A.METRICS],
        "variants": names,
        "aggregate": agg,
        "deltas": deltas,
        "verdict": verdicts,
        "raw": {name: {str(s): results[name][s] for s in seeds} for name in names},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "baseline_gate.json"
    md_path = report_dir / "baseline_gate.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = build_markdown(names, agg, deltas, verdicts, ref_name, args, seeds)
    md_path.write_text(md, encoding="utf-8")

    print(f"\nwrote {_rel(json_path)} and {_rel(md_path)}\n")
    print(md)


if __name__ == "__main__":
    main()
