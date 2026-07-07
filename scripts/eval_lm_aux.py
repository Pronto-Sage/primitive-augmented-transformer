#!/usr/bin/env python3
"""Evaluate a tiny supervised PAT-ER checkpoint (LM + auxiliary + event-role heads).

Loads a checkpoint written by train_lm_aux.py (model state + tokenizer vocab +
config), rebuilds the model and tokenizer, and reports on a dataset split:

    LM loss (token-weighted);
    categorical heads      acc vs random and empirical-majority baselines
      (primitive, support, tool_intent, schema, idk, verifier, role_to_primitive,
       role_ambiguity, predicate_event);
    pointer heads          acc (event_token, argument_start/end, evidence_pointer);
    argument span exact-match accuracy;
    event-arg link accuracy;
    proto_role / arg_role  micro-F1 (+ precision/recall).

Heads with no labels on the split report n/a. CPU/GPU-safe; no downloads.

Usage:
    python3 scripts/eval_lm_aux.py --checkpoint artifacts/checkpoints/tiny_aux_overfit/latest.pt --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

import torch

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import maybe_autocast

import train_lm_aux as T
import pater_hf_tokenizer as H

DEFAULT_DATASET_DIR = ROOT / "artifacts" / "datasets" / "pater_synthetic"

CATEGORICAL_HEADS = [
    "primitive", "support", "tool_intent", "schema", "idk", "verifier",
    "role_to_primitive", "role_ambiguity", "predicate_event",
]
# Heads considered for the event-role architecture gate (categorical/structural).
GATE_HEADS = ["predicate_event", "event_token", "argument_start", "argument_end",
              "event_arg", "role_to_primitive", "role_ambiguity"]

# How to read each categorical head's gold label off a record, for the
# empirical-majority baseline.
_MAJORITY_LABEL = {
    "primitive": lambda r: r["labels"].get("primitive_class"),
    "role_to_primitive": lambda r: r["labels"].get("primitive_class"),
    "support": lambda r: r["labels"].get("support_status"),
    "tool_intent": lambda r: r["labels"].get("tool_intent"),
    "schema": lambda r: bool(r["labels"].get("schema_validity")),
    "idk": lambda r: r["labels"].get("idk_action"),
    "verifier": lambda r: bool(r["labels"].get("verifier_accept")),
    "role_ambiguity": lambda r: r["labels"].get("role_ambiguity"),
    "predicate_event": lambda r: r["labels"].get("event_type") if r.get("event_graph") else None,
}


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return requested


def load_checkpoint(path: Path, device: str, tokenizer_override: Path | None = None):
    # Checkpoints contain only tensors + simple types (Paths sanitized to str),
    # so weights_only=True is safe.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = PATERConfig.from_dict(checkpoint["config"])
    # Prefer an explicit override, then the saved HF tokenizer path, then the
    # reference vocab embedded in the checkpoint.
    tokenizer_path = str(tokenizer_override) if tokenizer_override else checkpoint.get("tokenizer_path")
    if tokenizer_path:
        tokenizer = H.load_pater_hf_tokenizer(tokenizer_path)
    else:
        tokenizer = T.reconstruct_tokenizer(checkpoint["vocab"])
    model = PATERForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    max_len = int(checkpoint.get("max_len", min(config.max_position_embeddings, 128)))
    return model, tokenizer, config, max_len, checkpoint


def majority_baseline(records, head: str) -> float | None:
    getter = _MAJORITY_LABEL.get(head)
    if getter is None:
        return None
    counts = Counter(getter(r) for r in records if getter(r) is not None)
    total = sum(counts.values())
    return (counts.most_common(1)[0][1] / total) if total else None


@torch.no_grad()
def _width_bucket(width: int) -> str:
    if width <= 1:
        return "w1"
    if width <= 4:
        return "w2-4"
    if width <= 15:
        return "w5-15"
    return "w16+"


@torch.no_grad()
def span_breakdown(model, records, tokenizer, config, max_len, render_mode,
                   device, dtype, max_width) -> dict:
    """Per-span start/end/exact accuracy, split by source origin, span width, and
    single- vs multi-subword. Independent decode and boundary-aware joint decode.

    Aligns batch row b -> records[i+b] and arg slot a -> the a-th argument, the
    same order collate() builds targets in, so each labeled span carries its
    record's mix_origin and its gold token width.
    """

    dims = T.dims_from_config(config)

    def _empty() -> dict:
        return {"start": 0, "end": 0, "exact_indep": 0, "exact_joint": 0, "n": 0}

    groups = {"by_origin": {}, "by_width": {}, "by_subword": {}}
    model.eval()
    for i in range(0, len(records), 8):
        sub = records[i:i + 8]
        batch = T.collate(tokenizer, sub, max_len, dims, render_mode).to(device)
        with maybe_autocast(device, dtype):
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, return_aux=True)
        start_logits = out.aux_outputs["argument_start_logits"]
        end_logits = out.aux_outputs["argument_end_logits"]
        ps = start_logits.argmax(dim=-1)
        pe = end_logits.argmax(dim=-1)
        js, je = T.joint_span_decode(start_logits, end_logits, max_width)
        gs, ge = batch.er["arg_start"], batch.er["arg_end"]
        valid = (gs != -100) & (ge != -100)
        for b in range(gs.shape[0]):
            origin = (sub[b].get("provenance") or {}).get("mix_origin") or "unknown"
            for a in range(gs.shape[1]):
                if not bool(valid[b, a]):
                    continue
                g_s, g_e = int(gs[b, a]), int(ge[b, a])
                width = g_e - g_s + 1
                s_ok = int(ps[b, a]) == g_s
                e_ok = int(pe[b, a]) == g_e
                ei = s_ok and e_ok
                ej = (int(js[b, a]) == g_s) and (int(je[b, a]) == g_e)
                for group, key in (("by_origin", origin), ("by_width", _width_bucket(width)),
                                   ("by_subword", "single" if width == 1 else "multi")):
                    c = groups[group].setdefault(key, _empty())
                    c["start"] += s_ok
                    c["end"] += e_ok
                    c["exact_indep"] += ei
                    c["exact_joint"] += ej
                    c["n"] += 1
    return groups


def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint {checkpoint_path} not found. Run scripts/train_lm_aux.py first.")
    model, tokenizer, config, max_len, checkpoint = load_checkpoint(
        checkpoint_path, device, getattr(args, "tokenizer", None))
    dims = T.dims_from_config(config)
    render_mode = checkpoint.get("render_mode") or ("input_only" if checkpoint.get("input_only") else "full")
    # Optional override (must match how the checkpoint was trained to be meaningful).
    override = getattr(args, "render_mode", None) or ("input_only" if getattr(args, "input_only", False) else None)
    if override and override != render_mode:
        print(f"note: overriding checkpoint render_mode={render_mode} with --render-mode {override}")
        render_mode = override

    split = None if args.split == "all" else args.split
    # Prefer explicit --dataset, then the dataset the checkpoint was trained on,
    # then --dataset-dir. This keeps eval on the same data as training.
    dataset_files = getattr(args, "dataset", None) or checkpoint.get("dataset")
    if dataset_files:
        records = T.load_record_files([Path(p) for p in dataset_files], split=split)
        source_desc = ", ".join(str(p) for p in dataset_files)
    else:
        records = T.load_records(Path(args.dataset_dir), split=split, auto_build=False)
        source_desc = str(args.dataset_dir)
    if not records:
        raise RuntimeError(f"no records for split={args.split!r} in {source_desc}")

    acc = T.MetricAccumulator.empty()
    mean_len = 0.0
    for i in range(0, len(records), args.batch_size):
        sub = records[i:i + args.batch_size]
        batch = T.collate(tokenizer, sub, max_len, dims, render_mode).to(device)
        mean_len += float(batch.attention_mask.sum())
        with maybe_autocast(device, args.dtype):
            output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                           labels=batch.input_ids, return_aux=True,
                           evidence_item_mask=batch.er.get("evidence_item_mask"))
        acc.update(output, batch)
    mean_len /= max(1, len(records))
    pointer_random = 1.0 / mean_len if mean_len else 0.0

    print("PAT-ER tiny supervised eval")
    print(f"checkpoint: {checkpoint_path}")
    tok_desc = str(getattr(args, "tokenizer", None) or checkpoint.get("tokenizer_path") or "reference")
    print(f"device={device} dtype={args.dtype} split={args.split} records={len(records)} "
          f"trained_steps={checkpoint.get('step')} render_mode={render_mode} tokenizer={tok_desc} "
          f"aux_from_token_state={config.aux_from_token_state} "
          f"generic_register_stream={getattr(config, 'generic_register_stream', False)}")
    print(f"LM loss: {acc.lm_loss():.4f}")
    report: dict = {
        "split": args.split,
        "records": len(records),
        "lm_loss": acc.lm_loss(),
        "condition_flags": {
            "aux_from_token_state": bool(config.aux_from_token_state),
            "generic_register_stream": bool(getattr(config, "generic_register_stream", False)),
            "use_event_stream": bool(config.use_event_stream),
            "use_primitive_stream": bool(config.use_primitive_stream),
            "use_role_primitive_ffn": bool(config.use_role_primitive_ffn),
            "use_vocab_pressure": bool(config.use_vocab_pressure),
        },
        "metrics": {},
    }

    print("categorical heads (acc | random | majority):")
    for name in CATEGORICAL_HEADS:
        a = acc.accuracy(name)
        if a is None:
            print(f"  {name:<18} n/a")
            report["metrics"][name] = None
            continue
        rand = T.RANDOM_BASELINE.get(name)
        maj = majority_baseline(records, name)
        beats = a > max(rand or 0.0, maj or 0.0) + 1e-9
        print(f"  {name:<18} acc={a:.3f}  n={acc.total[name]:<4} random={rand:.3f} majority={(maj or 0):.3f}"
              f"  {'[above baseline]' if beats else '[at/below baseline]'}")
        report["metrics"][name] = {"accuracy": a, "n": acc.total[name], "random": rand, "majority": maj}

    # Reasoning-supervision heads (only present when --use-reasoning-heads). Reported
    # with per-class recall for entailment_state (entailed/refuted/unknown).
    for name in ("entailment_state", "proof_depth", "rule_chain_length"):
        a = acc.accuracy(name)
        if a is None:
            continue
        entry = {"accuracy": a, "n": acc.total[name]}
        if name == "entailment_state":
            pcr = acc.per_class_recall(name, list(T.ENTAILMENT_STATES))
            pprf = acc.per_class_prf(name, list(T.ENTAILMENT_STATES))
            entry["per_class_recall"] = {k: f"{c}/{t}" for k, (c, t) in pcr.items()}
            entry["per_class_prf"] = pprf
        report["metrics"][name] = entry
        msg = f"  {name:<18} acc={a:.3f}  n={acc.total[name]}"
        if name == "entailment_state":
            ref = entry.get("per_class_prf", {}).get("refuted")
            if ref:
                msg += f"  refuted P={ref['precision']:.3f} R={ref['recall']:.3f} F1={ref['f1']:.3f}"
            msg += f"  per_class={entry.get('per_class_recall')}"
        print(msg)
    for name in ("supporting_fact", "contradicting_fact", "supporting_item", "contradicting_item"):
        f1 = acc.micro_f1(name)
        if f1 is not None:
            report["metrics"][name] = {"micro_f1": f1[0], "precision": f1[1], "recall": f1[2]}
            print(f"  {name:<18} F1={f1[0]:.3f}  P={f1[1]:.3f}  R={f1[2]:.3f}")

    # Bridge heads: macro-F1, per-class recall, and the worst confusions.
    print("bridge heads (macro-F1 over 14 primitive classes):")
    for name in ("primitive", "role_to_primitive"):
        mf1 = acc.macro_f1(name)
        a = acc.accuracy(name)
        if mf1 is None:
            print(f"  {name:<18} n/a")
            report["metrics"].setdefault(name, {})
            continue
        per_class = acc.per_class_recall(name, T.builder.PRIMITIVE_CLASSES)
        per_class_prf = acc.per_class_prf(name, T.builder.PRIMITIVE_CLASSES)
        confs = acc.top_confusions(name, T.builder.PRIMITIVE_CLASSES, k=4)
        report["metrics"].setdefault(name, {})
        report["metrics"][name].update({"macro_f1": mf1, "accuracy": a,
                                        "per_class_prf": per_class_prf,
                                        "per_class_recall": {k: f"{c}/{t}" for k, (c, t) in per_class.items()},
                                        "top_confusions": [f"{g}->{p}:{n}" for g, p, n in confs]})
        worst = sorted(per_class.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))[:4]
        print(f"  {name:<18} macroF1={mf1:.3f}  acc={a:.3f}")
        contra = per_class_prf.get("contradiction")
        if contra:
            print("      contradiction: "
                  f"P={contra['precision']:.3f} R={contra['recall']:.3f} F1={contra['f1']:.3f} "
                  f"support={contra['support']} predicted={contra['predicted']}")
        print(f"      worst classes: " + ", ".join(f"{k} {c}/{t}" for k, (c, t) in worst))
        print(f"      top confusions: " + (", ".join(f"{g}->{p}({n})" for g, p, n in confs) or "none"))

    print(f"pointer heads (acc | random pointer ~= {pointer_random:.3f} over ~{mean_len:.0f} tokens):")
    for name in ("event_token", "argument_start", "argument_end", "evidence_pointer"):
        a = acc.accuracy(name)
        if a is None:
            print(f"  {name:<18} n/a")
            report["metrics"][name] = None
            continue
        print(f"  {name:<18} acc={a:.3f}  n={acc.total[name]}")
        report["metrics"][name] = {"accuracy": a, "n": acc.total[name], "random_pointer": pointer_random}

    span_labels = {
        "argument_start_any": "start any-occ", "argument_end_any": "end any-occ",
        "arg_span_exact": "exact-span (strict,indep)", "arg_span_joint": "exact-span (strict,joint)",
        "arg_span_any": "exact-span (any-occ,indep)", "arg_span_joint_any": "exact-span (any-occ,joint)",
        "event_arg": "link element acc",
    }
    for name, label in span_labels.items():
        a = acc.accuracy(name)
        if a is None:
            print(f"  {name:<20} n/a")
            report["metrics"][name] = None
        else:
            print(f"  {name:<20} {label}={a:.3f}  n={acc.total[name]}")
            report["metrics"][name] = {"accuracy": a, "n": acc.total[name]}

    # Span breakdown by origin / width / subword-count (start vs end vs exact).
    breakdown = span_breakdown(model, records, tokenizer, config, max_len, render_mode,
                               device, args.dtype, T.SPAN_MAX_WIDTH)
    report["span_breakdown"] = breakdown
    print("span breakdown (start | end | exact-indep | exact-joint | n):")
    for group in ("by_origin", "by_width", "by_subword"):
        for key, c in breakdown[group].items():
            n = c["n"]
            if not n:
                continue
            print(f"  {group[3:]:>8}={key:<10} start={c['start']/n:.3f} end={c['end']/n:.3f} "
                  f"exact={c['exact_indep']/n:.3f} joint={c['exact_joint']/n:.3f}  n={n}")

    print("multi-label heads (micro-F1 | P | R):")
    for name in T.F1_HEADS:
        f1 = acc.micro_f1(name)
        if f1 is None:
            print(f"  {name:<18} n/a")
            report["metrics"][name] = None
        else:
            print(f"  {name:<18} F1={f1[0]:.3f}  P={f1[1]:.3f}  R={f1[2]:.3f}")
            report["metrics"][name] = {"micro_f1": f1[0], "precision": f1[1], "recall": f1[2]}

    # Gates.
    prev_heads = ["primitive", "support", "tool_intent", "idk"]
    prev_pass = all((acc.accuracy(h) or 0.0) > T.RANDOM_BASELINE[h] for h in prev_heads if acc.total.get(h, 0) > 0)
    er_checks = []
    for h in GATE_HEADS:
        a = acc.accuracy(h)
        if a is None or acc.total.get(h, 0) == 0:
            continue
        base = T.RANDOM_BASELINE.get(h, pointer_random)
        er_checks.append(a > base + 1e-9)
    er_f1_learn = all((acc.micro_f1(h) or (0,))[0] > 0 for h in T.F1_HEADS if acc.micro_f1(h) is not None)
    er_pass = bool(er_checks) and all(er_checks) and er_f1_learn

    print(f"\nprev-gate (primitive/support/tool/IDK > random): {'PASS' if prev_pass else 'not yet'}")
    print(f"event-role gate (event/predicate/arg/role heads > random, proto/arg-role F1>0): "
          f"{'PASS' if er_pass else 'not yet'}")
    report["prev_gate_pass"] = bool(prev_pass)
    report["event_role_gate_pass"] = bool(er_pass)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote metrics: {args.json_out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a tiny supervised PAT-ER checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "checkpoints" / "tiny_aux_overfit" / "latest.pt")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset", type=Path, nargs="+", default=None,
                        help="Explicit JSONL file(s) to evaluate on; default reuses the checkpoint's training dataset.")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="Override the tokenizer dir (default reuses the checkpoint's HF tokenizer path / reference vocab).")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--input-only", action="store_true", help="Override checkpoint render_mode to input_only.")
    parser.add_argument("--render-mode", choices=["full", "input_only", "plain", "prefix"], default=None,
                        help="Override the checkpoint's render mode (must match training to be meaningful).")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
