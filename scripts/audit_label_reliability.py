#!/usr/bin/env python3
"""Offline reliability audit for PAT-ER converted/synthetic supervision.

The audit is intentionally deterministic. It does not ask an LLM to judge
labels; it checks the supervision pipeline invariants that reviewers can
inspect:

  * primitive/support labels are in the declared vocabularies;
  * event_graph primitive/support tokens agree with record labels when the
    record-level primitive is a reasoning primitive;
  * input-only rendering does not leak proof metadata / chain-of-thought fields;
  * argument spans, predicate tokens, and evidence ids relocate in the rendered
    input under the active tokenizer;
  * provenance/source metadata is present.

Usage:
    python3 scripts/audit_label_reliability.py \
      --dataset artifacts/datasets/mixed/pater_mixed_reason_big.jsonl \
      --tokenizer artifacts/tokenizers/qwen_pater_extended \
      --sample-per-source 60 \
      --output-md artifacts/reports/label_reliability_audit.md \
      --output-json artifacts/reports/label_reliability_audit.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pat_er import PATERConfig

import build_synthetic_pater_dataset as builder
import pater_hf_tokenizer as H
import train_lm_aux as T


FORBIDDEN_INPUT_KEYS = (
    "proofsWithIntermediates",
    "proofs",
    "QDep",
    "supporting_fact_ids",
    "contradicting_fact_ids",
    "supporting_fact_texts",
    "contradicting_fact_texts",
    "rule_chain_length",
    "proof_depth",
    "inv-proof",
)

# These record-level primitive classes supervise output/interface behavior rather
# than the event-local reasoning primitive. For example, a tool record can carry
# primitive_class="tool" while its event node is an observation licensed by
# evidence. Counting that as a primitive mismatch would conflate two label
# layers, so the audit reports it separately instead of failing consistency.
EVENT_LOCAL_PRIMITIVE_EXEMPTIONS = {"tool", "schema", "provenance"}


def load_records(paths: list[Path], split: str) -> list[dict[str, Any]]:
    wanted = None if split == "all" else split
    return T.load_record_files(paths, split=wanted)


def source_bucket(record: dict[str, Any]) -> str:
    prov = record.get("provenance") or {}
    return str(
        prov.get("mix_origin")
        or prov.get("source")
        or record.get("source")
        or record.get("family")
        or "unknown"
    )


def token_to_label(token: str, prefix: str) -> str | None:
    if not isinstance(token, str):
        return None
    if token.startswith(prefix) and token.endswith(">"):
        return token[len(prefix):-1]
    return None


def event_consistency(record: dict[str, Any]) -> tuple[bool, list[str], bool]:
    labels = record.get("labels") or {}
    events = record.get("event_graph") or []
    if not events:
        return True, [], False
    ev = events[0]
    failures: list[str] = []
    prim = ev.get("primitive")
    support = ev.get("support")
    prim_label = labels.get("primitive_class")
    support_label = labels.get("support_status")
    exempt_event_local = prim_label in EVENT_LOCAL_PRIMITIVE_EXEMPTIONS

    prim_from_token = token_to_label(prim, "<prim:")
    reg_from_token = token_to_label(prim, "<reg:")
    if (not exempt_event_local) and prim_from_token is not None and prim_label is not None and prim_from_token != prim_label:
        failures.append(f"primitive token {prim} != label {prim_label}")
    if (not exempt_event_local) and reg_from_token is not None and prim_label is not None and reg_from_token != prim_label:
        failures.append(f"register token {prim} != label {prim_label}")

    support_from_token = token_to_label(support, "<support:")
    if support_from_token is not None and support_label is not None and support_from_token != support_label:
        failures.append(f"support token {support} != label {support_label}")
    return not failures, failures, exempt_event_local


def target_recovery(record: dict[str, Any], tokenizer: Any, dims: T.Dims, render_mode: str) -> dict[str, int]:
    enc = T.encode_for_targets(tokenizer, record, max_len=100_000, render_mode=render_mode)
    seq_len = len(enc.ids)
    er = T._event_role_targets([record], [enc], tokenizer, dims, seq_len)  # deterministic target builder
    evidence = T._evidence_gold_index(tokenizer, record, enc)

    arg_source = 0
    pred_source = 0
    for ev in (record.get("event_graph") or [])[:1]:
        if isinstance(ev.get("predicate"), str):
            pred_source += 1
        for arg in ev.get("arguments") or []:
            if isinstance(arg.get("span"), str):
                arg_source += 1
    evidence_source = 1 if record.get("labels", {}).get("evidence_ids") else 0

    return {
        "arg_source": arg_source,
        "arg_located": int((er["arg_start"][0] != -100).sum().item()),
        "predicate_source": pred_source,
        "predicate_located": int((er["event_token"][0] != -100).sum().item()),
        "evidence_source": evidence_source,
        "evidence_located": int(evidence != -100),
    }


def audit_record(record: dict[str, Any], tokenizer: Any, dims: T.Dims, render_mode: str) -> dict[str, Any]:
    labels = record.get("labels") or {}
    text = T.render_for_mode(record, render_mode)
    forbidden = [key for key in FORBIDDEN_INPUT_KEYS if key in text]
    ev_ok, ev_failures, ev_exempt = event_consistency(record)
    targets = target_recovery(record, tokenizer, dims, render_mode)
    prov = record.get("provenance") or {}
    has_provenance = bool(prov or record.get("source") or record.get("family"))
    return {
        "primitive_vocab_ok": labels.get("primitive_class") in builder.PRIMITIVE_CLASSES,
        "support_vocab_ok": labels.get("support_status") in builder.SUPPORT_STATUSES,
        "idk_vocab_ok": labels.get("idk_action") in builder.IDK_ACTIONS if labels.get("idk_action") is not None else True,
        "event_consistency_ok": ev_ok,
        "event_local_exempt": ev_exempt,
        "event_failures": ev_failures,
        "no_cot_leakage": not forbidden,
        "forbidden_keys": forbidden,
        "has_provenance": has_provenance,
        **targets,
    }


def pct(num: int, den: int) -> str:
    return "n/a" if den == 0 else f"{num / den:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, nargs="+", required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pat_er_qwen3_warmstart.yaml")
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--render-mode", choices=list(T.RENDER_MODES), default="input_only")
    ap.add_argument("--sample-per-source", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-md", type=Path, default=ROOT / "artifacts" / "reports" / "label_reliability_audit.md")
    ap.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "reports" / "label_reliability_audit.json")
    args = ap.parse_args()

    records = load_records(args.dataset, args.split)
    if not records:
        raise RuntimeError("no records loaded")
    tokenizer = H.load_pater_hf_tokenizer(args.tokenizer) if args.tokenizer else T.build_tokenizer(records)
    dims = T.dims_from_config(PATERConfig.from_yaml(args.config))
    rng = random.Random(args.seed)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[source_bucket(record)].append(record)

    sampled: list[dict[str, Any]] = []
    for bucket, rows in sorted(by_source.items()):
        rows = list(rows)
        rng.shuffle(rows)
        sampled.extend(rows[: args.sample_per_source])

    audited = []
    for record in sampled:
        row = audit_record(record, tokenizer, dims, args.render_mode)
        row["source_bucket"] = source_bucket(record)
        row["record_id"] = record.get("id") or record.get("source_id") or (record.get("provenance") or {}).get("source_id")
        audited.append(row)

    summary: dict[str, Any] = {}
    for bucket in sorted({r["source_bucket"] for r in audited}):
        rows = [r for r in audited if r["source_bucket"] == bucket]
        n = len(rows)
        arg_src = sum(r["arg_source"] for r in rows)
        arg_loc = sum(r["arg_located"] for r in rows)
        pred_src = sum(r["predicate_source"] for r in rows)
        pred_loc = sum(r["predicate_located"] for r in rows)
        ev_src = sum(r["evidence_source"] for r in rows)
        ev_loc = sum(r["evidence_located"] for r in rows)
        summary[bucket] = {
            "n": n,
            "primitive_vocab_ok": sum(bool(r["primitive_vocab_ok"]) for r in rows),
            "support_vocab_ok": sum(bool(r["support_vocab_ok"]) for r in rows),
            "event_consistency_ok": sum(bool(r["event_consistency_ok"]) for r in rows),
            "event_local_exempt": sum(bool(r["event_local_exempt"]) for r in rows),
            "no_cot_leakage": sum(bool(r["no_cot_leakage"]) for r in rows),
            "has_provenance": sum(bool(r["has_provenance"]) for r in rows),
            "arg_located": arg_loc,
            "arg_source": arg_src,
            "predicate_located": pred_loc,
            "predicate_source": pred_src,
            "evidence_located": ev_loc,
            "evidence_source": ev_src,
        }

    lines = [
        "# Label Reliability Audit",
        "",
        "Deterministic audit of converted/synthetic supervision. This audit checks pipeline invariants; it does not use an LLM as a label judge.",
        "",
        f"Dataset files: {', '.join(str(p) for p in args.dataset)}",
        f"Split: `{args.split}`; render mode: `{args.render_mode}`; sample per source: {args.sample_per_source}",
        "",
        "| source bucket | n | primitive vocab | support vocab | event primitive consistency | event-local exemptions | no CoT leakage | provenance | arg span located | predicate located | evidence located |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, s in summary.items():
        n = s["n"]
        lines.append(
            f"| {bucket} | {n} | {pct(s['primitive_vocab_ok'], n)} | {pct(s['support_vocab_ok'], n)} | "
            f"{pct(s['event_consistency_ok'], n)} | {s['event_local_exempt']} | "
            f"{pct(s['no_cot_leakage'], n)} | {pct(s['has_provenance'], n)} | "
            f"{pct(s['arg_located'], s['arg_source'])} ({s['arg_located']}/{s['arg_source']}) | "
            f"{pct(s['predicate_located'], s['predicate_source'])} ({s['predicate_located']}/{s['predicate_source']}) | "
            f"{pct(s['evidence_located'], s['evidence_source'])} ({s['evidence_located']}/{s['evidence_source']}) |"
        )

    failures = [
        r for r in audited
        if not (r["primitive_vocab_ok"] and r["support_vocab_ok"] and r["event_consistency_ok"]
                and r["no_cot_leakage"] and r["has_provenance"])
    ]
    if failures:
        lines.extend(["", "## Non-Target Failures", ""])
        for row in failures[:25]:
            lines.append(f"- {row['source_bucket']} `{row.get('record_id')}`: {row}")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps({"summary": summary, "sampled": audited}, indent=2), encoding="utf-8")
    print(f"audited {len(audited)} records across {len(summary)} source buckets")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
