#!/usr/bin/env python3
"""Aggregate contradiction precision/recall/F1 from matrix eval JSON files.

This is a reviewer-facing companion to compute_matrix_stats.py. It reports the
per-class contradiction row for both primitive and role_to_primitive heads.

Usage:
    python3 scripts/report_contradiction_prf.py --seeds 0-7 \
      --output-md artifacts/reports/contradiction_prf.md \
      --output-json artifacts/reports/contradiction_prf.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_seeds(s: str) -> list[int]:
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def _row(path: Path, head: str) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    metrics = data.get("metrics") or {}
    h = metrics.get(head) or {}
    pc = h.get("per_class_prf") or {}
    row = pc.get("contradiction")
    if not row:
        return None
    return {
        "precision": float(row.get("precision", 0.0)),
        "recall": float(row.get("recall", 0.0)),
        "f1": float(row.get("f1", 0.0)),
        "support": float(row.get("support", 0.0)),
        "predicted": float(row.get("predicted", 0.0)),
    }


def _mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def _std(vals: list[float]) -> float | None:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None


def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in ("precision", "recall", "f1", "support", "predicted"):
        vals = [r[key] for r in rows if key in r]
        out[key] = {"mean": _mean(vals), "std": _std(vals), "values": vals}
    return out


def fmt(cell: dict[str, Any]) -> str:
    if cell["mean"] is None:
        return "-"
    return f"{cell['mean']:.3f} ± {cell['std']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", type=Path, default=Path("artifacts/reports"))
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--conditions", default="b,g,c,d")
    ap.add_argument("--datasets", default="shallow,deep")
    ap.add_argument("--output-md", type=Path, default=Path("artifacts/reports/contradiction_prf.md"))
    ap.add_argument("--output-json", type=Path, default=Path("artifacts/reports/contradiction_prf.json"))
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    labels = {
        "b": "B token-state",
        "g": "G generic-register",
        "c": "C PAT-ER",
        "d": "D warm-start",
    }

    result: dict[str, Any] = {}
    lines = [
        "# Contradiction Precision/Recall/F1",
        "",
        "Mean ± std across available seeds. Values are the contradiction class row from the head confusion matrix.",
        "",
        "| condition | split | head | n seeds | precision | recall | F1 | support | predicted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond in conditions:
        for ds in datasets:
            for head in ("primitive", "role_to_primitive"):
                rows = []
                for seed in seeds:
                    path = args.report_dir / f"matrix_{cond}_{ds}_s{seed}.json"
                    row = _row(path, head)
                    if row:
                        rows.append(row)
                key = f"{cond}_{ds}_{head}"
                summary = summarize(rows)
                result[key] = summary
                lines.append(
                    f"| {labels.get(cond, cond)} | {ds} | {head} | {summary['n']} | "
                    f"{fmt(summary['precision'])} | {fmt(summary['recall'])} | {fmt(summary['f1'])} | "
                    f"{fmt(summary['support'])} | {fmt(summary['predicted'])} |"
                )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
