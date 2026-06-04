#!/usr/bin/env python3
"""Evaluate PAT-ER tokenizer fragmentation on the generated corpus.

This measures whether a tokenizer carries the PAT-ER interface without breaking
control tokens or exploding structured constructs into many subword pieces.

It always evaluates the **local reference tokenizer first** (the dependency-free
``ReferenceTokenizer`` used by the smoke scripts). If Hugging Face ``transformers``
is installed, an optional pretrained tokenizer can also be evaluated via
``--hf-tokenizer NAME_OR_PATH``; by default the PAT-ER spec is applied to it
(the production extension path) and loading stays **offline** unless
``--allow-download`` is passed.

Reported metrics (per tokenizer):

- control-token atomicity (special + normal/control tokens);
- tokens per formula / evidence block / event graph / tool schema / Hermes call;
- schema-key fragmentation (bare and JSON-quoted);
- dynamic-ID fragmentation (bare and JSON-quoted);
- Hermes tool-call parse rate after encode -> decode;
- compression ratio (characters and bytes per token) over full records;
- unknown / fallback rate over full records.

Hard requirement: the script **fails (non-zero exit)** if any required PAT-ER
special token is not atomic under an evaluated tokenizer.

CPU-only. No model is loaded, no training is run, and nothing is downloaded
unless ``--allow-download`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

from pat_er.sample_data import build_reference_tokenizer  # noqa: E402
from pat_er.serialization import parse_hermes_tool_calls  # noqa: E402
from pat_er.tokenizer_spec import build_pater_tokenizer_spec  # noqa: E402

# build_tokenizer_corpus lives next to this script; import it for the in-memory
# fallback when on-disk artifacts have not been written yet.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tokenizer_corpus as corpus_builder  # noqa: E402

DEFAULT_CORPUS_DIR = ROOT / "artifacts" / "tokenizer_corpus"


# ---------------------------------------------------------------------------
# Tokenizer adapters: a uniform surface so the same metric code runs on the
# reference tokenizer and on an optional Hugging Face tokenizer.
# ---------------------------------------------------------------------------
class TokenizerAdapter:
    name: str

    def encode_ids(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def is_unk(self, token_id: int) -> bool:
        return False

    def count(self, text: str) -> int:
        return len(self.encode_ids(text))


class ReferenceAdapter(TokenizerAdapter):
    """Local dependency-free reference tokenizer, fit to the corpus vocabulary."""

    def __init__(self, corpus_texts: Sequence[str]) -> None:
        self.tokenizer = build_reference_tokenizer(extra_texts=list(corpus_texts))
        self.unk_id = self.tokenizer.token_to_id[self.tokenizer.unk_token]
        self.name = "reference"

    def encode_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def is_unk(self, token_id: int) -> bool:
        return token_id == self.unk_id


class HFAdapter(TokenizerAdapter):
    """Optional pretrained Hugging Face tokenizer, optionally PAT-ER-extended."""

    def __init__(self, name_or_path: str, extend: bool, allow_download: bool) -> None:
        from transformers import AutoTokenizer  # local import keeps default path dependency-free

        from pat_er.tokenizer_spec import apply_to_hf_tokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            local_files_only=not allow_download,
            use_fast=True,
        )
        suffix = ""
        if extend:
            counts = apply_to_hf_tokenizer(self.tokenizer, include_schema_key_candidates=True)
            suffix = f"+pater({counts['added_special']}sp/{counts['added_normal']}nm)"
        self.unk_id = getattr(self.tokenizer, "unk_token_id", None)
        self.name = f"hf:{name_or_path}{suffix}"

    def encode_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def is_unk(self, token_id: int) -> bool:
        return self.unk_id is not None and token_id == self.unk_id


# ---------------------------------------------------------------------------
# Metric helpers.
# ---------------------------------------------------------------------------
def _summary(counts: Sequence[int]) -> dict[str, float]:
    if not counts:
        return {"n": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0, "min": 0, "total": 0}
    ordered = sorted(counts)
    n = len(ordered)

    def pct(p: float) -> int:
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return ordered[k]

    return {
        "n": n,
        "mean": round(sum(ordered) / n, 3),
        "p50": pct(50),
        "p95": pct(95),
        "max": ordered[-1],
        "min": ordered[0],
        "total": sum(ordered),
    }


def _fragmentation(adapter: TokenizerAdapter, items: Sequence[str]) -> dict[str, Any]:
    counts = [adapter.count(item) for item in items]
    fragmented = sum(1 for c in counts if c > 1)
    stats = _summary(counts)
    stats["fragmented"] = fragmented
    stats["fragmentation_rate"] = round(fragmented / len(counts), 4) if counts else 0.0
    return stats


def _atomicity(adapter: TokenizerAdapter, tokens: Sequence[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for token in tokens:
        ids = adapter.encode_ids(token)
        if len(ids) != 1 or adapter.is_unk(ids[0]):
            failures.append({"token": token, "ids": ids})
    total = len(tokens)
    atomic = total - len(failures)
    return {
        "total": total,
        "atomic": atomic,
        "rate": round(atomic / total, 4) if total else 1.0,
        "failures": failures,
    }


def _hermes_parse_rate(adapter: TokenizerAdapter, calls: Sequence[str]) -> dict[str, Any]:
    parsed_ok = 0
    for call in calls:
        decoded = adapter.decode(adapter.encode_ids(call))
        try:
            results = parse_hermes_tool_calls(decoded)
        except Exception:
            continue
        if results and all(isinstance(r.name, str) and r.name and isinstance(r.arguments, dict) for r in results):
            parsed_ok += 1
    total = len(calls)
    return {"total": total, "parsed": parsed_ok, "parse_rate": round(parsed_ok / total, 4) if total else 1.0}


def _compression(adapter: TokenizerAdapter, records: Sequence[str]) -> dict[str, Any]:
    total_chars = sum(len(r) for r in records)
    total_bytes = sum(len(r.encode("utf-8")) for r in records)
    token_counts = [adapter.count(r) for r in records]
    total_tokens = sum(token_counts)
    return {
        "records": len(records),
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 4) if total_tokens else 0.0,
        "bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens else 0.0,
        "tokens_per_record": _summary(token_counts),
    }


def _fallback_rate(adapter: TokenizerAdapter, records: Sequence[str]) -> dict[str, Any]:
    total_tokens = 0
    unknown = 0
    for record in records:
        for token_id in adapter.encode_ids(record):
            total_tokens += 1
            if adapter.is_unk(token_id):
                unknown += 1
    return {
        "total_tokens": total_tokens,
        "unknown": unknown,
        "fallback_rate": round(unknown / total_tokens, 6) if total_tokens else 0.0,
    }


def _collect_fragments(records: Sequence[dict[str, Any]], frag_type: str) -> list[str]:
    fragments: list[str] = []
    for record in records:
        fragments.extend(record.get("fragments", {}).get(frag_type, []))
    return fragments


def _quoted(value: str) -> str:
    """How an identifier/key appears inside canonical JSON: as a quoted string."""

    return json.dumps(value, ensure_ascii=False)


def evaluate(adapter: TokenizerAdapter, records: Sequence[dict[str, Any]], dynamic_ids: Sequence[str]) -> dict[str, Any]:
    spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
    full_texts = [record["text"] for record in records]

    schema_keys = list(spec.schema_key_candidates)
    metrics: dict[str, Any] = {
        "tokenizer": adapter.name,
        "atomicity": {
            "special": _atomicity(adapter, spec.special_tokens),
            "normal": _atomicity(adapter, spec.normal_tokens),
        },
        "tokens_per_construct": {
            "formula": _fragmentation(adapter, _collect_fragments(records, "formula")),
            "evidence_block": _fragmentation(adapter, _collect_fragments(records, "evidence")),
            "event_graph": _fragmentation(adapter, _collect_fragments(records, "event_graph")),
            "tool_schema": _fragmentation(adapter, _collect_fragments(records, "tool_schema")),
            "hermes_tool_call": _fragmentation(adapter, _collect_fragments(records, "hermes_tool_call")),
        },
        "schema_key_fragmentation": {
            "bare": _fragmentation(adapter, schema_keys),
            "json_quoted": _fragmentation(adapter, [_quoted(k) for k in schema_keys]),
        },
        "dynamic_id_fragmentation": {
            "bare": _fragmentation(adapter, list(dynamic_ids)),
            "json_quoted": _fragmentation(adapter, [_quoted(i) for i in dynamic_ids]),
        },
        "hermes_parse_rate": _hermes_parse_rate(adapter, _collect_fragments(records, "hermes_tool_call")),
        "compression": _compression(adapter, full_texts),
        "fallback": _fallback_rate(adapter, full_texts),
    }
    return metrics


# ---------------------------------------------------------------------------
# Corpus loading.
# ---------------------------------------------------------------------------
def load_corpus(corpus_dir: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    corpus_path = corpus_dir / "corpus.jsonl"
    dynamic_path = corpus_dir / "dynamic_ids.json"
    if corpus_path.exists() and dynamic_path.exists():
        records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dynamic_ids = json.loads(dynamic_path.read_text(encoding="utf-8"))
        return records, dynamic_ids, "disk"
    # Fallback: build in-memory so the evaluator never spuriously fails when the
    # builder has not been run yet.
    records, dynamic_ids = corpus_builder.build_corpus()
    return records, dynamic_ids, "in-memory"


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _fmt_frag(stats: dict[str, Any]) -> str:
    return (
        f"n={stats['n']:>3} mean={stats['mean']:>7} p50={stats['p50']:>4} "
        f"p95={stats['p95']:>4} max={stats['max']:>4} frag_rate={stats['fragmentation_rate']}"
    )


def print_report(metrics: dict[str, Any]) -> None:
    print(f"=== tokenizer: {metrics['tokenizer']} ===")

    special = metrics["atomicity"]["special"]
    normal = metrics["atomicity"]["normal"]
    print(
        f"control-token atomicity: special {special['atomic']}/{special['total']} (rate={special['rate']}), "
        f"normal {normal['atomic']}/{normal['total']} (rate={normal['rate']})"
    )
    for label, block in (("special", special), ("normal", normal)):
        for failure in block["failures"][:10]:
            print(f"  NON-ATOMIC [{label}] {failure['token']} -> {failure['ids']}")

    print("tokens per construct:")
    for name, stats in metrics["tokens_per_construct"].items():
        print(f"  {name:<17} {_fmt_frag(stats)}")

    sk = metrics["schema_key_fragmentation"]
    print("schema-key fragmentation:")
    print(f"  bare        {_fmt_frag(sk['bare'])}")
    print(f"  json_quoted {_fmt_frag(sk['json_quoted'])}")

    did = metrics["dynamic_id_fragmentation"]
    print("dynamic-id fragmentation:")
    print(f"  bare        {_fmt_frag(did['bare'])}")
    print(f"  json_quoted {_fmt_frag(did['json_quoted'])}")

    hpr = metrics["hermes_parse_rate"]
    print(f"hermes parse rate: {hpr['parsed']}/{hpr['total']} (rate={hpr['parse_rate']})")

    comp = metrics["compression"]
    print(
        f"compression: chars/token={comp['chars_per_token']} bytes/token={comp['bytes_per_token']} "
        f"tokens/record mean={comp['tokens_per_record']['mean']} (max={comp['tokens_per_record']['max']}) "
        f"total_tokens={comp['total_tokens']}"
    )
    fb = metrics["fallback"]
    print(f"fallback rate: {fb['unknown']}/{fb['total_tokens']} (rate={fb['fallback_rate']})")
    print()


def atomicity_ok(metrics: dict[str, Any]) -> bool:
    return not metrics["atomicity"]["special"]["failures"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PAT-ER tokenizer fragmentation (CPU-only).")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR, help="Directory with corpus.jsonl.")
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default=None,
        help="Optional pretrained Hugging Face tokenizer name or local path to compare.",
    )
    parser.add_argument(
        "--no-extend",
        action="store_true",
        help="Do NOT apply the PAT-ER spec to the HF tokenizer (shows raw fragmentation).",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit downloading the HF tokenizer (network). Off by default; requires approval.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Where to write the metrics JSON report.")
    parser.add_argument(
        "--no-fail-on-nonatomic",
        action="store_true",
        help="Do not exit non-zero when required special tokens are non-atomic (diagnostic only).",
    )
    args = parser.parse_args()

    records, dynamic_ids, source = load_corpus(args.corpus_dir)
    print(f"loaded corpus ({source}): {len(records)} records, {len(dynamic_ids)} dynamic ids\n")

    corpus_texts = [record["text"] for record in records]
    adapters: list[TokenizerAdapter] = [ReferenceAdapter(corpus_texts)]

    if args.hf_tokenizer:
        try:
            adapters.append(
                HFAdapter(args.hf_tokenizer, extend=not args.no_extend, allow_download=args.allow_download)
            )
        except Exception as exc:  # noqa: BLE001  (surface the reason, keep reference result)
            print(f"WARNING: could not load HF tokenizer '{args.hf_tokenizer}': {exc}")
            if not args.allow_download:
                print("         (loading is offline by default; pass --allow-download to fetch, requires approval)")

    all_metrics = [evaluate(adapter, records, dynamic_ids) for adapter in adapters]
    for metrics in all_metrics:
        print_report(metrics)

    nonatomic = [m["tokenizer"] for m in all_metrics if not atomicity_ok(m)]

    report = {
        "corpus_source": source,
        "num_records": len(records),
        "num_dynamic_ids": len(dynamic_ids),
        "tokenizers": all_metrics,
        "nonatomic_tokenizers": nonatomic,
    }
    json_out = args.json_out or (args.corpus_dir / "fragmentation_report.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote metrics report: {json_out}")

    if nonatomic:
        print(f"FAIL: required PAT-ER special tokens are not atomic under: {', '.join(nonatomic)}")
        if not args.no_fail_on_nonatomic:
            sys.exit(1)
    else:
        print("PASS: all required PAT-ER special tokens are atomic under every evaluated tokenizer")


if __name__ == "__main__":
    main()
