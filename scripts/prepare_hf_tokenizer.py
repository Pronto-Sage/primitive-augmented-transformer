#!/usr/bin/env python3
"""Build the production PAT-ER tokenizer: a pretrained HF tokenizer extended with
the PAT-ER special/normal tokens, saved under artifacts/tokenizers/.

Loads the base tokenizer from the local HF cache only by default (no network);
pass --allow-download to permit a fetch (requires explicit approval). Applies the
98 special + 34 normal PAT-ER tokens via pat_er.tokenizer_spec, verifies every
control token is atomic, saves the extended tokenizer with save_pretrained, and
writes a manifest with token counts, atomicity, and pad/bos/eos ids.

Exits non-zero if any PAT-ER special token is non-atomic under the base
tokenizer (the production extension would be broken).

CPU-only. No model is loaded, no training is run.

Usage:
    python3 scripts/prepare_hf_tokenizer.py --base Qwen/Qwen3-0.6B --local-only \
        --output artifacts/tokenizers/qwen_pater_extended
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

from pat_er.tokenizer_spec import build_pater_tokenizer_spec  # noqa: E402
import pater_hf_tokenizer as H  # noqa: E402

DEFAULT_OUTPUT = ROOT / "artifacts" / "tokenizers" / "qwen_pater_extended"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PAT-ER-extended pretrained tokenizer (offline by default).")
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-0.6B", help="Base HF tokenizer name or local path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to save the extended tokenizer.")
    parser.add_argument("--local-only", action="store_true", default=True,
                        help="Load the base tokenizer from local cache only (default).")
    parser.add_argument("--allow-download", action="store_true",
                        help="Permit downloading the base tokenizer (network). Off by default; requires approval.")
    args = parser.parse_args()

    spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
    try:
        tok, counts = H.build_extended_tokenizer(args.base, allow_download=args.allow_download)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not load base tokenizer '{args.base}': {exc}")
        if not args.allow_download:
            print("      loading is offline by default; the base must be in the local HF cache, or pass")
            print("      --allow-download (requires explicit approval) to fetch it.")
        sys.exit(2)

    base_vocab = len(tok) - counts["added_special"] - counts["added_normal"]
    special_atom = H.atomicity(tok, spec.special_tokens)
    normal_atom = H.atomicity(tok, spec.normal_tokens)

    args.output.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(args.output))

    manifest = {
        "base": args.base,
        "loaded_offline": not args.allow_download,
        "base_vocab_size": base_vocab,
        "added_special": counts["added_special"],
        "added_normal": counts["added_normal"],
        "extended_vocab_size": len(tok),
        "atomicity": {"special": special_atom, "normal": normal_atom},
        "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
        "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id,
        "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
        "unk_token": tok.unk_token, "unk_token_id": getattr(tok, "unk_token_id", None),
    }
    (args.output / "pater_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"base tokenizer: {args.base}  (offline={not args.allow_download})")
    print(f"base vocab={base_vocab}  +special={counts['added_special']}  +normal={counts['added_normal']}  "
          f"-> extended vocab={len(tok)}")
    print(f"atomicity: special {special_atom['atomic']}/{special_atom['total']} (rate={special_atom['rate']})  "
          f"normal {normal_atom['atomic']}/{normal_atom['total']} (rate={normal_atom['rate']})")
    print(f"pad={tok.pad_token}({tok.pad_token_id}) bos={tok.bos_token}({tok.bos_token_id}) "
          f"eos={tok.eos_token}({tok.eos_token_id}) unk={tok.unk_token}")
    print(f"saved extended tokenizer -> {args.output}")
    print(f"wrote manifest -> {args.output / 'pater_manifest.json'}")

    if special_atom["failures"] or normal_atom["failures"]:
        for f in (special_atom["failures"] + normal_atom["failures"])[:10]:
            print(f"  NON-ATOMIC {f['token']} -> {f['ids']}")
        print("FAIL: PAT-ER control tokens are not all atomic under this base tokenizer")
        sys.exit(1)
    print("PASS: all 98 special + 34 normal PAT-ER tokens are atomic")


if __name__ == "__main__":
    main()
