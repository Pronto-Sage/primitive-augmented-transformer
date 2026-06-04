#!/usr/bin/env python3
"""Production HF tokenizer adapter for the PAT-ER training/eval path.

Wraps a PAT-ER-extended pretrained (Qwen) tokenizer behind the small surface the
train/eval scripts already expect from the reference tokenizer (encode, pad/eos
ids, vocab_size, token_to_id) and adds byte-level-safe span/evidence location via
character offsets. Loading is OFFLINE by default (local_files_only); nothing is
downloaded unless explicitly allowed.

Why offsets: the reference (word-level) tokenizer locates predicate/argument
spans by re-encoding the span text and finding it as a contiguous id subsequence.
Under byte-level BPE the same characters tokenize differently depending on
context (leading space, byte boundaries), so that subsequence match fails. This
adapter instead finds the gold span's character range in the rendered text and
maps it to token indices through the fast tokenizer's offset mapping -- the
production-safe migration of argument_start/end, event_token, and evidence
pointer labels.

CPU-only. transformers is imported lazily so the default reference path stays
dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er.tokenizer_spec import apply_to_hf_tokenizer, build_pater_tokenizer_spec  # noqa: E402

Offsets = list[tuple[int, int]]


class HFPaterTokenizer:
    """ReferenceTokenizer-compatible wrapper around a fast HF tokenizer.

    ``supports_offsets`` lets the collation code branch to offset-based span
    location. PAT-ER framing tokens (``<pat_er> ... </pat_er>``) are the
    structural delimiters, so we never add the base tokenizer's auto specials
    (Qwen has no BOS); encode/offsets always use add_special_tokens=False so the
    offset mapping is aligned 1:1 with input_ids.
    """

    supports_offsets = True

    def __init__(self, hf_tokenizer: Any, path: str | None = None) -> None:
        self.tok = hf_tokenizer
        self.path = path
        self.spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
        self._vocab = hf_tokenizer.get_vocab()
        # Qwen byte-BPE has no bos/unk; fall back to pad/eos where an int is
        # structurally required so callers never see None.
        self.pad_token = hf_tokenizer.pad_token or hf_tokenizer.eos_token
        self.eos_token = hf_tokenizer.eos_token or hf_tokenizer.pad_token
        self.bos_token = hf_tokenizer.bos_token  # may be None
        self.unk_token = hf_tokenizer.unk_token  # may be None

    # --- ReferenceTokenizer-compatible surface -------------------------------
    @property
    def token_to_id(self) -> dict[str, int]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def pad_token_id(self) -> int:
        pid = self.tok.pad_token_id
        return int(pid if pid is not None else self.tok.eos_token_id)

    @property
    def eos_token_id(self) -> int:
        eid = self.tok.eos_token_id
        return int(eid if eid is not None else self.pad_token_id)

    @property
    def bos_token_id(self) -> int | None:
        return self.tok.bos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def encode_with_offsets(self, text: str, add_special_tokens: bool = False) -> tuple[list[int], Offsets]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def decode(self, ids: Sequence[int] | Any, skip_special_tokens: bool = False) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)


def load_pater_hf_tokenizer(path: str | Path, *, allow_download: bool = False) -> HFPaterTokenizer:
    """Load a previously-saved PAT-ER-extended tokenizer directory (offline)."""

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=not allow_download, use_fast=True)
    return HFPaterTokenizer(tok, path=str(path))


def build_extended_tokenizer(base: str, *, allow_download: bool = False) -> tuple[Any, dict[str, int]]:
    """Load a base HF tokenizer (offline by default) and apply the PAT-ER spec."""

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, local_files_only=not allow_download, use_fast=True)
    counts = apply_to_hf_tokenizer(tok, include_schema_key_candidates=True)
    return tok, counts


# --- offset-based location helpers (byte-BPE safe) ---------------------------
def locate_char_span(offsets: Offsets, char_start: int, char_end: int) -> tuple[int, int] | None:
    """First/last token index whose char range overlaps [char_start, char_end).

    Zero-width tokens (added/special tokens map to (0, 0)) are skipped via the
    ``b > a`` guard, so they never spuriously claim a span position.
    """

    hit = [i for i, (a, b) in enumerate(offsets) if b > a and a < char_end and b > char_start]
    if not hit:
        return None
    return hit[0], hit[-1]


def locate_text_span(text: str, offsets: Offsets, span_text: str) -> tuple[int, int] | None:
    """Find ``span_text`` in ``text`` (first occurrence) and map it to tokens."""

    if not span_text:
        return None
    cs = text.find(span_text)
    if cs < 0:
        return None
    return locate_char_span(offsets, cs, cs + len(span_text))


def atomicity(tok: Any, tokens: Sequence[str]) -> dict[str, Any]:
    """How many of ``tokens`` encode to exactly one (non-unk) id."""

    unk = getattr(tok, "unk_token_id", None)
    failures = []
    for token in tokens:
        ids = tok.encode(token, add_special_tokens=False)
        if len(ids) != 1 or (unk is not None and ids[0] == unk):
            failures.append({"token": token, "ids": ids})
    total = len(tokens)
    return {"total": total, "atomic": total - len(failures), "rate": round((total - len(failures)) / total, 4) if total else 1.0,
            "failures": failures}
