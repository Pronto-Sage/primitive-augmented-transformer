from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .schemas import EncodedBatch
from .tokenizer_spec import PATERTokenizerSpec, build_pater_tokenizer_spec


SAMPLES: list[str] = [
    "evidence e1 log shows service_a restarted",
    "rule if tests_passed implies release_ok evidence e2 tests_passed",
    "rule if cleanup_job ran implies file_deleted evidence e3 file_deleted maybe cleanup_job ran",
    "alice approved bob",
    "bob approved alice",
    "john cut meat with knife",
    "john burgled house with accomplice",
    "john loaded truck with rocks",
    "tool verify_release requires tests_passed produces release_ok",
]


@dataclass
class ReferenceTokenizer:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    spec: PATERTokenizerSpec
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, text: str) -> list[str]:
        return tokenize_with_pater_spec(text, self.spec)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in self.tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special_tokens and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            tokens.append(token)
        return " ".join(tokens)


def tokenize_with_pater_spec(text: str, spec: PATERTokenizerSpec) -> list[str]:
    """Reference smoke tokenizer that keeps PAT-ER/Hermes controls atomic.

    This is not a production tokenizer. It is a deterministic local tokenizer
    for architecture smoke runs and serialization checks. Production should use
    a pretrained byte-level or byte-fallback tokenizer extended with this spec.
    """

    protected = sorted(set(spec.special_tokens + spec.normal_tokens), key=len, reverse=True)
    protected_pattern = "|".join(re.escape(token) for token in protected)
    token_pattern = re.compile(
        rf"{protected_pattern}"
        r'|"(?:[^"\\]|\\.)*"'
        r"|[A-Za-z_][A-Za-z0-9_./-]*"
        r"|\d+(?:\.\d+)?"
        r"|[{}\[\]():,]"
        r"|\S"
    )
    return token_pattern.findall(text)


def build_reference_tokenizer(extra_texts: list[str] | None = None) -> ReferenceTokenizer:
    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)
    architecture_tokens = list(spec.special_tokens) + list(spec.normal_tokens)
    texts = list(SAMPLES)
    if extra_texts:
        texts.extend(extra_texts)
    words = sorted({token for sample in texts for token in tokenize_with_pater_spec(sample, spec)})
    vocab = special + architecture_tokens + words
    token_to_id = {token: idx for idx, token in enumerate(dict.fromkeys(vocab))}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return ReferenceTokenizer(token_to_id=token_to_id, id_to_token=id_to_token, spec=spec)


def encode_texts(tokenizer: ReferenceTokenizer, texts: list[str]) -> EncodedBatch:
    encoded = [tokenizer.encode(text) for text in texts]
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask, texts=texts)


def build_sample_batch() -> tuple[ReferenceTokenizer, EncodedBatch]:
    tokenizer = build_reference_tokenizer()
    return tokenizer, encode_texts(tokenizer, SAMPLES)
