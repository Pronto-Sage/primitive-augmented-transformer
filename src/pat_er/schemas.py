from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    texts: list[str]


@dataclass
class GenerationResult:
    prompt: str
    token_ids: list[int]
    text: str

