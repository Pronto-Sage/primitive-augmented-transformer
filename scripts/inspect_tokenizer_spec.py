from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

from pat_er.tokenizer_spec import build_pater_tokenizer_spec


def main() -> None:
    spec_with_schema = build_pater_tokenizer_spec(include_schema_key_candidates=True)
    spec = spec_with_schema

    print("PATERTokenizer-v1 spec")
    print(f"special tokens: {spec.num_special_tokens}")
    print(f"normal formula/operator/schema tokens: {spec.num_normal_tokens}")
    print(f"total added tokens: {spec.total_added_tokens_without_schema_keys}")
    print("embedding impact at hidden_size=1536 with tied embeddings:")
    print(f"  with schema keys: {spec_with_schema.total_added_tokens_without_schema_keys * 1536:,} params")
    print("Hermes/vLLM tool-call control tokens:")
    print("  <tool_call>")
    print("  </tool_call>")
    print("first special tokens:")
    for token in spec.special_tokens[:20]:
        print(f"  {token}")
    print("normal tokens:")
    for token in spec.normal_tokens:
        print(f"  {token}")


if __name__ == "__main__":
    main()
