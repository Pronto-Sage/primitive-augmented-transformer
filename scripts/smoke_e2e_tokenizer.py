from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

from pat_er.sample_data import build_reference_tokenizer
from pat_er.serialization import (
    build_reference_pater_record,
    parse_hermes_tool_calls,
    render_hermes_tool_call,
)
from pat_er.tokenizer_spec import build_pater_tokenizer_spec


def assert_atomic(tokenizer, tokens: tuple[str, ...]) -> None:
    failures: list[str] = []
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.token_to_id[tokenizer.unk_token]:
            failures.append(f"{token} -> {ids}")
    if failures:
        preview = "\n".join(failures[:20])
        raise AssertionError(f"non-atomic PAT-ER tokens:\n{preview}")


def main() -> None:
    record = build_reference_pater_record()
    tool_call = render_hermes_tool_call(
        "verify_release",
        {"run_id": "r_17", "requires": "tests_passed"},
    )
    tokenizer = build_reference_tokenizer(extra_texts=[record, tool_call])
    spec = build_pater_tokenizer_spec(include_schema_key_candidates=True)

    assert_atomic(tokenizer, spec.special_tokens)
    assert_atomic(tokenizer, spec.normal_tokens)

    record_ids = tokenizer.encode(record)
    decoded_record = tokenizer.decode(record_ids)
    tool_ids = tokenizer.encode(tool_call)
    decoded_tool_call = tokenizer.decode(tool_ids)
    parsed_calls = parse_hermes_tool_calls(decoded_tool_call)
    if len(parsed_calls) != 1:
        raise AssertionError(f"expected one parsed tool call, got {len(parsed_calls)}")
    parsed = parsed_calls[0]
    if parsed.name != "verify_release" or parsed.arguments["run_id"] != "r_17":
        raise AssertionError(f"bad parsed tool call: {parsed}")

    print("PAT-ER e2e tokenizer smoke complete")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"special tokens: {spec.num_special_tokens}")
    print(f"normal/control tokens: {spec.num_normal_tokens}")
    print(f"record token count: {len(record_ids)}")
    print(f"tool-call token count: {len(tool_ids)}")
    print("sample PAT-ER record:")
    print(record)
    print("Hermes-style tool call:")
    print(tool_call)
    print("decoded Hermes-style tool call:")
    print(decoded_tool_call)
    print("parsed OpenAI-compatible call:")
    print(parsed.to_openai_tool_call())


if __name__ == "__main__":
    main()
