from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

import torch

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import count_parameters, estimate_decoder_params, iter_module_summary
from pat_er.sample_data import build_sample_batch


def main() -> None:
    torch.manual_seed(13)
    tokenizer, batch = build_sample_batch()

    tiny = PATERConfig.from_yaml(ROOT / "configs" / "tiny_smoke.yaml")
    tiny.vocab_size = max(tiny.vocab_size, tokenizer.vocab_size)
    model = PATERForCausalLM(tiny)
    total, trainable = count_parameters(model)

    print("PAT-ER inspect_shapes")
    print(f"tiny config vocab_size={tiny.vocab_size}")
    print(f"tiny parameters: total={total:,} trainable={trainable:,}")
    print(f"event registers: {tiny.num_event_registers}")
    print(f"argument registers: {tiny.num_argument_registers}")
    print(f"event-role registers total: {tiny.num_event_role_registers}")
    print(f"primitive registers: {tiny.num_primitive_registers}")
    print(f"proto-role properties: {tiny.num_proto_role_properties}")
    print("module summary:")
    for line in iter_module_summary(model, max_depth=2):
        print(f"  {line}")

    with torch.no_grad():
        output = model(batch.input_ids, attention_mask=batch.attention_mask, return_aux=True)
    print("sample forward shapes:")
    print(f"  input_ids: {tuple(batch.input_ids.shape)}")
    print(f"  logits: {tuple(output.logits.shape)}")
    print(f"  event_registers: {tuple(output.event_registers.shape)}")
    print(f"  primitive_registers: {tuple(output.primitive_registers.shape)}")
    for name, value in sorted(output.aux_outputs.items()):
        print(f"  aux.{name}: {tuple(value.shape)}")

    target = PATERConfig.from_yaml(ROOT / "configs" / "pat_er_760m.yaml")
    with torch.device("meta"):
        target_model = PATERForCausalLM(target)
    target_total, target_trainable = count_parameters(target_model)
    rough_base = estimate_decoder_params(
        vocab_size=target.vocab_size,
        hidden_size=target.hidden_size,
        intermediate_size=target.intermediate_size,
        num_layers=target.num_hidden_layers,
        num_heads=target.num_attention_heads,
        num_kv_heads=target.num_key_value_heads,
        tie_embeddings=target.tie_word_embeddings,
    )
    print("target config actual instantiated params:")
    print(f"  total={target_total:,} trainable={target_trainable:,}")
    print(f"  rough base decoder only={rough_base:,}")
    print("  actual count includes selected cross-stream layers, adapters, pressure heads, and aux heads")


if __name__ == "__main__":
    main()
