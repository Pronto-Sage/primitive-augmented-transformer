from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

import torch

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import maybe_autocast
from pat_er.sample_data import SAMPLES, build_reference_tokenizer


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return requested


@torch.no_grad()
def generate(dtype_name: str, device: str, max_new_tokens: int) -> None:
    torch.manual_seed(11)
    tokenizer = build_reference_tokenizer()
    config = PATERConfig.from_yaml(ROOT / "configs" / "tiny_smoke.yaml")
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
    model = PATERForCausalLM(config).to(device).eval()

    prompt = SAMPLES[1]
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        with maybe_autocast(device, dtype_name):
            output = model(input_ids=input_ids, attention_mask=attention_mask, return_aux=True)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break

    print("PAT-ER smoke_generate complete")
    print(f"device: {device}")
    print(f"requested dtype: {dtype_name}")
    print(f"prompt: {prompt}")
    print(f"generated ids: {input_ids[0].detach().cpu().tolist()}")
    print(f"generated text: {tokenizer.decode(input_ids[0])}")
    print(f"final logits shape: {tuple(output.logits.shape)}")
    print(f"event register shape: {tuple(output.event_registers.shape)}")
    print(f"primitive register shape: {tuple(output.primitive_registers.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    device = resolve_device(args.device)
    try:
        generate(args.dtype, device, args.max_new_tokens)
    except Exception as exc:
        if args.dtype == "bf16":
            print(f"bf16 path failed: {exc}")
            print("rerunning in fp32")
            generate("fp32", device, args.max_new_tokens)
        else:
            raise


if __name__ == "__main__":
    main()
