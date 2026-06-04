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
from pat_er.modules.utils import count_parameters, maybe_autocast
from pat_er.sample_data import build_sample_batch


def shape_of(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)} dtype={value.dtype}"
    return str(type(value))


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return requested


def maybe_compile(model: torch.nn.Module, should_compile: bool) -> tuple[torch.nn.Module, bool, bool]:
    if not should_compile:
        return model, False, False
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; continuing uncompiled")
        return model, True, False
    try:
        compiled = torch.compile(model)
        return compiled, True, True
    except Exception as exc:
        print(f"torch.compile setup failed: {exc}; continuing uncompiled")
        return model, True, False


def run_once(dtype_name: str, device: str, should_compile: bool) -> None:
    torch.manual_seed(7)
    tokenizer, batch = build_sample_batch()
    config = PATERConfig.from_yaml(ROOT / "configs" / "tiny_smoke.yaml")
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)

    model = PATERForCausalLM(config).to(device)
    total, trainable = count_parameters(model)
    model, compile_attempted, compile_succeeded = maybe_compile(model, should_compile)

    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    labels = input_ids.clone()

    model.train()
    with maybe_autocast(device, dtype_name):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_aux=True,
        )
        loss = output.loss
    assert loss is not None
    loss.backward()

    print("PAT-ER smoke_forward complete")
    print(f"device: {device}")
    print(f"requested dtype: {dtype_name}")
    print(f"torch.compile attempted: {compile_attempted}")
    print(f"torch.compile succeeded: {compile_succeeded}")
    print(f"parameters: total={total:,} trainable={trainable:,}")
    print(f"loss: {float(loss.detach().cpu()):.6f}")
    print(f"logits: {shape_of(output.logits)}")
    print(f"event_registers: {shape_of(output.event_registers)}")
    print(f"primitive_registers: {shape_of(output.primitive_registers)}")
    print("aux_outputs:")
    for name, value in sorted(output.aux_outputs.items()):
        print(f"  {name}: {shape_of(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    try:
        run_once(args.dtype, device, args.compile)
    except Exception as exc:
        if args.dtype == "bf16":
            print(f"bf16 path failed: {exc}")
            print("rerunning in fp32")
            run_once("fp32", device, args.compile)
        else:
            raise


if __name__ == "__main__":
    main()
