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


def shape_of(value: torch.Tensor) -> str:
    return f"{tuple(value.shape)} device={value.device} dtype={value.dtype}"


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return requested


def smoke_meta(batch_size: int, seq_len: int) -> None:
    cfg = PATERConfig.from_yaml(ROOT / "configs" / "pat_er_760m.yaml")
    with torch.device("meta"):
        model = PATERForCausalLM(cfg)
        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device="meta")
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device="meta")
        output = model(input_ids=input_ids, attention_mask=attention_mask, return_aux=True)

    total, trainable = count_parameters(model)
    print("PAT-ER 760M smoke complete")
    print("mode: meta shape smoke")
    print(f"parameters: total={total:,} trainable={trainable:,}")
    print(f"layers: {cfg.num_hidden_layers}")
    print(f"hidden_size: {cfg.hidden_size}")
    print(f"intermediate_size: {cfg.intermediate_size}")
    print(f"cross_attention_every_n_layers: {cfg.cross_attention_every_n_layers}")
    print(f"pressure_rank: {cfg.pressure_rank}")
    print(f"logits: {shape_of(output.logits)}")
    print(f"event_registers: {shape_of(output.event_registers)}")
    print(f"primitive_registers: {shape_of(output.primitive_registers)}")
    print("aux_outputs:")
    for name, value in sorted(output.aux_outputs.items()):
        print(f"  {name}: {shape_of(value)}")


@torch.no_grad()
def smoke_real(device: str, dtype_name: str, batch_size: int, seq_len: int) -> None:
    cfg = PATERConfig.from_yaml(ROOT / "configs" / "pat_er_760m.yaml")
    torch.manual_seed(17)
    model = PATERForCausalLM(cfg).to(device).eval()
    total, trainable = count_parameters(model)

    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    with maybe_autocast(device, dtype_name):
        output = model(input_ids=input_ids, attention_mask=attention_mask, return_aux=True)

    print("PAT-ER 760M smoke complete")
    print("mode: real forward")
    print(f"device: {device}")
    print(f"requested dtype: {dtype_name}")
    print(f"parameters: total={total:,} trainable={trainable:,}")
    print(f"logits: {shape_of(output.logits)}")
    print(f"event_registers: {shape_of(output.event_registers)}")
    print(f"primitive_registers: {shape_of(output.primitive_registers)}")
    print("aux_outputs:")
    for name, value in sorted(output.aux_outputs.items()):
        print(f"  {name}: {shape_of(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["meta", "real"], default="meta")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "meta":
        smoke_meta(batch_size=args.batch_size, seq_len=args.seq_len)
        return

    device = resolve_device(args.device)
    try:
        smoke_real(device=device, dtype_name=args.dtype, batch_size=args.batch_size, seq_len=args.seq_len)
    except Exception as exc:
        if args.dtype == "bf16":
            print(f"bf16 path failed: {exc}")
            print("rerunning in fp32")
            smoke_real(device=device, dtype_name="fp32", batch_size=args.batch_size, seq_len=args.seq_len)
        else:
            raise


if __name__ == "__main__":
    main()

