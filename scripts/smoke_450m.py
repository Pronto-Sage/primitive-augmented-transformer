#!/usr/bin/env python3
"""GPU smoke for the ~450M PAT-ER config on the extended Qwen tokenizer.

Exercises the full path the 450M training gate needs, on real CUDA:

- forward (logits + event/primitive register + all aux-head shapes);
- backward (LM loss -> backward -> grad-norm -> one AdamW step);
- greedy generate (autoregressive argmax decode, no KV cache, like smoke_generate);
- CUDA memory report (peak allocated / reserved).

Defaults to configs/pat_er_450m_qwen.yaml and the extended Qwen tokenizer so the
real vocab (151782) and context (1024) are used. Loads the tokenizer offline; no
downloads. Use --device cuda for the real GPU smoke (falls back to CPU if no GPU).

Usage:
    python3 scripts/smoke_450m.py --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", message="Failed to initialize NumPy*")

import torch

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import count_parameters, maybe_autocast
import pater_hf_tokenizer as H

DEFAULT_CONFIG = ROOT / "configs" / "pat_er_450m_qwen.yaml"
DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizers" / "qwen_pater_extended"
PROMPT = (
    "<pat_er><task>Determine whether the conclusion follows from the premises.</task>"
    "<context>All compilers translate source code. GCC is a compiler.</context>"
    "<question>Does GCC translate source code?</question>"
)


def shape_of(value: torch.Tensor) -> str:
    return f"{tuple(value.shape)} device={value.device} dtype={value.dtype}"


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return requested


def grad_global_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return total ** 0.5


def run(config_path: Path, tokenizer_path: Path, device: str, dtype_name: str,
        batch_size: int, seq_len: int, max_new_tokens: int) -> None:
    torch.manual_seed(17)
    cfg = PATERConfig.from_yaml(config_path)
    tokenizer = H.load_pater_hf_tokenizer(tokenizer_path)
    cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)

    model = PATERForCausalLM(cfg).to(device)
    total, trainable = count_parameters(model)
    print("PAT-ER 450M GPU smoke")
    print(f"config: {config_path.name}  tokenizer: {tokenizer_path.name} (vocab={tokenizer.vocab_size})")
    print(f"device: {device}  requested dtype: {dtype_name}")
    print(f"parameters: total={total:,} trainable={trainable:,}")
    print(f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"inter={cfg.intermediate_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"ctx={cfg.max_position_embeddings} pressure_rank={cfg.pressure_rank}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- forward (+ aux shapes) ----
    model.train()
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    with maybe_autocast(device, dtype_name):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_aux=True)
    print("\n[forward]")
    print(f"  logits: {shape_of(out.logits)}")
    print(f"  event_registers: {shape_of(out.event_registers)}")
    print(f"  primitive_registers: {shape_of(out.primitive_registers)}")
    print(f"  lm_loss: {float(out.loss):.4f}")
    print("  aux_outputs:")
    for name, value in sorted(out.aux_outputs.items()):
        print(f"    {name}: {shape_of(value)}")

    # ---- backward (+ one optimizer step) ----
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    gnorm = grad_global_norm(model)
    opt.step()
    with maybe_autocast(device, dtype_name):
        out2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_aux=False)
    print("\n[backward]")
    print(f"  grad global-norm: {gnorm:.4f}")
    print(f"  lm_loss before step: {float(out.loss):.4f}  after step: {float(out2.loss):.4f}")
    print(f"  step reduced loss: {float(out2.loss) < float(out.loss)}")

    # ---- greedy generate ----
    model.eval()
    ids = torch.tensor([tokenizer.encode(PROMPT, add_special_tokens=True)], dtype=torch.long, device=device)
    prompt_len = ids.shape[1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            amask = torch.ones_like(ids, dtype=torch.long, device=device)
            with maybe_autocast(device, dtype_name):
                gout = model(input_ids=ids, attention_mask=amask, return_aux=False)
                nxt = torch.argmax(gout.logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if tokenizer.eos_token_id is not None and int(nxt.item()) == tokenizer.eos_token_id:
                break
    new_ids = ids[0, prompt_len:].detach().cpu().tolist()
    print("\n[generate]")
    print(f"  prompt tokens: {prompt_len}  new tokens: {len(new_ids)}")
    print(f"  generated ids: {new_ids}")
    print(f"  generated text: {tokenizer.decode(new_ids)!r}")

    # ---- memory report ----
    print("\n[memory]")
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"  peak allocated: {peak_alloc:.2f} GiB  peak reserved: {peak_resv:.2f} GiB")
        print(f"  device: {torch.cuda.get_device_name(0)}")
    else:
        print("  (cpu run; no CUDA memory stats)")
    print("\nSMOKE_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="450M PAT-ER GPU smoke (forward/backward/generate/memory).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    device = resolve_device(args.device)
    try:
        run(args.config, args.tokenizer, device, args.dtype, args.batch_size, args.seq_len, args.max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        if args.dtype == "bf16":
            print(f"bf16 path failed: {exc}\nrerunning in fp32")
            run(args.config, args.tokenizer, device, "fp32", args.batch_size, args.seq_len, args.max_new_tokens)
        else:
            raise


if __name__ == "__main__":
    main()
