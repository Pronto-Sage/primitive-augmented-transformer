"""Phase 2 Stage 3 — real-text LM degradation + generation sanity report.

Loads the warmstart-init (BEFORE) and trained (AFTER) Stage-3 checkpoints,
measures LM loss on real English text under two modes for each:

  clean  — PAT-ER side-streams disabled (pure pretrained backbone)
  full   — all PAT-ER streams enabled (the actual forward path after training)

Pass conditions (printed and written to --json-out):
  1. backbone_frozen   : clean_before ≈ clean_after  (backbone unchanged by design)
  2. lm_not_degraded   : full_after < full_before + 0.5  (side-state doesn't blow LM)
  3. full_after_finite  : full_after is a finite number
  4. generation_sane    : model generates coherent text after training

Usage:
    python3 scripts/report_warmstart_realtext.py \
        --init  artifacts/checkpoints/qwen_ws_stage3_init_s0/latest.pt \
        --trained artifacts/checkpoints/qwen_ws_stage3_s0/latest.pt \
        --tokenizer artifacts/tokenizers/qwen_pater_extended \
        --json-out artifacts/reports/qwen_ws_stage3_realtext_s0.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er import PATERConfig, PATERForCausalLM

# Canonical real-text sentences used in Stage-2 parity test.
TEXTS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "If all men are mortal and Socrates is a man, then Socrates is mortal.",
]
GEN_PROMPT = "The capital of France is"
GEN_STEPS = 20


def load_ckpt(path: Path, device: str, tokenizer_path: str | None):
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    config = PATERConfig.from_dict(checkpoint["config"])
    tok_dir = tokenizer_path or checkpoint.get("tokenizer_path")
    if not tok_dir:
        raise RuntimeError("no tokenizer path in checkpoint or --tokenizer flag")
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True, use_fast=True)
    model = PATERForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, config


def _disable_streams(model: PATERForCausalLM) -> None:
    model.config.use_event_stream = False
    model.config.use_primitive_stream = False
    model.config.use_vocab_pressure = False
    for layer in model.layers:
        layer.ffn.enabled = False


def _enable_streams(model: PATERForCausalLM) -> None:
    model.config.use_event_stream = True
    model.config.use_primitive_stream = True
    model.config.use_vocab_pressure = True
    for layer in model.layers:
        layer.ffn.enabled = True


@torch.no_grad()
def lm_loss(model: PATERForCausalLM, ids: torch.Tensor, attn: torch.Tensor) -> float:
    lbl = ids.clone()
    lbl[attn == 0] = -100
    out = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False)
    return out.loss.item()


@torch.no_grad()
def generate_text(
    model: PATERForCausalLM, tokenizer, prompt: str, steps: int, device: str
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    n_prompt = ids.shape[1]
    for _ in range(steps):
        attn = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=attn, return_aux=False)
        nxt = out.logits[:, -1:].argmax(-1)
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True)


def measure(model: PATERForCausalLM, tokenizer, device: str) -> dict:
    enc = tokenizer(TEXTS, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    _disable_streams(model)
    clean = lm_loss(model, ids, attn)

    _enable_streams(model)
    full = lm_loss(model, ids, attn)

    gen = generate_text(model, tokenizer, GEN_PROMPT, GEN_STEPS, device)
    return {"clean_loss": clean, "full_loss": full, "gen": gen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="BEFORE checkpoint (warmstart-init, 0 steps)")
    ap.add_argument("--trained", required=True, help="AFTER checkpoint (Stage-3 trained)")
    ap.add_argument("--tokenizer", default=None, help="path to qwen_pater_extended tokenizer dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok_path = args.tokenizer

    print("=== report_warmstart_realtext.py ===")
    print(f"device : {args.device}")
    print(f"BEFORE : {args.init}")
    print(f"AFTER  : {args.trained}")
    print()

    print("-- BEFORE (warmstart-init) --")
    model_b, tok_b, _ = load_ckpt(Path(args.init), args.device, tok_path)
    before = measure(model_b, tok_b, args.device)
    del model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("-- AFTER (Stage-3 trained) --")
    model_a, tok_a, _ = load_ckpt(Path(args.trained), args.device, tok_path)
    after = measure(model_a, tok_a, args.device)
    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- pass conditions ----
    backbone_frozen = math.isfinite(before["clean_loss"]) and math.isfinite(after["clean_loss"]) and \
        abs(after["clean_loss"] - before["clean_loss"]) < 0.05
    full_after_finite = math.isfinite(after["full_loss"])
    lm_not_degraded = full_after_finite and (after["full_loss"] < before["full_loss"] + 0.5)
    generation_sane = len(after["gen"].split()) >= 3

    passed = backbone_frozen and full_after_finite and lm_not_degraded and generation_sane

    # ---- report ----
    print()
    print("=== REAL-TEXT LM DEGRADATION REPORT ===")
    print(f"  {'':30s}  {'BEFORE':>10}  {'AFTER':>10}  {'DELTA':>10}")
    c_delta = after["clean_loss"] - before["clean_loss"]
    f_delta = after["full_loss"] - before["full_loss"]
    print(f"  {'clean backbone loss':30s}  {before['clean_loss']:10.4f}  {after['clean_loss']:10.4f}  {c_delta:+10.4f}")
    print(f"  {'full PAT-ER loss':30s}  {before['full_loss']:10.4f}  {after['full_loss']:10.4f}  {f_delta:+10.4f}")
    print()
    print("  Generation (BEFORE):")
    print(f"    prompt : {repr(GEN_PROMPT)}")
    print(f"    output : {repr(before['gen'])}")
    print("  Generation (AFTER):")
    print(f"    prompt : {repr(GEN_PROMPT)}")
    print(f"    output : {repr(after['gen'])}")
    print()
    print("  Pass conditions:")
    print(f"    backbone_frozen   (|Δclean| < 0.05) : {backbone_frozen}  (Δ={c_delta:+.4f})")
    print(f"    full_after_finite                   : {full_after_finite}")
    print(f"    lm_not_degraded   (Δfull < +0.5)   : {lm_not_degraded}  (Δ={f_delta:+.4f})")
    print(f"    generation_sane   (≥3 words)        : {generation_sane}  ({repr(after['gen'][:60])}...)")
    print()
    print("STAGE-3 REALTEXT:", "PASS" if passed else "FAIL")

    result = {
        "before": before,
        "after": after,
        "delta_clean": c_delta,
        "delta_full": f_delta,
        "pass_conditions": {
            "backbone_frozen": backbone_frozen,
            "full_after_finite": full_after_finite,
            "lm_not_degraded": lm_not_degraded,
            "generation_sane": generation_sane,
        },
        "passed": passed,
        "gen_prompt": GEN_PROMPT,
        "gen_steps": GEN_STEPS,
        "texts": TEXTS,
    }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"JSON report -> {args.json_out}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
