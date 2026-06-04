"""Phase 2 Stage 2 — validate the REAL Qwen3-0.6B warm-start load.

Loads the actual Qwen3-0.6B ``model.safetensors`` into the PAT-ER warm-start
backbone and checks the user's pass conditions:

  A. coverage   — every Qwen backbone tensor loads, no shape drift, PAT-ER-only
                  tensors stay random-init.
  B. parity     — with PAT-ER side-computation bypassed (streams off, FFN
                  adapters off, no vocab pressure) the model reduces to a pure
                  Qwen decoder; its logits match HF Qwen3-0.6B on shared vocab
                  columns (the decisive correctness proof for the reimplemented
                  attention: head_dim 128, q/k-norm, GQA, RoPE).
  C. real loss  — clean-backbone LM loss on real English text is at *pretrained*
                  scale (low), not random-init scale; the full PAT-ER-wrapped
                  loss is finite and elevated (random side-state, fixed by Stage 3).
  D. train      — forward/backward/generate run; grads reach both backbone and
                  PAT-ER-stream params.

Run: python3 scripts/validate_warmstart_real.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from warmstart_from_qwen import (  # noqa: E402
    QWEN_SHARED_VOCAB_ROWS,
    build_warmstart_model,
    load_safetensors_state_dict,
    qwen_backbone_name_map,
)

CONFIG = "configs/pat_er_qwen3_warmstart.yaml"
QWEN_REPO = "Qwen/Qwen3-0.6B"
TEXTS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "If all men are mortal and Socrates is a man, then Socrates is mortal.",
]


def _qwen_weights_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(QWEN_REPO, "model.safetensors")


def main() -> None:
    torch.manual_seed(0)
    ok = True
    wpath = _qwen_weights_path()
    qwen_state = load_safetensors_state_dict(Path(wpath))
    model, report = build_warmstart_model(CONFIG, qwen_state)
    model.eval()
    params = dict(model.named_parameters())

    print("== A. coverage ==")
    name_map = qwen_backbone_name_map(model.config.num_hidden_layers)
    expected = set(name_map.values()) | {"model.embed_tokens.weight"}
    qkeys = set(qwen_state)
    unconsumed = sorted(qkeys - expected)
    print(f"  Qwen tensors in file        : {len(qkeys)}")
    print(f"  mapped to PAT-ER backbone   : {report['n_loaded_params']}")
    print(f"  unconsumed Qwen keys        : {unconsumed} (expect only tied lm_head)")
    print(f"  PAT-ER random-init params   : {report['n_random_init_params']}")
    pre, tot = report["pretrained_numel"], report["total_numel"]
    print(f"  pretrained numel            : {pre/1e6:.1f}M / {tot/1e6:.1f}M ({100*pre/tot:.1f}%)")
    cov_ok = (
        not report["missing_src"]
        and not report["shape_mismatch"]
        and unconsumed == ["lm_head.weight"]
    )
    ok &= cov_ok
    print(f"  coverage clean (no drift)   : {cov_ok}")

    # Reduce the warm-start model to a pure Qwen decoder for parity / clean loss.
    model.config.use_event_stream = False
    model.config.use_primitive_stream = False
    model.config.use_vocab_pressure = False
    for layer in model.layers:
        layer.ffn.enabled = False

    # tokenizer (extended) for shared-vocab ids
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("artifacts/tokenizers/qwen_pater_extended")
    enc = tok(TEXTS, return_tensors="pt", padding=True)
    ids, attn = enc["input_ids"], enc["attention_mask"]
    assert int(ids.max()) < QWEN_SHARED_VOCAB_ROWS, "real text must stay in shared vocab"

    print("== B. parity vs HF Qwen3-0.6B (pure-backbone) ==")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(QWEN_REPO, torch_dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        pat_logits = model(input_ids=ids, attention_mask=attn, return_aux=False).logits
        hf_logits = hf(input_ids=ids, attention_mask=attn).logits
    n = QWEN_SHARED_VOCAB_ROWS
    diff = (pat_logits[:, :, :n] - hf_logits[:, :, :n]).abs()
    max_abs = diff.max().item()
    pat_top1 = pat_logits[:, :, :n].argmax(-1)
    hf_top1 = hf_logits[:, :, :n].argmax(-1)
    top1_match = (pat_top1 == hf_top1).float().mean().item()
    parity_ok = max_abs < 1e-2 and top1_match > 0.999
    ok &= parity_ok
    print(f"  max |logit diff| (shared cols): {max_abs:.2e}")
    print(f"  next-token top-1 agreement    : {100*top1_match:.2f}%")
    print(f"  parity OK                     : {parity_ok}")

    print("== C. real-text LM loss ==")
    lbl = ids.clone()
    lbl[attn == 0] = -100
    with torch.no_grad():
        clean = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False).loss.item()
    # full PAT-ER warm-start path (streams + adapters + pressure back on)
    model.config.use_event_stream = True
    model.config.use_primitive_stream = True
    model.config.use_vocab_pressure = True
    for layer in model.layers:
        layer.ffn.enabled = True
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False).loss.item()
    import math

    rand_scale = math.log(model.config.vocab_size)
    # With zero-init side-state, the full warm-start forward == backbone at init.
    full_ok = math.isfinite(full) and abs(full - clean) < 0.05
    clean_ok = clean < 6.0 and full_ok
    ok &= clean_ok
    print(f"  clean-backbone LM loss      : {clean:.3f}  (random-init scale ~{rand_scale:.1f})")
    print(f"  full PAT-ER warm-start loss : {full:.3f}  (== backbone at init via zero-init side-state)")
    print(f"  loss at pretrained scale    : {clean_ok}")

    print("== D. backward + generate ==")
    model.train()
    model.zero_grad()
    out = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=True)
    out.loss.backward()
    bb = params["layers.0.self_attn.q_proj.weight"]
    # The side-stream ENTRY params (injection / adapter-up / pressure) receive
    # gradient at init and "open" the side-stream as training starts; deeper
    # registers get their signal from the aux-head losses in Stage-3 training.
    inj = next(params[k] for k in params if k.endswith("role_fuse.weight"))
    reg = next(params[k] for k in params if k.startswith("register_bank."))
    bb_g = bb.grad is not None and torch.isfinite(bb.grad).all() and bb.grad.abs().sum() > 0
    inj_g = inj.grad is not None and torch.isfinite(inj.grad).all() and inj.grad.abs().sum() > 0
    reg_g = reg.grad is not None and reg.grad.abs().sum() > 0
    ok &= bool(bb_g and inj_g)
    print(f"  backbone grad flows          : {bool(bb_g)}")
    print(f"  PAT-ER injection grad flows  : {bool(inj_g)} (opens the side-stream)")
    print(f"  register_bank grad (LM-only) : {bool(reg_g)} (0 by design; trained via aux losses)")
    model.eval()
    g = ids[:1, :5].clone()
    with torch.no_grad():
        for _ in range(8):
            o = model(input_ids=g, attention_mask=torch.ones_like(g), return_aux=False)
            g = torch.cat([g, o.logits[:, -1:].argmax(-1)], dim=1)
    gen_ok = g.shape[1] == 13
    ok &= gen_ok
    print(f"  greedy generate 8 steps     : {gen_ok}")
    print("  decoded:", repr(tok.decode(g[0, 5:])))

    print("\n" + ("STAGE-2 PASS" if ok else "STAGE-2 FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
