"""Phase 2 Stage 1 — warm-start smoke.

Verifies the warm-start MECHANISM end-to-end, offline, against a shape-matched
random stand-in for Qwen3-0.6B (no 1.2GB download needed):

  1. load   — Qwen backbone weights land in the PAT-ER backbone params exactly;
              PAT-ER side-streams stay at their random init.
  2. embed  — shared leading vocab rows copied; PAT-ER special-token rows random.
  3. forward — logits are finite, shape [B, T, vocab].
  4. backward — grads flow to BOTH a backbone param and a PAT-ER-stream param
              (the warm-started model is trainable end-to-end).
  5. generate — greedy decode runs for a few steps.

This proves the plumbing. It does NOT claim transfer of real linguistic
knowledge — that requires the real Qwen3-0.6B weights (separate download).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from warmstart_from_qwen import (  # noqa: E402
    QWEN_SHARED_VOCAB_ROWS,
    build_warmstart_model,
    synthetic_qwen_state_dict,
)

CONFIG = "configs/pat_er_qwen3_warmstart.yaml"


def main() -> None:
    torch.manual_seed(0)
    qwen_state = synthetic_qwen_state_dict()
    model, report = build_warmstart_model(CONFIG, qwen_state)
    model.eval()
    params = dict(model.named_parameters())
    ok = True

    print("== 1. load report ==")
    print(f"  loaded backbone params : {report['n_loaded_params']}")
    print(f"  random-init params     : {report['n_random_init_params']} (PAT-ER streams/heads)")
    pre, tot = report["pretrained_numel"], report["total_numel"]
    print(f"  pretrained numel       : {pre/1e6:.1f}M / {tot/1e6:.1f}M ({100*pre/tot:.1f}%)")
    if report["missing_src"] or report["shape_mismatch"]:
        ok = False
        print(f"  !! missing={report['missing_src'][:3]} mismatch={report['shape_mismatch'][:3]}")

    # backbone params must equal the (synthetic) source exactly
    checks = {
        "layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_norm.weight": "model.layers.0.self_attn.q_norm.weight",
        "layers.27.ffn.base.gate_proj.weight": "model.layers.27.mlp.gate_proj.weight",
        "norm.weight": "model.norm.weight",
    }
    for pname, qkey in checks.items():
        same = torch.equal(params[pname], qwen_state[qkey])
        ok &= same
        print(f"  backbone {pname:42s} == Qwen src : {same}")

    print("== 2. embedding row split ==")
    emb, src_emb = params["embed_tokens.weight"], qwen_state["model.embed_tokens.weight"]
    n = QWEN_SHARED_VOCAB_ROWS
    shared_ok = torch.equal(emb[:n], src_emb[:n])
    specials_random = not torch.equal(emb[n:], src_emb[n : emb.shape[0]])
    ok &= shared_ok and specials_random
    print(f"  shared rows[:{n}] copied        : {shared_ok}")
    print(f"  PAT-ER specials[{n}:] random   : {specials_random} ({emb.shape[0]-n} rows)")

    print("== 3. forward ==")
    B, T = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (B, T))
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids, return_aux=True)
    logits_ok = tuple(out.logits.shape) == (B, T, model.config.vocab_size) and torch.isfinite(out.logits).all()
    ok &= bool(logits_ok)
    print(f"  logits shape {tuple(out.logits.shape)} finite : {logits_ok}")
    print(f"  aux heads emitted        : {sorted(out.aux_outputs.keys())[:6]} ...")
    print(f"  LM loss                  : {out.loss.item():.4f}")

    print("== 4. backward (end-to-end trainable) ==")
    model.train()
    model.zero_grad()
    out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids, return_aux=True)
    out.loss.backward()
    bb = params["layers.0.self_attn.q_proj.weight"]
    # side-state is zero-init (== backbone at init); the injection params still
    # receive gradient and open the side-stream as training starts.
    pat = next(params[n] for n in params if n.endswith("role_fuse.weight"))
    bb_grad = bb.grad is not None and torch.isfinite(bb.grad).all() and bb.grad.abs().sum() > 0
    pat_grad = pat.grad is not None and torch.isfinite(pat.grad).all() and pat.grad.abs().sum() > 0
    ok &= bool(bb_grad and pat_grad)
    print(f"  backbone q_proj grad flows  : {bool(bb_grad)}")
    print(f"  PAT-ER injection grad flows : {bool(pat_grad)}")

    print("== 5. generate (greedy, 8 steps) ==")
    model.eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        for _ in range(8):
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids), return_aux=False)
            nxt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    gen_ok = ids.shape[1] == 14
    ok &= gen_ok
    print(f"  generated to length {ids.shape[1]}     : {gen_ok}")

    print("\n" + ("SMOKE PASS" if ok else "SMOKE FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
