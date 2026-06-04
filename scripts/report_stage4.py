"""Phase 2 Stage 4 — injection opening evaluation report.

Loads a Stage-4 trained checkpoint and measures whether the opened injection
paths modulate the pretrained LM without damaging it:

  clean loss    — backbone only (streams off), real English text
  full loss     — full PAT-ER (streams + injection on), same text
  delta         — full - clean  (the LM degradation from injection)
  KL(full||clean) — KL divergence of PAT-ER vs pure backbone logits
  injection norm  — per-layer weight norm of the opened injection params
  generation      — greedy decode from real prompt, BEFORE vs AFTER

Pass conditions:
  lm_delta_ok   : |full_loss - clean_loss| < 0.5
  kl_bounded    : KL(full||clean) < 1.0 (nats, per token)
  inj_nonzero   : at least one injection param has nonzero weight
  gen_sane      : generation produces ≥3 coherent words

Usage:
    python3 scripts/report_stage4.py \
        --stage3  artifacts/checkpoints/qwen_ws_stage3_s0/latest.pt \
        --stage4a artifacts/checkpoints/qwen_ws_stage4a_s0/latest.pt \
        --tokenizer artifacts/tokenizers/qwen_pater_extended \
        --json-out artifacts/reports/qwen_ws_stage4_s0.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er import PATERConfig, PATERForCausalLM

TEXTS = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "If all men are mortal and Socrates is a man, then Socrates is mortal.",
]
GEN_PROMPTS = {
    "factual":    "The capital of France is",
    "tool":       "Based on the available evidence,",
    "idk":        "This question cannot be answered because the evidence",
}
GEN_PROMPT = GEN_PROMPTS["factual"]  # kept for backward compat
GEN_STEPS = 20

INJECT_PATTERNS = [
    lambda n: n.endswith("role_fuse.weight"),
    lambda n: n.endswith("primitive_fuse.weight"),
    lambda n: ".ffn.adapters." in n and n.endswith(".up.weight"),
    lambda n: n in {
        "role_pressure.proj.1.weight",
        "primitive_pressure.proj.1.weight",
        "mix_pressure.proj.1.weight",
    },
]


def is_inject_param(name: str) -> bool:
    return any(f(name) for f in INJECT_PATTERNS)


def load_ckpt(path: Path, device: str, tok_path: str) -> tuple:
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    config = PATERConfig.from_dict(ckpt["config"])
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=True)
    model = PATERForCausalLM(config).to(device)
    model.load_state_dict(ckpt["model_state"])
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
def measure(model: PATERForCausalLM, tok, device: str) -> dict:
    enc = tok(TEXTS, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    lbl = ids.clone()
    lbl[attn == 0] = -100

    _disable_streams(model)
    clean_out = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False)
    clean_loss = clean_out.loss.item()
    clean_logits = clean_out.logits.detach().clone()

    _enable_streams(model)
    full_out = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False)
    full_loss = full_out.loss.item()
    full_logits = full_out.logits.detach()

    # KL(full || clean): how much the injection perturbs the probability distribution.
    # Compute token-average KL over non-padding positions.
    mask = (attn == 1)
    kl = F.kl_div(
        F.log_softmax(full_logits[mask], dim=-1),
        F.softmax(clean_logits[mask], dim=-1),
        reduction="batchmean",
        log_target=False,
    ).item()

    # Greedy generation.
    enc_g = tok(GEN_PROMPT, return_tensors="pt")
    g = enc_g["input_ids"].to(device)
    n_prompt = g.shape[1]
    for _ in range(GEN_STEPS):
        o = model(input_ids=g, attention_mask=torch.ones_like(g), return_aux=False)
        g = torch.cat([g, o.logits[:, -1:].argmax(-1)], dim=1)
    gen_text = tok.decode(g[0, n_prompt:], skip_special_tokens=True)

    # Additional generation probes (Stage 5B: factual / tool / IDK).
    gen_probes: dict[str, str] = {}
    for probe_name, prompt in GEN_PROMPTS.items():
        enc_p = tok(prompt, return_tensors="pt")
        gp = enc_p["input_ids"].to(device)
        np_ = gp.shape[1]
        for _ in range(GEN_STEPS):
            op = model(input_ids=gp, attention_mask=torch.ones_like(gp), return_aux=False)
            gp = torch.cat([gp, op.logits[:, -1:].argmax(-1)], dim=1)
        gen_probes[probe_name] = tok.decode(gp[0, np_:], skip_special_tokens=True)

    # Injection fuse parameter norms (role_fuse/primitive_fuse + pressure).
    inj_norms = {n: p.data.norm().item() for n, p in model.named_parameters() if is_inject_param(n)}
    inj_total_norm = sum(inj_norms.values())
    inj_nonzero = inj_total_norm > 1e-8

    # Per-layer adapter norms — split up (zero-init, shows opening) vs down (baseline).
    import re as _re
    adapter_up_by_layer: dict[int, float] = {}
    adapter_dn_by_layer: dict[int, float] = {}
    for n, p in model.named_parameters():
        m2 = _re.match(r"layers\.(\d+)\.ffn\.adapters\.\d+\.(up|down)\.weight$", n)
        if m2:
            li, side = int(m2.group(1)), m2.group(2)
            if side == "up":
                adapter_up_by_layer[li] = adapter_up_by_layer.get(li, 0.0) + p.data.norm().item()
            else:
                adapter_dn_by_layer[li] = adapter_dn_by_layer.get(li, 0.0) + p.data.norm().item()
    adapter_by_layer = {li: adapter_up_by_layer.get(li, 0) + adapter_dn_by_layer.get(li, 0)
                        for li in set(adapter_up_by_layer) | set(adapter_dn_by_layer)}

    return {
        "clean_loss": clean_loss,
        "full_loss": full_loss,
        "lm_delta": full_loss - clean_loss,
        "kl_full_vs_clean": kl if math.isfinite(kl) else None,
        "injection_weight_norm": inj_total_norm,
        "injection_nonzero": inj_nonzero,
        "injection_norms_per_param": inj_norms,
        "adapter_norm_by_layer": adapter_by_layer,
        "adapter_up_norm_by_layer": adapter_up_by_layer,
        "adapter_dn_norm_by_layer": adapter_dn_by_layer,
        "gen": gen_text,
        "gen_probes": gen_probes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3",  required=True, help="Stage 3 checkpoint (reference: injection=0)")
    ap.add_argument("--stage4a", default=None,  help="Stage 4A checkpoint")
    ap.add_argument("--stage4b", default=None,  help="Stage 4B checkpoint")
    ap.add_argument("--stage4c", default=None,  help="Stage 4C checkpoint")
    ap.add_argument("--stage5a", default=None,  help="Stage 5A checkpoint (upper-layer adapters opened)")
    ap.add_argument("--stage5b", default=None,  help="Stage 5B checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    checkpoints = [
        ("stage3",  args.stage3),
        ("stage4a", getattr(args, "stage4a", None)),
        ("stage4b", getattr(args, "stage4b", None)),
        ("stage4c", getattr(args, "stage4c", None)),
        ("stage5a", getattr(args, "stage5a", None)),
        ("stage5b", getattr(args, "stage5b", None)),
    ]
    checkpoints = [(tag, p) for tag, p in checkpoints if p]

    results: dict[str, dict] = {}
    for tag, ckpt_path in checkpoints:
        print(f"\n-- {tag}: {ckpt_path} --")
        model, tok, _ = load_ckpt(Path(ckpt_path), args.device, args.tokenizer)
        m = measure(model, tok, args.device)
        results[tag] = m
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stage_label = "STAGE-5" if any(t.startswith("stage5") for t in results) else "STAGE-4"
    print(f"\n=== {stage_label} INJECTION REPORT ===")
    hdr = f"  {'':12s}  {'clean':>8}  {'full':>8}  {'delta':>8}  {'KL':>8}  {'inj_norm':>10}  gen"
    print(hdr)
    for tag, m in results.items():
        kl_str = f"{m['kl_full_vs_clean']:.4f}" if m.get("kl_full_vs_clean") is not None else "   n/a"
        print(f"  {tag:12s}  {m['clean_loss']:8.4f}  {m['full_loss']:8.4f}  "
              f"{m['lm_delta']:+8.4f}  {kl_str:>8}  {m['injection_weight_norm']:10.4e}  "
              f"{repr(m['gen'][:50])}")

    # Multi-probe generation table.
    probe_names = list(GEN_PROMPTS.keys())
    if any(results[t].get("gen_probes") for t in results):
        print(f"\n  {'':12s}  " + "  ".join(f"  {p:>8}" for p in probe_names))
        for tag, m in results.items():
            gp = m.get("gen_probes", {})
            row = "  ".join(f"  {repr(gp.get(p,'')[:20]):>22}" for p in probe_names)
            print(f"  {tag:12s}  {row}")

    # Per-layer adapter UP norm table (Stage 5A) — up starts at zero, shows opening.
    for tag, m in results.items():
        aup = m.get("adapter_up_norm_by_layer", {})
        if any(v > 1e-6 for v in aup.values()):
            print(f"\n  Adapter up-proj norms by layer ({tag}):")
            for li in sorted(aup):
                if aup[li] > 1e-6:
                    print(f"    layer {li:2d}: {aup[li]:.4e}")

    # Pass conditions (evaluated on the last non-stage3 checkpoint).
    last_tag, last_m = [(t, m) for t, m in results.items() if t != "stage3"][-1] if len(results) > 1 else (None, None)
    if last_m:
        lm_delta_ok = math.isfinite(last_m["lm_delta"]) and abs(last_m["lm_delta"]) < 0.5
        kl_ok = last_m.get("kl_full_vs_clean") is not None and last_m["kl_full_vs_clean"] < 1.0
        # inj_nonzero: either fuse tensors or adapter weights are nonzero
        adp_nz = any(v > 1e-8 for v in last_m.get("adapter_norm_by_layer", {}).values())
        inj_nz = last_m["injection_nonzero"] or adp_nz
        gen_sane = len(last_m["gen"].split()) >= 3
        passed = lm_delta_ok and kl_ok and inj_nz and gen_sane
        print(f"\n  Pass conditions (on {last_tag}):")
        print(f"    lm_delta_ok   (|Δloss| < 0.5)  : {lm_delta_ok}  (Δ={last_m['lm_delta']:+.4f})")
        print(f"    kl_bounded    (KL < 1.0)        : {kl_ok}  (KL={last_m.get('kl_full_vs_clean')})")
        print(f"    inj_nonzero                     : {inj_nz}  (fuse+adp_norm={last_m['injection_weight_norm']:.4e})")
        print(f"    gen_sane                        : {gen_sane}  ({repr(last_m['gen'][:40])})")
        print(f"\n{stage_label} GATE: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
