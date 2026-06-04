"""Phase 2 Stage 1 — warm-start loader: Qwen3-0.6B decoder -> PAT-ER backbone.

Maps a pretrained Qwen3-0.6B ``state_dict`` onto the PAT-ER token backbone
(self-attn q/k/v/o + q/k-norm, SwiGLU FFN, input/post-attn/final RMSNorm, and
the shared leading rows of the tied embedding) and leaves every PAT-ER-specific
module (event-role + primitive registers, cross-attention, role/primitive FFN
adapters, pressure heads, aux heads) at its random init.

The backbone of ``configs/pat_er_qwen3_warmstart.yaml`` is shaped to match
Qwen3-0.6B exactly (28 layers, head_dim 128 via ``attn_head_dim``, 8 KV heads,
q/k-norm on, intermediate 3072), so the copy is shape-checked and strict.

Two weight sources:
  --qwen-weights PATH   real Qwen3-0.6B ``model.safetensors`` (needs download)
  --synthetic           a shape-matched RANDOM stand-in (offline smoke only)

``--synthetic`` exercises the entire mapping + load path without the ~1.2GB
download, so Stage-1 forward/backward/generate can be verified offline. It does
NOT carry real linguistic knowledge — that is what the real download is for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pat_er.configuration_pat_er import PATERConfig
from pat_er.modeling_pat_er import PATERForCausalLM

# Reference Qwen3-0.6B backbone dims (Qwen/Qwen3-0.6B config.json).
QWEN3_0_6B = {
    "hidden": 1024,
    "layers": 28,
    "n_heads": 16,
    "n_kv": 8,
    "head_dim": 128,
    "intermediate": 3072,
    "vocab": 151936,
}
# Qwen base tokenizer length (151643 BPE + 26 Qwen specials). The PAT-ER
# extended tokenizer is these 151669 rows + 113 PAT-ER special tokens on top;
# the 113 specials are random-init, the 151669 shared rows copy from Qwen.
QWEN_SHARED_VOCAB_ROWS = 151669


def qwen_backbone_name_map(num_layers: int) -> dict[str, str]:
    """PAT-ER backbone param name -> Qwen3 state_dict key (excludes embedding)."""

    name_map = {"norm.weight": "model.norm.weight"}
    for i in range(num_layers):
        p = f"layers.{i}"
        q = f"model.layers.{i}"
        name_map[f"{p}.input_norm.weight"] = f"{q}.input_layernorm.weight"
        name_map[f"{p}.ffn_norm.weight"] = f"{q}.post_attention_layernorm.weight"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name_map[f"{p}.self_attn.{proj}.weight"] = f"{q}.self_attn.{proj}.weight"
        name_map[f"{p}.self_attn.q_norm.weight"] = f"{q}.self_attn.q_norm.weight"
        name_map[f"{p}.self_attn.k_norm.weight"] = f"{q}.self_attn.k_norm.weight"
        name_map[f"{p}.ffn.base.gate_proj.weight"] = f"{q}.mlp.gate_proj.weight"
        name_map[f"{p}.ffn.base.up_proj.weight"] = f"{q}.mlp.up_proj.weight"
        name_map[f"{p}.ffn.base.down_proj.weight"] = f"{q}.mlp.down_proj.weight"
    return name_map


def synthetic_qwen_state_dict(ref: dict = QWEN3_0_6B) -> dict[str, torch.Tensor]:
    """Random, shape-matched stand-in for the real Qwen3-0.6B state_dict.

    Same keys/shapes the real safetensors would have, so the loader and its
    shape checks are exercised offline. Deterministic per-key (no global RNG
    state assumptions) so the smoke is reproducible.
    """

    h, hd, nh, nkv = ref["hidden"], ref["head_dim"], ref["n_heads"], ref["n_kv"]
    inter, vocab, layers = ref["intermediate"], ref["vocab"], ref["layers"]
    gen = torch.Generator().manual_seed(0)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * 0.02

    sd: dict[str, torch.Tensor] = {"model.embed_tokens.weight": rnd(vocab, h), "model.norm.weight": torch.ones(h)}
    for i in range(layers):
        q = f"model.layers.{i}"
        sd[f"{q}.input_layernorm.weight"] = torch.ones(h)
        sd[f"{q}.post_attention_layernorm.weight"] = torch.ones(h)
        sd[f"{q}.self_attn.q_proj.weight"] = rnd(nh * hd, h)
        sd[f"{q}.self_attn.k_proj.weight"] = rnd(nkv * hd, h)
        sd[f"{q}.self_attn.v_proj.weight"] = rnd(nkv * hd, h)
        sd[f"{q}.self_attn.o_proj.weight"] = rnd(h, nh * hd)
        sd[f"{q}.self_attn.q_norm.weight"] = torch.ones(hd)
        sd[f"{q}.self_attn.k_norm.weight"] = torch.ones(hd)
        sd[f"{q}.mlp.gate_proj.weight"] = rnd(inter, h)
        sd[f"{q}.mlp.up_proj.weight"] = rnd(inter, h)
        sd[f"{q}.mlp.down_proj.weight"] = rnd(h, inter)
    return sd


def load_safetensors_state_dict(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def resolve_qwen_weights(spec: str = "auto") -> str:
    """Return a path to Qwen3-0.6B model.safetensors. ``auto`` resolves it from
    the offline HF cache (the weights are downloaded once, gitignored)."""

    if spec and spec != "auto":
        return spec
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from huggingface_hub import hf_hub_download

    return hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors")


def load_qwen_into_pater(
    model: PATERForCausalLM,
    qwen_state: dict[str, torch.Tensor],
    shared_vocab_rows: int = QWEN_SHARED_VOCAB_ROWS,
) -> dict[str, object]:
    """Copy Qwen backbone weights into ``model`` in-place. Returns a report."""

    params = dict(model.named_parameters())
    name_map = qwen_backbone_name_map(model.config.num_hidden_layers)

    loaded, missing_src, shape_mismatch = [], [], []
    with torch.no_grad():
        # Embedding: copy the shared leading rows, leave PAT-ER specials random.
        emb = params["embed_tokens.weight"]
        src_emb = qwen_state.get("model.embed_tokens.weight")
        if src_emb is None:
            missing_src.append("model.embed_tokens.weight")
        else:
            n = min(shared_vocab_rows, emb.shape[0], src_emb.shape[0])
            emb[:n].copy_(src_emb[:n])
            loaded.append(("embed_tokens.weight", f"rows[:{n}]"))

        for pater_name, qwen_key in name_map.items():
            param = params.get(pater_name)
            src = qwen_state.get(qwen_key)
            if param is None:
                continue
            if src is None:
                missing_src.append(qwen_key)
                continue
            if tuple(param.shape) != tuple(src.shape):
                shape_mismatch.append((pater_name, tuple(param.shape), tuple(src.shape)))
                continue
            param.copy_(src)
            loaded.append((pater_name, "full"))

    backbone_names = set(name_map) | {"embed_tokens.weight"}
    random_init = [n for n in params if n not in backbone_names]
    pretrained_numel = sum(params[n].numel() for n, _ in loaded if n != "embed_tokens.weight")
    pretrained_numel += min(shared_vocab_rows, params["embed_tokens.weight"].shape[0]) * model.config.hidden_size
    return {
        "loaded": loaded,
        "random_init": random_init,
        "missing_src": missing_src,
        "shape_mismatch": shape_mismatch,
        "n_loaded_params": len(loaded),
        "n_random_init_params": len(random_init),
        "pretrained_numel": pretrained_numel,
        "total_numel": sum(p.numel() for p in model.parameters()),
        "shared_vocab_rows": shared_vocab_rows,
    }


def zero_init_warmstart_side_state(model: PATERForCausalLM) -> list[str]:
    """Zero the PAT-ER -> token-stream entry paths so the warm-started model
    starts *exactly* equal to the pretrained backbone.

    With a pretrained (properly-scaled) backbone, random side-state injections
    blow up the residual stream (nan). Zeroing only the injection/pressure
    OUTPUT projections — not the whole side-stream — means at init every PAT-ER
    contribution to the token stream is 0, so forward == pure Qwen decoder, and
    Stage-3 training grows the side-state from zero (standard adapter init).
    The injection params themselves still RECEIVE gradient (they multiply
    nonzero activations), so the side-stream opens up as soon as training starts.
    """

    zeroed = []
    pressure_outs = {
        "role_pressure.proj.1.weight",
        "primitive_pressure.proj.1.weight",
        "mix_pressure.proj.1.weight",
    }
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (
                name.endswith("role_fuse.weight")
                or name.endswith("primitive_fuse.weight")
                or (".ffn.adapters." in name and name.endswith(".up.weight"))
                or name in pressure_outs
            ):
                p.zero_()
                zeroed.append(name)
    return zeroed


def build_warmstart_model(
    config_path: str,
    qwen_state: dict[str, torch.Tensor],
    zero_init_side_state: bool = True,
):
    cfg = PATERConfig.from_yaml(config_path)
    model = PATERForCausalLM(cfg)
    report = load_qwen_into_pater(model, qwen_state)
    if zero_init_side_state:
        report["zeroed_side_state"] = zero_init_warmstart_side_state(model)
    return model, report


def _print_report(report: dict) -> None:
    print(f"  loaded backbone params : {report['n_loaded_params']}")
    print(f"  random-init params     : {report['n_random_init_params']} (PAT-ER streams/heads)")
    pre, tot = report["pretrained_numel"], report["total_numel"]
    print(f"  pretrained numel       : {pre/1e6:.1f}M / {tot/1e6:.1f}M total ({100*pre/tot:.1f}%)")
    print(f"  embedding shared rows  : {report['shared_vocab_rows']}")
    if report["missing_src"]:
        print(f"  !! missing Qwen keys   : {len(report['missing_src'])} -> {report['missing_src'][:4]}")
    if report["shape_mismatch"]:
        print(f"  !! shape mismatches    : {report['shape_mismatch'][:4]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pat_er_qwen3_warmstart.yaml")
    ap.add_argument("--qwen-weights", type=str, default=None, help="path to Qwen3-0.6B model.safetensors")
    ap.add_argument("--synthetic", action="store_true", help="use a random shape-matched Qwen stand-in (offline)")
    ap.add_argument("--out", type=str, default=None, help="optional path to save the warm-started state_dict")
    args = ap.parse_args()

    if args.qwen_weights:
        print(f"[warmstart] loading real Qwen3-0.6B weights: {args.qwen_weights}")
        qwen_state = load_safetensors_state_dict(Path(args.qwen_weights))
    elif args.synthetic:
        print("[warmstart] SYNTHETIC stand-in (shape-matched random; offline smoke only)")
        qwen_state = synthetic_qwen_state_dict()
    else:
        raise SystemExit("provide --qwen-weights PATH or --synthetic")

    model, report = build_warmstart_model(args.config, qwen_state)
    _print_report(report)

    if args.out:
        torch.save(model.state_dict(), args.out)
        print(f"[warmstart] saved warm-started state_dict -> {args.out}")


if __name__ == "__main__":
    main()
