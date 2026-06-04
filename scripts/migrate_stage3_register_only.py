"""Migrate a Stage-3 residual checkpoint → register_only shape.

Stage-3 training froze role_fuse / primitive_fuse at zero throughout, so
those weights carry no information.  This script copies all parameters from
the Stage-3 checkpoint into a new register_only model, replacing the fuse
weights with fresh zero tensors of the correct shape [h, h] (was [h, h*3]).

All other parameters (registers, cross-attention projections, aux heads, etc.)
are copied verbatim.  The resulting checkpoint is functionally identical to
the original at zero injection, but the fuse weights are compatible with the
register_only architecture so Stage 4A can open them cleanly.

Usage:
    python3 scripts/migrate_stage3_register_only.py \
        --in-ckpt  artifacts/checkpoints/qwen_ws_stage3_s0/latest.pt \
        --out-ckpt artifacts/checkpoints/qwen_ws_stage3_ro_s0/latest.pt \
        --config   configs/pat_er_qwen3_warmstart.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pat_er import PATERConfig, PATERForCausalLM


def migrate(in_ckpt_path: Path, out_ckpt_path: Path, config_path: str) -> None:
    ckpt = torch.load(str(in_ckpt_path), map_location="cpu", weights_only=True)
    old_state = ckpt["model_state"]

    # Build new register_only model using the ORIGINAL checkpoint config (so all
    # features — reasoning heads, adapters, etc. — are preserved), overriding
    # only injection_fusion_mode.
    ref_cfg = PATERConfig.from_yaml(config_path)
    assert ref_cfg.injection_fusion_mode == "register_only", (
        f"config {config_path} must have injection_fusion_mode: register_only"
    )
    cfg_dict = ckpt.get("config", {})
    cfg_dict["injection_fusion_mode"] = "register_only"
    config = PATERConfig.from_dict(cfg_dict)
    model = PATERForCausalLM(config)
    new_state = model.state_dict()

    fuse_keys = {n for n in new_state if n.endswith("role_fuse.weight") or n.endswith("primitive_fuse.weight")}
    copied = skipped = reshaped = 0
    for name, new_p in new_state.items():
        if name in fuse_keys:
            # The Stage-3 source weights were frozen at zero throughout;
            # the new-model init is kaiming_uniform (non-zero). Explicitly
            # zero to match the source semantics.
            new_state[name] = torch.zeros_like(new_p)
            reshaped += 1
            continue
        if name in old_state:
            old_p = old_state[name]
            if old_p.shape == new_p.shape:
                new_state[name] = old_p
                copied += 1
            else:
                print(f"  shape mismatch: {name}  {tuple(old_p.shape)} → {tuple(new_p.shape)} (keeping new)")
                skipped += 1
        else:
            print(f"  missing in source: {name} (keeping new init)")
            skipped += 1

    model.load_state_dict(new_state)

    print(f"migration: copied={copied}  reshaped(fuse→zero)={reshaped}  skipped={skipped}")
    print(f"  fuse param names: {sorted(fuse_keys)[:4]} ...")

    # Verify fuse weights are zero.
    for name in fuse_keys:
        w = dict(model.named_parameters())[name]
        assert w.abs().max().item() < 1e-9, f"{name} is not zero after migration"
    print("  fuse weights zero-verified ✓")

    # Save new checkpoint using the same metadata as the original.
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    new_ckpt = dict(ckpt)  # copy metadata
    new_ckpt["model_state"] = model.state_dict()
    new_ckpt["config"] = config.to_dict()
    torch.save(new_ckpt, str(out_ckpt_path))
    print(f"saved → {out_ckpt_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ckpt",  required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--config",   default="configs/pat_er_qwen3_warmstart.yaml")
    args = ap.parse_args()
    migrate(Path(args.in_ckpt), Path(args.out_ckpt), args.config)


if __name__ == "__main__":
    main()
