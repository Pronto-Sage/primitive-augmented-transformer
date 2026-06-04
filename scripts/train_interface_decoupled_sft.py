#!/usr/bin/env python3
"""Decoupled interface SFT via mode-gated top-K layers (option 3).

The shared-layer SFT taught D to generate but cost −0.094 r2p (the top layers carry
the role→primitive bridge). The logit-only output head preserved the side-state
exactly but could not supply in-context tool-name copying (that needs attention).

This trainer gets both: it trains COPIES of the top-K decoder blocks that are used
ONLY in `interface_mode`. Base mode always runs the original frozen blocks, so the
aux heads / side-state read exactly the Condition-D representation — r2p is preserved
by construction (directly asserted). Interface mode runs the trained copies, whose
adapted attention can copy novel tool names, like the shared-layer SFT did.

    base-mode forward (matrix eval, aux heads) == base D, exactly.
    interface-mode forward (product generation) == adapted top-K layers.

Frozen: everything except the top-K interface copies. No replay / KL guard needed.

Usage:
    python3 scripts/train_interface_decoupled_sft.py \
        --checkpoint artifacts/checkpoints/qwen_ws_stage5b_s0/latest.pt \
        --tokenizer  artifacts/tokenizers/qwen_pater_extended \
        --adapt-layers 2 --steps 600 --lr 3e-5 --device cuda \
        --out-dir artifacts/checkpoints/qwen_ws_d_decoupled_s0
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import maybe_autocast
from transformers import AutoTokenizer
from train_interface_sft import tokenize_rows, make_batches, KL_PROBES


def load_base_with_adapt(ckpt_path: Path, adapt_layers: int, device: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    raw = ckpt.get('config')
    cfg = {k: v for k, v in raw.items() if k in PATERConfig.__dataclass_fields__} if isinstance(raw, dict) else {}
    cfg['interface_adapt_layers'] = adapt_layers
    config = PATERConfig(**cfg)
    model = PATERForCausalLM(config)
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    other_missing = [m for m in missing if not m.startswith('interface_layers')]
    assert not other_missing, f"unexpected missing (non-interface) params: {other_missing[:5]}"
    assert not unexpected, f"unexpected params in checkpoint: {unexpected[:5]}"
    # Initialize the interface copies FROM the base-D top-K blocks (start == D).
    n, k = config.num_hidden_layers, adapt_layers
    for j in range(k):
        model.interface_layers[j].load_state_dict(model.layers[n - k + j].state_dict())
    model.to(device)
    max_len = int(ckpt.get('max_len', min(config.max_position_embeddings, 768)))
    return model, config, max_len, ckpt


@torch.no_grad()
def base_logits(model, tok, device):
    enc = tok(KL_PROBES, return_tensors='pt', padding=True)
    ids, attn = enc['input_ids'].to(device), enc['attention_mask'].to(device)
    was = model.training; model.eval()
    out = model(input_ids=ids, attention_mask=attn, return_aux=False, interface_mode=False).logits.clone()
    if was: model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--train', default='artifacts/datasets/interface/train.jsonl')
    ap.add_argument('--val', default='artifacts/datasets/interface/val.jsonl')
    ap.add_argument('--adapt-layers', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-seq-len', type=int, default=320)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--warmup-steps', type=int, default=60)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-interval', type=int, default=150)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='d_decoupled')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model, config, base_max_len, base_ckpt = load_base_with_adapt(Path(args.checkpoint), args.adapt_layers, device)

    # Freeze EVERYTHING except the interface copies.
    for nme, p in model.named_parameters():
        p.requires_grad_(nme.startswith('interface_layers'))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: top-{args.adapt_layers} interface layers only ({n_train/1e6:.1f}M params)")

    train_rows = [json.loads(l) for l in Path(args.train).read_text().splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in Path(args.val).read_text().splitlines() if l.strip()]
    train_data = tokenize_rows(train_rows, tok, args.max_seq_len)
    val_data = tokenize_rows(val_rows, tok, args.max_seq_len)
    print(f"interface: train={len(train_data)} val={len(val_data)}")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step):
        if step <= args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    def stream():
        while True:
            for b in make_batches(train_data, args.batch_size, pad_id, rng):
                yield b
    it = stream()

    base_ref = base_logits(model, tok, device)
    model.train()
    first = last = None
    skipped = 0
    for step in range(1, args.steps + 1):
        cur_lr = lr_at(step)
        for g in optimizer.param_groups:
            g['lr'] = cur_lr
        input_ids, labels, attn = next(it)
        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, args.dtype):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                        return_aux=False, interface_mode=True)
            loss = out.loss
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True); skipped += 1; continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
        if not torch.isfinite(gn):
            optimizer.zero_grad(set_to_none=True); skipped += 1; continue
        optimizer.step()
        last = float(loss.detach())
        if first is None:
            first = last
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(f"  step {step:>4}: lr={cur_lr:.2e} loss={last:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        tot, ntok = 0.0, 0
        for input_ids, labels, attn in make_batches(val_data, args.batch_size, pad_id, rng, shuffle=False):
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            with maybe_autocast(device, args.dtype):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels,
                            return_aux=False, interface_mode=True)
            n = int((labels != -100).sum().item())
            tot += float(out.loss) * n; ntok += n
        val_loss = tot / max(ntok, 1)
    base_drift = float((base_logits(model, tok, device) - base_ref).abs().max())
    print(f"DONE: first={first:.4f} last={last:.4f} val_lm={val_loss:.4f} "
          f"base_drift={base_drift:.2e} skipped={skipped}")
    assert base_drift == 0.0, f"base mode drifted (={base_drift}) — decoupling broken!"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'config': config.to_dict(),
        'vocab': base_ckpt.get('vocab'),
        'tokenizer_path': args.tokenizer,
        'max_len': base_max_len,           # carry so eval_lm_aux truncates like base D
        'render_mode': base_ckpt.get('render_mode', 'input_only'),
        'model_state': model.state_dict(),
        'sft': {'first': first, 'last': last, 'val_lm': val_loss, 'adapt_layers': args.adapt_layers,
                'steps': args.steps, 'lr': args.lr, 'skipped': skipped},
        'tag': args.tag,
    }
    torch.save(checkpoint, out_dir / 'latest.pt')
    print(f"saved → {out_dir / 'latest.pt'}  (max_len={base_max_len})")


if __name__ == '__main__':
    main()
