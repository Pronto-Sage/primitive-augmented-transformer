#!/usr/bin/env python3
"""D-interface SFT: make Condition D usable without touching the architecture.

Condition D (warm-start Stage 5B) proved the PAT-ER architecture but was trained
``--input-only`` (empty <output> block), so it never learned to GENERATE tool
calls, evidence-grounded answers, or abduction-as-hypothesis text. This script
fine-tunes the GENERATION side on a small supervised interface corpus (built by
scripts/build_interface_sft.py) while keeping the proven side-state intact.

Conservative train/freeze policy (architecture is FIXED):
  FROZEN  : token embeddings + tied LM head, lower backbone, registers,
            cross-attention, ALL aux heads  (the proven side-state)
  TRAINED : top-K Qwen backbone layers (optional, low LR, KL-guarded),
            FFN adapters in the top Ka layers (zero-init up-proj → safe),
            role/primitive/mix vocab-pressure heads (direct logit shaping —
            the architecturally-aligned lever for emitting control tokens)

Loss: response-masked causal LM (prompt tokens = -100). No aux losses are added;
the aux heads are frozen and already proven. A KL guard against the base-D model's
logits on fixed probe sentences stops training if general LM drifts too far.

The pass condition is measured afterward by scripts/eval_product.py (Hermes,
JSON, IDK, generation) and scripts/eval_lm_aux.py (primitive/r2p/LM must not
materially regress).

Usage:
    python3 scripts/train_interface_sft.py \
        --checkpoint artifacts/checkpoints/qwen_ws_stage5b_s0/latest.pt \
        --tokenizer  artifacts/tokenizers/qwen_pater_extended \
        --train artifacts/datasets/interface/train.jsonl \
        --val   artifacts/datasets/interface/val.jsonl \
        --device cuda --backbone-layers 2 --adapter-layers 6 \
        --steps 700 --out-dir artifacts/checkpoints/qwen_ws_d_iface_s0
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from pat_er import PATERConfig, PATERForCausalLM
from pat_er.modules.utils import maybe_autocast
from transformers import AutoTokenizer

# Freeze helpers + the full collate/loss machinery are reused verbatim from the
# main trainer (import is side-effect free). The replay stream runs PAT-ER-format
# records through the SAME collate + aux losses Stage 3 used, so the gradient
# actively holds the frozen aux heads' inputs in place (primitive/r2p preserved).
import train_lm_aux as T
import pater_hf_tokenizer as H
from train_lm_aux import open_top_backbone_layers, open_upper_layer_adapters


KL_PROBES = [
    "The capital of France is Paris, a city on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "If all men are mortal and Socrates is a man, then Socrates is mortal.",
    "The quarterly report summarizes revenue, costs, and net profit for the team.",
]


# ── checkpoint / model ──────────────────────────────────────────────────────

def load_condition_d(ckpt_path: Path, device: str):
    # weights_only=True is sufficient: this repo's checkpoints store config as a
    # plain dict plus tensors (verified), so no arbitrary unpickling is needed.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    raw = ckpt.get('config')
    if isinstance(raw, dict):
        config = PATERConfig(**{k: v for k, v in raw.items() if k in PATERConfig.__dataclass_fields__})
    elif raw is not None:
        config = raw
    else:
        config = PATERConfig()
    model = PATERForCausalLM(config)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.to(device)
    return model, config


def open_pressure_heads(model) -> list[str]:
    """Unfreeze the three vocab-pressure heads (proj.0 bottleneck + proj.1 to
    vocab). These add directly to the LM logits and are the cleanest lever for
    emitting control/structure tokens (<tool_call>, <support:*>) while the tied
    LM head stays frozen."""
    opened = []
    for name, p in model.named_parameters():
        if (name.startswith('role_pressure.') or name.startswith('primitive_pressure.')
                or name.startswith('mix_pressure.')):
            p.requires_grad_(True)
            opened.append(name)
    return opened


def apply_freeze(model, backbone_layers: int, adapter_layers: int, use_pressure: bool = True):
    for p in model.parameters():
        p.requires_grad_(False)
    bb = open_top_backbone_layers(model, backbone_layers)[1] if backbone_layers > 0 else []
    ad = open_upper_layer_adapters(model, adapter_layers)[1] if adapter_layers > 0 else []
    pr = open_pressure_heads(model) if use_pressure else []
    return set(bb), set(ad), set(pr)


# ── data ────────────────────────────────────────────────────────────────────

def tokenize_rows(rows, tok, max_len: int):
    """Each row -> (input_ids, labels) with the prompt masked to -100 and an eos
    appended to the response. add_special_tokens=False (this tokenizer adds no
    BOS); eos terminates generation."""
    eos = tok.eos_token_id
    out = []
    for r in rows:
        p_ids = tok(r['prompt'], add_special_tokens=False)['input_ids']
        r_ids = tok(' ' + r['response'], add_special_tokens=False)['input_ids'] + [eos]
        ids = (p_ids + r_ids)[:max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len]
        if len(labels) <= len(p_ids):  # response fully truncated away
            continue
        out.append((ids, labels))
    return out


def make_batches(data, batch_size, pad_id, rng, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [data[j] for j in idx[i:i + batch_size]]
        maxlen = max(len(ids) for ids, _ in chunk)
        input_ids, labels, attn = [], [], []
        for ids, lab in chunk:
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        yield (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


# ── KL guard vs base-D ──────────────────────────────────────────────────────

@torch.no_grad()
def probe_logits(model, tok, device):
    enc = tok(KL_PROBES, return_tensors='pt', padding=True)
    ids = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    was_training = model.training
    model.eval()
    out = model(input_ids=ids, attention_mask=attn, return_aux=False)
    if was_training:
        model.train()
    return out.logits.detach(), attn.bool()


@torch.no_grad()
def probe_lm_loss(model, tok, device):
    enc = tok(KL_PROBES, return_tensors='pt', padding=True)
    ids = enc['input_ids'].to(device); attn = enc['attention_mask'].to(device)
    lbl = ids.clone(); lbl[attn == 0] = -100
    was_training = model.training
    model.eval()
    loss = model(input_ids=ids, attention_mask=attn, labels=lbl, return_aux=False).loss.item()
    if was_training:
        model.train()
    return loss


def kl_vs_base(model, tok, device, base_logits, mask):
    cur, _ = probe_logits(model, tok, device)
    return F.kl_div(F.log_softmax(cur[mask], dim=-1),
                    F.softmax(base_logits[mask], dim=-1),
                    reduction='batchmean', log_target=False).item()


# ── train ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--train', default='artifacts/datasets/interface/train.jsonl')
    ap.add_argument('--val', default='artifacts/datasets/interface/val.jsonl')
    ap.add_argument('--replay', default=None,
                    help='PAT-ER reason dataset (jsonl) replayed as input-only full-LM to '
                         'anchor side-state. Strongly recommended.')
    ap.add_argument('--replay-frac', type=float, default=0.5,
                    help='Fraction of training items drawn from the replay (PAT-ER format) pool.')
    ap.add_argument('--role-anchor-boost', type=float, default=1.0,
                    help='Multiplier on the role-bridge aux loss weights during replay '
                         '(role_to_primitive / role_ambiguity / proto_role / arg_role). The '
                         'r2p bridge is the most layer-sensitive head; boost to hold it.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    ap.add_argument('--steps', type=int, default=700)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-seq-len', type=int, default=320)
    ap.add_argument('--lr', type=float, default=3e-5, help='LR for FFN adapters.')
    ap.add_argument('--pressure-lr', type=float, default=0.0,
                    help='LR for vocab-pressure heads (0 = same as --lr). Pressure adds a '
                         'per-sequence bias to ALL logits, so it pollutes general text fast; '
                         'keep it well below --lr or disable with --no-pressure.')
    ap.add_argument('--no-pressure', action='store_true',
                    help='Do not train the vocab-pressure heads (recommended: they globally '
                         'bias logits and blow the KL guard).')
    ap.add_argument('--backbone-lr', type=float, default=1e-6, help='LR for opened top backbone layers.')
    ap.add_argument('--backbone-layers', type=int, default=0)
    ap.add_argument('--adapter-layers', type=int, default=6)
    ap.add_argument('--warmup-steps', type=int, default=60)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-interval', type=int, default=50)
    ap.add_argument('--kl-guard-threshold', type=float, default=0.5)
    ap.add_argument('--kl-eval-interval', type=int, default=100)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='d_iface')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print(f"loading Condition D: {args.checkpoint}")
    model, config = load_condition_d(Path(args.checkpoint), device)

    bb_set, ad_set, pr_set = apply_freeze(model, args.backbone_layers, args.adapter_layers,
                                          use_pressure=not args.no_pressure)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: backbone={len(bb_set)} adapter={len(ad_set)} pressure={len(pr_set)} "
          f"params  ({n_train/1e6:.1f}M)")

    train_rows = [json.loads(l) for l in Path(args.train).read_text().splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in Path(args.val).read_text().splitlines() if l.strip()]
    train_data = tokenize_rows(train_rows, tok, args.max_seq_len)
    val_data = tokenize_rows(val_rows, tok, args.max_seq_len)
    print(f"interface: train={len(train_data)} val={len(val_data)} (max_seq_len={args.max_seq_len})")

    # Aux-anchored replay: PAT-ER-format records collated EXACTLY as Stage 3 (offset
    # tokenizer, full event-role + primitive targets). The replay loss is LM + the
    # side-state aux losses, so the gradient pulls the trainable top layers to keep
    # the FROZEN aux heads' predictions correct — directly preserving primitive/r2p.
    replay_records = []
    wrapped_tok = dims = None
    if args.replay and args.replay_frac > 0:
        replay_records = [json.loads(l) for l in Path(args.replay).read_text().splitlines() if l.strip()]
        rng.shuffle(replay_records)
        wrapped_tok = H.load_pater_hf_tokenizer(args.tokenizer)
        dims = T.dims_from_config(config)
        # Anchor the sequence-level side-state heads strongly; zero the span/evidence/
        # fact heads (offset-fragile, variance-dominated, not the gate metrics).
        for k in ('argument_start', 'argument_end', 'argument_span', 'event_token', 'event_arg',
                  'evidence_pointer', 'proof_depth', 'rule_chain_length', 'supporting_fact',
                  'contradicting_fact', 'supporting_item', 'contradicting_item', 'predicate_event'):
            T.LOSS_WEIGHTS[k] = 0.0
        if args.role_anchor_boost != 1.0:
            for k in ('role_to_primitive', 'role_ambiguity', 'proto_role', 'arg_role'):
                T.LOSS_WEIGHTS[k] = T.LOSS_WEIGHTS.get(k, 1.0) * args.role_anchor_boost
        print(f"replay: {len(replay_records)} PAT-ER records, frac={args.replay_frac}, "
              f"aux-anchored (role-bridge boost ×{args.role_anchor_boost})")

    # optimizer: separate groups so pressure (global logit bias) can run far below
    # the adapter LR — adapters are per-position and safe; pressure is not.
    pressure_lr = args.pressure_lr if args.pressure_lr > 0 else args.lr
    ad_params = [p for n, p in model.named_parameters() if p.requires_grad and n in ad_set]
    pr_params = [p for n, p in model.named_parameters() if p.requires_grad and n in pr_set]
    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n in bb_set]
    groups = []
    if ad_params:
        groups.append({'params': ad_params, 'lr': args.lr, 'lr_scale': 1.0})
    if pr_params:
        groups.append({'params': pr_params, 'lr': pressure_lr, 'lr_scale': pressure_lr / max(args.lr, 1e-12)})
    if bb_params:
        groups.append({'params': bb_params, 'lr': args.backbone_lr, 'lr_scale': args.backbone_lr / max(args.lr, 1e-12)})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    print(f"lr groups: adapter={args.lr:.1e}"
          + (f"  pressure={pressure_lr:.1e}" if pr_params else "")
          + (f"  backbone={args.backbone_lr:.1e}" if bb_params else ""))

    def lr_at(step):
        if step <= args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    # base-D probe reference for the KL guard
    base_logits, probe_mask = probe_logits(model, tok, device)
    base_probe_loss = probe_lm_loss(model, tok, device)
    print(f"base-D probe LM loss: {base_probe_loss:.4f}")

    def iface_stream():
        while True:
            for b in make_batches(train_data, args.batch_size, pad_id, rng):
                yield b
    stream = iface_stream()

    def replay_cursor():
        i = 0
        while True:
            if i + args.batch_size > len(replay_records):
                rng.shuffle(replay_records); i = 0
            yield replay_records[i:i + args.batch_size]
            i += args.batch_size
    rep_stream = replay_cursor() if replay_records else None

    model.train()
    first_loss = last_loss = None
    skipped = 0
    kl_stop = False
    n_replay_steps = 0
    for step in range(1, args.steps + 1):
        if kl_stop:
            break
        cur_lr = lr_at(step)
        for g in optimizer.param_groups:
            g['lr'] = cur_lr * g.get('lr_scale', 1.0)
        optimizer.zero_grad(set_to_none=True)
        is_replay = rep_stream is not None and rng.random() < args.replay_frac
        if is_replay:
            n_replay_steps += 1
            sub = next(rep_stream)
            batch = T.collate(wrapped_tok, sub, args.max_seq_len, dims, 'input_only').to(device)
            with maybe_autocast(device, args.dtype):
                out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                            labels=batch.input_ids, return_aux=True,
                            evidence_item_mask=batch.er.get('evidence_item_mask'))
                loss = T.total_loss(T.compute_losses(out, batch))
        else:
            input_ids, labels, attn = next(stream)
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            with maybe_autocast(device, args.dtype):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels, return_aux=False)
                loss = out.loss
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True); skipped += 1
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
        if not torch.isfinite(gn):
            optimizer.zero_grad(set_to_none=True); skipped += 1
            continue
        optimizer.step()
        last_loss = float(loss.detach())
        if first_loss is None:
            first_loss = last_loss

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            msg = f"  step {step:>4}: lr={cur_lr:.2e} loss={last_loss:.4f} ({'replay' if is_replay else 'iface'})"
            print(msg, flush=True)
            if step % args.kl_eval_interval == 0:
                kl = kl_vs_base(model, tok, device, base_logits, probe_mask)
                dl = probe_lm_loss(model, tok, device) - base_probe_loss
                print(f"  [kl-guard] step={step} KL(cur||baseD)={kl:.4f} probe_lm_delta={dl:+.4f}", flush=True)
                if kl > args.kl_guard_threshold:
                    print(f"  KL guard triggered — stopping at step {step}")
                    kl_stop = True

    if skipped:
        print(f"skipped {skipped}/{args.steps} steps (non-finite)")

    # final val loss
    model.eval()
    with torch.no_grad():
        tot, ntok = 0.0, 0
        for input_ids, labels, attn in make_batches(val_data, args.batch_size, pad_id, rng, shuffle=False):
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            with maybe_autocast(device, args.dtype):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels, return_aux=False)
            n = int((labels != -100).sum().item())
            tot += float(out.loss) * n; ntok += n
        val_loss = tot / max(ntok, 1)
    final_kl = kl_vs_base(model, tok, device, base_logits, probe_mask)
    print(f"DONE: first={first_loss:.4f} last={last_loss:.4f} val_lm={val_loss:.4f} "
          f"final_KL(cur||baseD)={final_kl:.4f} replay_steps={n_replay_steps}/{args.steps}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'config': config.to_dict(),
        'tokenizer_path': args.tokenizer,
        'model_state': model.state_dict(),
        'sft': {'first_lm': first_loss, 'last_lm': last_loss, 'val_lm': val_loss,
                'final_kl': final_kl, 'steps': args.steps, 'skipped': skipped,
                'backbone_layers': args.backbone_layers, 'adapter_layers': args.adapter_layers,
                'lr': args.lr, 'backbone_lr': args.backbone_lr},
        'tag': args.tag,
    }
    torch.save(checkpoint, out_dir / 'latest.pt')
    print(f"saved → {out_dir / 'latest.pt'}")


if __name__ == '__main__':
    main()
