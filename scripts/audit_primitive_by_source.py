#!/usr/bin/env python3
"""Diagnostic: per-source primitive-class recall for one primitive, on a mix's val
split, using a trained checkpoint. Tells whether a bridge confusion is a synthetic
label issue or an external (real-data) reasoning-hardness issue. Offline, GPU/CPU."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import eval_lm_aux as E, train_lm_aux as T

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=ROOT/"artifacts/tokenizers/qwen_pater_extended")
    p.add_argument("--primitive", default="contradiction")
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    model, tok, cfg, max_len, _ = E.load_checkpoint(args.checkpoint, args.device, args.tokenizer)
    model.eval()
    recs = [json.loads(l) for l in open(args.dataset) if json.loads(l)["split"] == args.split]
    tgt = [r for r in recs if r["labels"]["primitive_class"] == args.primitive]
    dims = T.dims_from_config(cfg); PC = T.builder.PRIMITIVE_CLASSES
    by = defaultdict(lambda: [0, 0]); wrong = Counter()
    for i in range(0, len(tgt), 8):
        sub = tgt[i:i+8]
        b = T.collate(tok, sub, max_len, dims, "input_only").to(args.device)
        with torch.no_grad():
            out = model(input_ids=b.input_ids, attention_mask=b.attention_mask, return_aux=True)
        pred = out.aux_outputs["primitive_class_logits"].argmax(-1).tolist()
        for r, pr in zip(sub, pred):
            src = (r.get("provenance") or {}).get("original_source") or r.get("source")
            src = "synthetic" if src == "synthetic_primitive" else src
            ok = PC[pr] == args.primitive
            by[src][0] += ok; by[src][1] += 1
            if not ok: wrong[f"{src}->{PC[pr]}"] += 1
    out = {"primitive": args.primitive, "by_source": {s: {"recall": c/n, "n": n} for s, (c, n) in by.items()},
           "misclassified": dict(wrong)}
    print(json.dumps(out))
if __name__ == "__main__":
    main()
