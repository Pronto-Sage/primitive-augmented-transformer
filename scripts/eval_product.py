#!/usr/bin/env python3
"""Product eval for PAT-ER Condition D (warm-start usable model).

Tests the model on held-out prompts across 6 categories:
  evidence_grounded  - should predict modus_ponens + proof support
  idk                - should predict idk + unknown support
  abduction_vs_proof - should predict correct primitive
  contradiction      - should predict contradiction + conflict support
  tool_call          - should emit <tool_call> Hermes format
  json_output        - should produce parseable JSON with required keys

Reports:
  primitive accuracy (predicted vs expected_primitive)
  support accuracy   (predicted vs expected_support)
  IDK precision and recall
  Hermes parse rate  (tool_call category)
  JSON validity rate (json_output category)
  generation sanity  (coherent continuation)

Usage:
    python3 scripts/eval_product.py \\
        --checkpoint artifacts/checkpoints/qwen_ws_stage5b_s0/latest.pt \\
        --tokenizer  artifacts/tokenizers/qwen_pater_extended \\
        --prompts    artifacts/datasets/product_eval_prompts.jsonl \\
        --device     cuda --json-out artifacts/reports/product_eval_s0.json
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from collections import defaultdict, Counter

import torch
from transformers import AutoTokenizer

# Reuse model loader from train script
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from pat_er import PATERForCausalLM, PATERConfig
import tool_specs as TS


def load_ckpt(ckpt_path: Path, device: str, tok_dir: str):
    # weights_only=False required: checkpoints contain PATERConfig objects and other
    # non-tensor Python data. Only load checkpoints produced by this repo's training
    # scripts; never load from untrusted sources.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # nosec
    raw_config = ckpt.get('config')
    if isinstance(raw_config, dict):
        config = PATERConfig(**{k: v for k, v in raw_config.items()
                                if k in PATERConfig.__dataclass_fields__})
    elif raw_config is not None:
        config = raw_config
    else:
        config = PATERConfig()
    model = PATERForCausalLM(config)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval().to(device)
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)
    return model, tok, config


@torch.no_grad()
def run_prompt(model, tok, prompt_text: str, device: str,
               max_new: int = 80, interface_mode: bool = False) -> tuple[str, dict]:
    """Run a text prompt and return (generated_text, aux_outputs). When the checkpoint
    carries a decoupled interface head, interface_mode=True applies its logit-only
    delta (generation only; aux heads are read in base mode below, unaffected)."""
    enc = tok(prompt_text, return_tensors='pt').to(device)
    n_prompt = enc['input_ids'].shape[1]
    ids = enc['input_ids']

    # Greedy generation
    for _ in range(max_new):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), return_aux=True,
                    interface_mode=interface_mode)
        next_tok = out.logits[:, -1:].argmax(-1)
        ids = torch.cat([ids, next_tok], dim=1)
        if next_tok.item() in (tok.eos_token_id, tok.pad_token_id):
            break

    # Keep special tokens: <tool_call>/</tool_call> and the <support:*>/<IDK>
    # control tokens are registered special tokens; skip_special_tokens=True would
    # strip them BEFORE the Hermes/structure checks run (making Hermes parse rate
    # un-measurable). Strip only the eos/pad markers for cleanliness.
    gen_text = tok.decode(ids[0, n_prompt:], skip_special_tokens=False)
    for _m in (tok.eos_token, tok.pad_token):
        if _m:
            gen_text = gen_text.replace(_m, '')
    gen_text = gen_text.strip()

    # Extract aux head predictions from the last output position
    aux = {}
    if hasattr(out, 'primitive_logits') and out.primitive_logits is not None:
        prims = ['observation', 'abduction', 'modus_ponens', 'syllogism',
                 'contradiction', 'contingency', 'tautology', 'provenance',
                 'schema', 'tool', 'idk', 'uncertainty', 'semantic_conflict', 'axiom']
        pred_idx = out.primitive_logits[:, -1, :].argmax(-1).item()
        aux['primitive_pred'] = prims[pred_idx] if pred_idx < len(prims) else f'idx_{pred_idx}'

    if hasattr(out, 'support_logits') and out.support_logits is not None:
        supports = ['proof', 'belief', 'hypothesis', 'unknown', 'conflict', 'no_progress']
        pred_idx = out.support_logits[:, -1, :].argmax(-1).item()
        aux['support_pred'] = supports[pred_idx] if pred_idx < len(supports) else f'idx_{pred_idx}'

    if hasattr(out, 'idk_logits') and out.idk_logits is not None:
        idk_actions = ['answer', 'needs_evidence', 'needs_verification', 'ask_clarification']
        pred_idx = out.idk_logits[:, -1, :].argmax(-1).item()
        aux['idk_action_pred'] = idk_actions[pred_idx] if pred_idx < len(idk_actions) else f'idx_{pred_idx}'

    if hasattr(out, 'tool_intent_logits') and out.tool_intent_logits is not None:
        # 5 classes: 0=none, 1=call, 2=result, 3=error, 4=schema
        pred_idx = out.tool_intent_logits[:, -1, :].argmax(-1).item()
        aux['tool_intent_pred'] = ['none', 'call', 'result', 'error', 'schema'][pred_idx] if pred_idx < 5 else f'idx_{pred_idx}'

    return gen_text, aux


def check_hermes(text: str) -> bool:
    """Check if text contains a parseable Hermes tool_call."""
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(1))
        return 'name' in obj and 'arguments' in obj
    except Exception:
        return False


def score_tool_call(text: str, pd: dict) -> dict:
    """Score a generated tool call against the gold spec: parse, exact name, exact
    args (all required present, no keys outside required∪optional), and value match."""
    res = {'hermes': 0, 'name_acc': None, 'args_exact': None, 'arg_value_acc': None}
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not m:
        return res
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return res
    if not (isinstance(obj, dict) and 'name' in obj and isinstance(obj.get('arguments'), dict)):
        return res
    res['hermes'] = 1
    name, args = obj['name'], obj['arguments']
    if pd.get('expected_name') is not None:
        res['name_acc'] = int(name == pd['expected_name'])
    req, opt = pd.get('required_args') or [], pd.get('optional_args') or []
    emitted, allowed = set(args.keys()), set(req) | set(opt)
    # exact: every required key present AND no hallucinated keys outside required∪optional
    res['args_exact'] = int(all(k in emitted for k in req) and emitted <= allowed)
    gold_vals = pd.get('arg_values') or {}
    if gold_vals:
        flat: dict[str, str] = {}
        for k, v in args.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[f"{k}.{sk}"] = str(sv)
            else:
                flat[k] = str(v)
        hits = sum(1 for gk, gv in gold_vals.items() if flat.get(gk) == str(gv))
        res['arg_value_acc'] = hits / len(gold_vals)
    return res


def check_json(text: str, required_keys: list[str]) -> tuple[bool, bool]:
    """Returns (is_valid_json, has_required_keys)."""
    # Find first JSON object in text
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        return False, False
    try:
        obj = json.loads(m.group())
        has_keys = all(k in obj for k in required_keys)
        return True, has_keys
    except Exception:
        return False, False


def check_generation_sane(text: str) -> bool:
    """Minimal generation sanity: non-empty, not pure repetition, >= 3 words."""
    words = text.strip().split()
    if len(words) < 3:
        return False
    # Check for high repetition
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio > 0.3


def build_prompt(prompt_data: dict) -> str:
    """Convert a product-eval record to a text prompt for the model."""
    cat = prompt_data.get('category', '')
    if cat in ('evidence_grounded', 'idk', 'contradiction'):
        return (f"Context: {prompt_data['context']}\n"
                f"Question: {prompt_data['question']}\nAnswer:")
    elif cat == 'abduction_vs_proof':
        return (f"Context: {prompt_data['context']}\n"
                f"Question: {prompt_data['question']}\nAnswer:")
    elif cat == 'tool_call':
        if 'schema' in prompt_data:  # schema-in-prompt (robustness gate)
            return TS.tool_prompt(prompt_data['task'], prompt_data['schema'])
        return (f"Task: {prompt_data['task']}\n"
                f"Use the {prompt_data['schema_name']} function to complete this task.\n"
                f"Response:")
    elif cat == 'json_output':
        keys = ', '.join(prompt_data['required_keys'])
        return (f"Task: {prompt_data['task']}\n"
                f"Respond as JSON with keys: {keys}\nResponse:")
    return str(prompt_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--prompts', default='artifacts/datasets/product_eval_prompts.jsonl')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--max-new', type=int, default=80)
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} on {args.device}")
    model, tok, config = load_ckpt(Path(args.checkpoint), args.device, args.tokenizer)

    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    interface_mode = (getattr(config, 'interface_head_rank', 0) > 0
                      or getattr(config, 'interface_adapt_layers', 0) > 0)
    print(f"Evaluating {len(prompts)} prompts (interface_mode={interface_mode})")

    results = []
    cat_metrics: dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for p in prompts:
        prompt_text = build_prompt(p)
        gen, aux = run_prompt(model, tok, prompt_text, args.device, args.max_new,
                              interface_mode=interface_mode)

        row = {
            'id': p['id'],
            'category': p['category'],
            'prompt': prompt_text[:200],
            'generation': gen,
            'aux': aux,
        }

        # Category-specific checks
        cat = p['category']

        # Primitive accuracy
        exp_prim = p.get('expected_primitive')
        pred_prim = aux.get('primitive_pred')
        prim_correct = (pred_prim == exp_prim) if (exp_prim and pred_prim) else None
        row['primitive_correct'] = prim_correct
        if prim_correct is not None:
            cat_metrics[cat]['primitive_correct'].append(int(prim_correct))

        # Support accuracy
        exp_supp = p.get('expected_support')
        pred_supp = aux.get('support_pred')
        supp_correct = (pred_supp == exp_supp) if (exp_supp and pred_supp) else None
        row['support_correct'] = supp_correct
        if supp_correct is not None:
            cat_metrics[cat]['support_correct'].append(int(supp_correct))

        # IDK precision/recall
        if cat == 'idk':
            is_idk_pred = (pred_prim == 'idk' or aux.get('idk_action_pred') != 'answer')
            cat_metrics['idk']['predicted_idk'].append(int(is_idk_pred))
            row['predicted_idk'] = is_idk_pred

        # Tool call: parse + exact name / args / values, with per-subtype breakdown
        if cat == 'tool_call':
            sc = score_tool_call(gen, p)
            row.update(sc)
            tgt = [cat_metrics['tool_call']]
            st = p.get('subtype')
            if st:
                tgt.append(cat_metrics[f'tool:{st}'])
            for cm in tgt:
                cm['hermes_parse'].append(sc['hermes'])
                if sc['name_acc'] is not None:
                    cm['name_acc'].append(sc['name_acc'])
                if sc['args_exact'] is not None:
                    cm['args_exact'].append(sc['args_exact'])
                if sc['arg_value_acc'] is not None:
                    cm['arg_value_acc'].append(sc['arg_value_acc'])

        # JSON validity
        if cat == 'json_output':
            valid, has_keys = check_json(gen, p.get('required_keys', []))
            cat_metrics['json_output']['json_valid'].append(int(valid))
            cat_metrics['json_output']['json_has_keys'].append(int(has_keys))
            row['json_valid'] = valid
            row['json_has_keys'] = has_keys

        # Generation sanity
        sane = check_generation_sane(gen)
        row['gen_sane'] = sane
        cat_metrics[cat]['gen_sane'].append(int(sane))

        print(f"  [{p['id']:30s}] prim={pred_prim or '—':18s} supp={pred_supp or '—':10s} "
              f"gen_sane={sane} | {gen[:60].replace(chr(10),' ')}")
        results.append(row)

    # Summary
    print('\n=== PRODUCT EVAL SUMMARY ===')
    all_metrics: dict[str, list] = defaultdict(list)
    for cat, mets in cat_metrics.items():
        print(f'\n{cat}:')
        for met, vals in mets.items():
            acc = sum(vals) / len(vals)
            all_metrics[met].extend(vals)
            print(f'  {met:25s}: {acc:.3f}  ({sum(vals)}/{len(vals)})')

    print('\nOverall:')
    for met, vals in sorted(all_metrics.items()):
        if vals:
            print(f'  {met:25s}: {sum(vals)/len(vals):.3f}  (n={len(vals)})')

    # IDK precision/recall
    idk_true = [p['category'] == 'idk' for p in prompts]
    idk_pred = [r.get('predicted_idk', False) for r in results]
    tp = sum(t and p for t, p in zip(idk_true, idk_pred))
    fp = sum((not t) and p for t, p in zip(idk_true, idk_pred))
    fn = sum(t and (not p) for t, p in zip(idk_true, idk_pred))
    idk_prec = tp / max(tp + fp, 1)
    idk_rec  = tp / max(tp + fn, 1)
    print(f'\nIDK precision: {idk_prec:.3f}  recall: {idk_rec:.3f}  '
          f'F1: {2*idk_prec*idk_rec/max(idk_prec+idk_rec, 1e-8):.3f}')

    if args.json_out:
        out = {
            'checkpoint': str(args.checkpoint),
            'n_prompts': len(prompts),
            'results': results,
            'category_metrics': {k: {m: (sum(v)/len(v) if v else None)
                                     for m, v in ms.items()}
                                  for k, ms in cat_metrics.items()},
            'idk': {'precision': idk_prec, 'recall': idk_rec,
                    'f1': 2*idk_prec*idk_rec/max(idk_prec+idk_rec, 1e-8)},
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f'JSON → {args.json_out}')


if __name__ == '__main__':
    main()
