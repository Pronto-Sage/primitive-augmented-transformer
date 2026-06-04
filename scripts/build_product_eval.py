#!/usr/bin/env python3
"""Build the big held-out product-eval set for the interface robustness gate.

~240 prompts:
  - ~100 tool prompts from EVAL_TOOLS (UNSEEN names, schema-in-prompt), across the
    four arg subtypes, each carrying a gold spec (expected_name, required, optional,
    arg_values) for exact name / args / value scoring.
  - expanded reasoning (evidence_grounded / abduction / contradiction), idk, and
    json prompts in clean natural language, plus the original 31 curated prompts.

None of these tools/prompts appear in training (build_interface_sft.py uses TRAIN_TOOLS,
a disjoint name set, and synthetic PAT-ER records). Output:
    artifacts/datasets/product_eval_big.jsonl

Usage: python3 scripts/build_product_eval.py --seed 0 --per-tool 10
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tool_specs as TS

# ── reasoning / idk / json templated pools (clean natural language) ────────────

MAMMAL = [('mammals', 'breathe air', 'dolphins'), ('birds', 'have feathers', 'robins'),
          ('metals', 'conduct electricity', 'copper'), ('prime numbers', 'are odd or two', 'seven'),
          ('reptiles', 'are cold-blooded', 'lizards'), ('squares', 'have four sides', 'this shape'),
          ('citrus fruits', 'contain vitamin C', 'lemons'), ('triangles', 'have three angles', 'this figure')]

CONTRA = [('The memo says the meeting is on Monday.', 'The calendar shows it is on Wednesday.', 'the meeting time'),
          ('The invoice lists the total as $500.', 'The receipt shows the total as $700.', 'the total'),
          ('Report A says revenue rose.', 'Report B says revenue fell.', 'the revenue trend'),
          ('The label says gluten-free.', 'The ingredients list wheat flour.', 'the gluten content'),
          ('The sensor reads 20 degrees.', 'The thermostat reads 30 degrees.', 'the temperature')]

IDK_Q = [('The factory produces 200 units per day.', "What is the factory manager's name?"),
         ('The library opens at 9 am.', 'How many books were borrowed last year?'),
         ('The train departs at noon.', 'What is the conductor wearing?'),
         ('The recipe needs two eggs.', 'Who invented this recipe?'),
         ('The store sells umbrellas.', 'What is the password to the safe?'),
         ('The report is ten pages long.', 'What color is the author\'s car?')]

ABDUCT = [('The ground is wet.', 'it rained last night'), ('The lights are off.', 'the power went out'),
          ('The plants are wilting.', 'they were not watered'), ('The phone is silent.', 'the battery died'),
          ('The room is cold.', 'the window was left open'), ('The cake did not rise.', 'the yeast was old')]

JSON_TASKS = ['Summarize the following: "The meeting was productive."',
              'Classify the sentiment of: "I love this product."',
              'Extract the entities from: "Alice met Bob in Paris."',
              'Analyze the statement: "Revenue grew by ten percent."',
              'Describe the document: "Quarterly financial report."']
JSON_KEYSETS = [['summary', 'sentiment'], ['summary', 'primitive', 'support_status'],
                ['entities', 'count'], ['analysis', 'confidence'], ['title', 'category', 'length']]


def build(seed: int, per_tool: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    # tools (eval names, unseen) — per_tool examples each, schema in prompt
    for sch in TS.EVAL_TOOLS:
        for _ in range(per_tool):
            r = TS.make_tool_record(sch, rng)
            rows.append({
                'id': f"tool_{sch['name']}_{len(rows)}", 'category': 'tool_call',
                'subtype': r['subtype'], 'prompt_kind': 'tool_schema',
                'task': r['task'], 'schema': r['schema'],
                'expected_name': r['name'], 'required_args': r['required'],
                'optional_args': r['optional'], 'arg_values': r['arg_values'],
                'expected_primitive': 'tool',
            })

    # evidence_grounded (modus ponens, proof)
    for i, (cls, prop, subj) in enumerate(MAMMAL * 4):
        if i >= 28:
            break
        rows.append({'id': f'evid_{i}', 'category': 'evidence_grounded',
                     'context': f'All {cls} {prop}. {subj.capitalize()} are {cls}.',
                     'question': f'Do {subj} {prop}?',
                     'expected_primitive': 'modus_ponens', 'expected_support': 'proof'})

    # contradiction
    for i, (a, b, what) in enumerate(CONTRA * 5):
        if i >= 25:
            break
        rows.append({'id': f'contra_{i}', 'category': 'contradiction',
                     'context': f'{a} {b}', 'question': f'Is there a conflict about {what}?',
                     'expected_primitive': 'contradiction', 'expected_support': 'conflict'})

    # idk
    for i, (ctx, q) in enumerate(IDK_Q * 5):
        if i >= 28:
            break
        rows.append({'id': f'idk_{i}', 'category': 'idk', 'context': ctx, 'question': q,
                     'expected_primitive': 'idk', 'expected_support': 'unknown'})

    # abduction
    for i, (obs, expl) in enumerate(ABDUCT * 5):
        if i >= 24:
            break
        rows.append({'id': f'abduct_{i}', 'category': 'abduction_vs_proof',
                     'context': obs, 'question': 'What is the most likely explanation?',
                     'expected_primitive': 'abduction', 'answer_hint': expl})

    # json_output (varied keys)
    for i in range(25):
        task = JSON_TASKS[i % len(JSON_TASKS)]
        keys = JSON_KEYSETS[i % len(JSON_KEYSETS)]
        rows.append({'id': f'json_{i}', 'category': 'json_output', 'task': task,
                     'required_keys': keys, 'expected_primitive': 'observation'})

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per-tool', type=int, default=10)
    ap.add_argument('--out', default='artifacts/datasets/product_eval_big.jsonl')
    args = ap.parse_args()

    rows = build(args.seed, args.per_tool)
    from collections import Counter
    cats = Counter(r['category'] for r in rows)
    subs = Counter(r.get('subtype') for r in rows if r['category'] == 'tool_call')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    print(f"wrote {len(rows)} prompts -> {args.out}")
    print("categories:", dict(cats))
    print("tool subtypes:", dict(subs))


if __name__ == '__main__':
    main()
