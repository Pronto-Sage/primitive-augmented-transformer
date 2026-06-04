#!/usr/bin/env python3
"""Build a small supervised *interface* corpus for the D-interface SFT gate.

Condition D (warm-start Stage 5B) is architecture-proven but was trained with
``--input-only``: the LM loss only ever saw ``<pat_er><text>...</text>`` with an
EMPTY ``<output>`` block. The structured prediction lives in the aux heads; the
generation side was never trained. That is why the product eval shows Hermes
tool-call parse rate = 0 and noisy free-form generation, while IDK and JSON
(carried by aux heads / simple structure) are already strong.

This script turns existing PAT-ER serialization into (prompt, response) pairs in
the SAME prompt format the product gate uses (scripts/eval_product.py:build_prompt),
so the gate becomes in-distribution. Gold responses are DERIVED deterministically
from existing record fields (formula atoms, evidence, labels.primitive_class /
support_status / idk_action / evidence_ids, and the gold <tool_call> already
present in model_text). No new labels are invented; this is a re-serialization.

The 31 held-out prompts in artifacts/datasets/product_eval_prompts.jsonl are the
TEST set and are never used here.

Output: artifacts/datasets/interface/{train,val}.jsonl with rows
    {"id", "category", "prompt", "response"}

Usage:
    python3 scripts/build_interface_sft.py --seed 0 \
        --datasets artifacts/datasets/mixed/pater_mixed_reason_big.jsonl \
                   artifacts/datasets/mixed/pater_mixed_deep.jsonl \
        --out-dir artifacts/datasets/interface
"""
from __future__ import annotations
import argparse, json, re, random, sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tool_specs as TS

ATOM_RE = re.compile(r'<atom>\s*(.*?)\s*</atom>', re.DOTALL)
TOOLCALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
PW_RE = re.compile(r'\s*facts\s+(.*?)\s+rules\s+(.*?)\s+question\s+(.*)\s*$',
                   re.DOTALL | re.IGNORECASE)


# ── helpers ─────────────────────────────────────────────────────────────────

def humanize(atom: str) -> str:
    """cache_warm -> 'cache warm'; leave natural-language atoms untouched."""
    a = atom.strip().rstrip('.')
    return a.replace('_', ' ') if '_' in a else a


def formula_atoms(formula: str | None) -> list[str]:
    return [humanize(a) for a in ATOM_RE.findall(formula or '')]


def gold_toolcall(model_text: str | None) -> dict | None:
    m = TOOLCALL_RE.search(model_text or '')
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get('name'), str) and isinstance(obj.get('arguments'), dict):
        return obj
    return None


def parse_proofwriter(text: str) -> tuple[str, str] | None:
    """Return (context, question_statement) for a ProofWriter-style record."""
    m = PW_RE.match(text)
    if not m:
        return None
    facts, rules, q = m.groups()
    facts = ' '.join(s.strip().rstrip('.') + '.' for s in facts.split(';') if s.strip())
    rules = ' '.join(s.strip().rstrip('.') + '.' for s in rules.split(';') if s.strip())
    context = (facts + ' ' + rules).strip()
    return context, q.strip().rstrip('.')


def reasoning_context_question(rec: dict) -> tuple[str, str] | None:
    """Build a natural (context, question) from a record. ProofWriter parses
    directly; synthetic records fall back to text + formula consequent."""
    pw = parse_proofwriter(rec.get('text', ''))
    if pw:
        return pw
    atoms = formula_atoms(rec.get('formula'))
    if len(atoms) >= 2:
        # synthetic rule record: context is the statement; ask about consequent
        return rec['text'].strip(), atoms[-1]
    return None


# ── prompt builders (mirror eval_product.build_prompt) ──────────────────────

def reasoning_prompt(context: str, question: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def tool_prompt(task: str, schema_name: str) -> str:
    return f"Task: {task}\nUse the {schema_name} function to complete this task.\nResponse:"


def json_prompt(task: str, keys: list[str]) -> str:
    return f"Task: {task}\nRespond as JSON with keys: {', '.join(keys)}\nResponse:"


# ── response builders (deterministic, short, coherent, control-token tail) ───

def r_proof(rng, atoms, goal):
    lead = rng.choice([
        f"Yes. {goal} follows from the premises by modus ponens.",
        f"Yes — {goal} is entailed; the antecedent holds, so the consequent follows.",
        f"Yes, {goal} holds: it is derivable from the stated rule and facts.",
    ])
    return f"{lead} <prim:modus_ponens> <support:proof>"


def r_evidence(rng, atoms, goal, ev_id):
    cite = f" (evidence {ev_id})" if ev_id else ""
    lead = rng.choice([
        f"Yes. {goal} is supported{cite}: the antecedent is reported, so the consequent follows.",
        f"Yes — {goal} follows by modus ponens{cite}.",
    ])
    return f"{lead} <prim:modus_ponens> <support:belief>"


def r_observation(rng, goal):
    return (rng.choice([
        f"Yes. {goal} is directly observed in the given information.",
        f"Yes — {goal} is stated as an observation.",
    ]) + " <prim:observation> <support:belief>")


def r_syllogism(rng, goal):
    return (rng.choice([
        f"Yes. {goal} follows by chaining the rules (syllogism).",
        f"Yes — {goal} is derivable through a multi-step rule chain.",
    ]) + " <prim:syllogism> <support:proof>")


def r_abduction(rng, atoms, goal):
    prem = atoms[0] if atoms else "a missing premise"
    lead = rng.choice([
        f"Most likely {prem}, which would explain {goal}. This is a hypothesis, not a proof.",
        f"A plausible explanation is {prem}; this is abduced, not proven.",
        f"The best explanation is that {prem} holds, but this remains a hypothesis to verify.",
    ])
    return f"{lead} <prim:abduction> <support:hypothesis> <needs_verification>"


def r_contradiction(rng, goal):
    lead = rng.choice([
        "There is a conflict: the premises cannot all hold together.",
        f"The claim conflicts with the premises; '{goal}' cannot be consistently concluded.",
        "This is a contradiction — the stated information is mutually inconsistent.",
    ])
    return f"{lead} <prim:contradiction> <support:conflict> <conflict>"


def r_idk_needs_evidence(rng, atoms):
    missing = atoms[0] if atoms else "a deciding premise"
    lead = rng.choice([
        f"Cannot be determined — {missing} has not been established.",
        f"Unknown from the given information; {missing} is missing.",
        "This cannot be answered: the information needed is not provided.",
    ])
    return f"{lead} <IDK> <needs_evidence> <support:unknown>"


def r_idk_clarify(rng):
    return (rng.choice([
        "I cannot answer without clarification of what is being asked.",
        "The question is underspecified; clarification is needed before answering.",
    ]) + " <IDK> <ask_clarification> <support:unknown>")


def r_toolcall(rng, call: dict) -> str:
    # A short natural-language preamble eases the model into the tool call: leading
    # with the rare <tool_call> special token straight after "Response:" is not
    # reachable with a frozen tied LM head, but emitting it after a few ordinary
    # tokens is. The vLLM/Hermes parser matches <tool_call>…</tool_call> anywhere,
    # so the preamble does not affect parsing.
    payload = json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
    pre = rng.choice([
        f"I will call the {call['name']} function.",
        f"Calling {call['name']} to complete this.",
        f"Use the {call['name']} tool:",
    ])
    return f"{pre}\n<tool_call>\n{payload}\n</tool_call>"


def _prim(rec): return rec['labels'].get('primitive_class', 'observation')
def _supp(rec): return rec['labels'].get('support_status', 'belief')

# A broad key pool with sensible flat string fillers. The point is to teach the model
# to copy WHATEVER keys the prompt requests (key-copying generalizes to unseen keys),
# not to memorize one schema. Unknown keys fall back to a short text snippet.
JSON_KEY_FILLERS = {
    'summary': lambda rec, txt: txt[:60].rstrip(),
    'primitive': lambda rec, txt: _prim(rec),
    'support_status': lambda rec, txt: _supp(rec),
    'support': lambda rec, txt: _supp(rec),
    'status': lambda rec, txt: _supp(rec),
    'answer': lambda rec, txt: {'proof': 'yes', 'belief': 'yes', 'conflict': 'no',
                                'unknown': 'unknown'}.get(_supp(rec), 'unknown'),
    'claim': lambda rec, txt: txt[:50].rstrip(),
    'confidence': lambda rec, txt: {'proof': 'high', 'belief': 'medium', 'hypothesis': 'low',
                                    'unknown': 'low', 'conflict': 'low'}.get(_supp(rec), 'medium'),
    'label': lambda rec, txt: _prim(rec),
    'category': lambda rec, txt: _prim(rec),
    'type': lambda rec, txt: _prim(rec),
    'sentiment': lambda rec, txt: 'neutral',
    'title': lambda rec, txt: txt[:40].rstrip(),
    'description': lambda rec, txt: txt[:40].rstrip(),
    'analysis': lambda rec, txt: txt[:40].rstrip(),
    'conclusion': lambda rec, txt: txt[:40].rstrip(),
    'entities': lambda rec, txt: txt[:30].rstrip(),
    'topic': lambda rec, txt: 'general',
    'length': lambda rec, txt: 'short',
    'count': lambda rec, txt: '1',
    'score': lambda rec, txt: '0.5',
    'result': lambda rec, txt: 'ok',
    'idk': lambda rec, txt: 'true' if rec['labels'].get('idk_action') != 'answer' else 'false',
}


def r_json(rec: dict, keys: list[str], txt: str) -> str:
    # Copy EXACTLY the requested keys, in order; fill unknown keys with a short snippet.
    obj = {k: JSON_KEY_FILLERS.get(k, lambda r, t: t[:30].rstrip())(rec, txt) for k in keys}
    return json.dumps(obj, ensure_ascii=False)


# ── main extraction ─────────────────────────────────────────────────────────

def build(records: list[dict], rng: random.Random, caps: dict[str, int]) -> list[dict]:
    out: list[dict] = []
    counts: Counter = Counter()

    def emit(cat, rec, prompt, response):
        out.append({"id": f"iface_{rec.get('id','x')}_{cat}", "category": cat,
                    "prompt": prompt, "response": response})
        counts[cat] += 1

    # shuffle once for sampling variety
    pool = records[:]
    rng.shuffle(pool)

    # 1) tool_call — schema-grounded examples generated from TRAIN_TOOLS (a name set
    # DISJOINT from the eval's EVAL_TOOLS). The schema is provided in-prompt, so the
    # model learns to copy the name + required keys from the schema and ground the
    # values in the task — generalizing to unseen tool names at eval. Covers single /
    # multi / optional / nested arg structures.
    i = 0
    while counts['tool_call'] < caps['tool_call']:
        sch = TS.TRAIN_TOOLS[i % len(TS.TRAIN_TOOLS)]
        i += 1
        r = TS.make_tool_record(sch, rng)
        out.append({'id': f"iface_tool_{i}", 'category': 'tool_call',
                    'prompt': TS.tool_prompt(r['task'], r['schema']),
                    'response': TS.tool_response(r['call'], rng)})
        counts['tool_call'] += 1

    # second pass over reasoning categories (so tool_call isn't starved)
    for rec in pool:
        L = rec.get('labels', {})
        prim = L.get('primitive_class')
        idk = L.get('idk_action')
        cq = reasoning_context_question(rec)
        if cq is None:
            continue
        context, question = cq
        ev = rec.get('evidence') or []
        ev_id = (rec.get('labels', {}).get('evidence_ids') or [None])[0]
        atoms = formula_atoms(rec.get('formula'))
        goal = question if question else (atoms[-1] if atoms else "the claim")

        if prim == 'abduction' and counts['abduction_vs_proof'] < caps['abduction_vs_proof']:
            emit('abduction_vs_proof', rec, reasoning_prompt(context, f"What is the most likely explanation that {goal}?"),
                 r_abduction(rng, atoms, goal))
        elif prim in ('contradiction', 'semantic_conflict') and counts['contradiction'] < caps['contradiction']:
            emit('contradiction', rec, reasoning_prompt(context, f"Is it true that {goal}?"),
                 r_contradiction(rng, goal))
        elif prim == 'modus_ponens' and ev and counts['evidence_grounded'] < caps['evidence_grounded']:
            emit('evidence_grounded', rec, reasoning_prompt(context, f"Is it true that {goal}?"),
                 r_evidence(rng, atoms, goal, ev_id))
        elif prim in ('modus_ponens', 'syllogism') and counts['proof'] < caps['proof']:
            resp = r_syllogism(rng, goal) if prim == 'syllogism' else r_proof(rng, atoms, goal)
            emit('proof', rec, reasoning_prompt(context, f"Is it true that {goal}?"), resp)
        elif prim == 'observation' and counts['evidence_grounded'] < caps['evidence_grounded']:
            emit('evidence_grounded', rec, reasoning_prompt(context, f"Is it true that {goal}?"),
                 r_observation(rng, goal))
        elif (prim in ('contingency', 'idk', 'uncertainty') or idk in ('needs_evidence', 'ask_clarification')) \
                and counts['idk'] < caps['idk']:
            resp = r_idk_clarify(rng) if idk == 'ask_clarification' else r_idk_needs_evidence(rng, atoms)
            emit('idk', rec, reasoning_prompt(context, f"Is it true that {goal}?"), resp)

    # third pass: json_output (cross-cutting). VARY the requested key set per example
    # so the model learns to copy whatever keys the prompt asks for (key-copying),
    # rather than memorizing one fixed schema.
    key_pool = list(JSON_KEY_FILLERS.keys())
    json_tasks = ['Classify the following', 'Analyze the following',
                  'Summarize the following', 'Describe the following']
    for rec in pool:
        if counts['json_output'] >= caps['json_output']:
            break
        text = (rec.get('text') or '').strip()
        if len(text) < 12:
            continue
        n_keys = rng.randint(2, 4)
        keys = rng.sample(key_pool, n_keys)
        if 'summary' not in keys and rng.random() < 0.5:
            keys[0] = 'summary'
        task_verb = rng.choice(json_tasks)
        emit('json_output', rec, json_prompt(f'{task_verb}: "{text[:120]}"', keys),
             r_json(rec, keys, text))

    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--out-dir', default='artifacts/datasets/interface')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--cap-tool', type=int, default=360)
    ap.add_argument('--cap-abduction', type=int, default=360)
    ap.add_argument('--cap-contradiction', type=int, default=300)
    ap.add_argument('--cap-evidence', type=int, default=260)
    ap.add_argument('--cap-proof', type=int, default=220)
    ap.add_argument('--cap-idk', type=int, default=240)
    ap.add_argument('--cap-json', type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records: list[dict] = []
    for ds in args.datasets:
        records.extend(json.loads(l) for l in Path(ds).read_text().splitlines() if l.strip())
    print(f"loaded {len(records)} source records from {len(args.datasets)} files")

    caps = {'tool_call': args.cap_tool, 'abduction_vs_proof': args.cap_abduction,
            'contradiction': args.cap_contradiction, 'evidence_grounded': args.cap_evidence,
            'proof': args.cap_proof, 'idk': args.cap_idk, 'json_output': args.cap_json}
    rows, counts = build(records, rng, caps)
    print("per-category counts:", dict(counts))
    print("total interface examples:", len(rows))

    # split train/val by hashing the row id (stable, category-stratified)
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    train, val = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train); rng.shuffle(val)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in train) + '\n')
    (out_dir / 'val.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in val) + '\n')
    print(f"train={len(train)} val={len(val)} -> {out_dir}")

    # show a couple of samples per category
    print("\n=== samples ===")
    shown = set()
    for r in train:
        if r['category'] in shown:
            continue
        shown.add(r['category'])
        print(f"\n[{r['category']}]")
        print("PROMPT:", r['prompt'][:200].replace('\n', ' / '))
        print("RESPONSE:", r['response'][:200].replace('\n', ' / '))


if __name__ == '__main__':
    main()
