#!/usr/bin/env python3
"""Compute mean/std/95% bootstrap CI for the B/G/C/D contribution matrix.

Usage:
    python3 scripts/compute_matrix_stats.py [--seeds 0-7] [--output artifacts/reports/matrix_stats.json]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import random

# ── parsing ───────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict:
    if not path.exists():
        return {}
    t = path.read_text()
    prim_sec = re.search(r'primitive\s+macroF1.*?role_to_primitive', t, re.DOTALL)
    r2p_sec  = re.search(r'role_to_primitive\s+macroF1.*?(?=pointer|$)', t, re.DOTALL)

    def contra_from(sec: str | None) -> float | None:
        m = re.search(r'contradiction\s+(\d+)/(\d+)', sec or '')
        return int(m.group(1)) / int(m.group(2)) if m else None

    def abduct_from(sec: str | None) -> float | None:
        m = re.search(r'abduction\s+(\d+)/(\d+)', sec or '')
        return int(m.group(1)) / int(m.group(2)) if m else None

    def syllog_from(sec: str | None) -> float | None:
        m = re.search(r'syllogism\s+(\d+)/(\d+)', sec or '')
        return int(m.group(1)) / int(m.group(2)) if m else None

    def mp_from(sec: str | None) -> float | None:
        m = re.search(r'modus_ponens\s+(\d+)/(\d+)', sec or '')
        return int(m.group(1)) / int(m.group(2)) if m else None

    prim  = re.search(r'primitive\s+macroF1=([\d.]+)', t)
    r2p   = re.search(r'role_to_primitive\s+macroF1=([\d.]+)', t)
    lm    = re.search(r'LM loss: ([\d.]+)', t)
    ent   = re.search(r"'refuted': '(\d+)/(\d+)'", t)
    idk   = re.search(r'idk\s+acc=([\d.]+)', t)
    supp  = re.search(r'support\s+acc=([\d.]+)', t)
    verif = re.search(r'verifier\s+acc=([\d.]+)', t)
    proto = re.search(r'proto_role\s+F1=([\d.]+)', t)
    argr  = re.search(r'arg_role\s+F1=([\d.]+)', t)

    return {
        'prim':        float(prim.group(1)) if prim else None,
        'r2p':         float(r2p.group(1))  if r2p else None,
        'lm':          float(lm.group(1))   if lm else None,
        'ent_ref':     int(ent.group(1)) / int(ent.group(2)) if ent else None,
        'ent_ref_precision': None,
        'ent_ref_f1': None,
        'idk':         float(idk.group(1))  if idk else None,
        'support':     float(supp.group(1)) if supp else None,
        'verifier':    float(verif.group(1)) if verif else None,
        'proto_role':  float(proto.group(1)) if proto else None,
        'arg_role':    float(argr.group(1)) if argr else None,
        'contra_prim': contra_from(prim_sec.group() if prim_sec else None),
        'contra_prim_precision': None,
        'contra_prim_f1': None,
        'contra_r2p':  contra_from(r2p_sec.group()  if r2p_sec else None),
        'contra_r2p_precision': None,
        'contra_r2p_f1': None,
        'abduct_prim': abduct_from(prim_sec.group() if prim_sec else None),
        'syllog_prim': syllog_from(prim_sec.group() if prim_sec else None),
        'mp_prim':     mp_from(prim_sec.group()     if prim_sec else None),
    }


def parse_json(path: Path) -> dict:
    if not path.exists():
        return {}
    r = json.loads(path.read_text())
    metrics = r.get("metrics") or {}

    def mget(name: str, key: str) -> float | None:
        v = metrics.get(name)
        if not isinstance(v, dict):
            return None
        x = v.get(key)
        return float(x) if isinstance(x, (int, float)) else None

    def cls(name: str, label: str, key: str) -> float | None:
        v = metrics.get(name)
        if not isinstance(v, dict):
            return None
        pc = v.get("per_class_prf") or {}
        row = pc.get(label) or {}
        x = row.get(key)
        return float(x) if isinstance(x, (int, float)) else None

    return {
        'prim': mget('primitive', 'macro_f1'),
        'r2p': mget('role_to_primitive', 'macro_f1'),
        'lm': float(r['lm_loss']) if isinstance(r.get('lm_loss'), (int, float)) else None,
        'ent_ref': cls('entailment_state', 'refuted', 'recall'),
        'ent_ref_precision': cls('entailment_state', 'refuted', 'precision'),
        'ent_ref_f1': cls('entailment_state', 'refuted', 'f1'),
        'idk': mget('idk', 'accuracy'),
        'support': mget('support', 'accuracy'),
        'verifier': mget('verifier', 'accuracy'),
        'proto_role': mget('proto_role', 'micro_f1'),
        'arg_role': mget('arg_role', 'micro_f1'),
        'contra_prim': cls('primitive', 'contradiction', 'recall'),
        'contra_prim_precision': cls('primitive', 'contradiction', 'precision'),
        'contra_prim_f1': cls('primitive', 'contradiction', 'f1'),
        'contra_r2p': cls('role_to_primitive', 'contradiction', 'recall'),
        'contra_r2p_precision': cls('role_to_primitive', 'contradiction', 'precision'),
        'contra_r2p_f1': cls('role_to_primitive', 'contradiction', 'f1'),
        'abduct_prim': cls('primitive', 'abduction', 'recall'),
        'syllog_prim': cls('primitive', 'syllogism', 'recall'),
        'mp_prim': cls('primitive', 'modus_ponens', 'recall'),
    }


def parse_eval(report_dir: Path, tag: str, ds: str, seed: int) -> dict:
    js = parse_json(report_dir / f'matrix_{tag}_{ds}_s{seed}.json')
    return js if js else parse_log(report_dir / f'matrix_{tag}_{ds}_s{seed}.log')

# ── stats ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 rng: random.Random | None = None) -> tuple[float, float]:
    rng = rng or random.Random(42)
    n = len(values)
    if n < 2:
        v = values[0] if values else float('nan')
        return v, v
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return lo, hi

def stats(vals: list[float | None]) -> dict:
    vs = [v for v in vals if v is not None]
    if not vs:
        return {'n': 0, 'mean': None, 'std': None, 'ci95_lo': None, 'ci95_hi': None, 'values': []}
    mean = sum(vs) / len(vs)
    std  = (sum((v - mean) ** 2 for v in vs) / max(len(vs) - 1, 1)) ** 0.5
    lo, hi = bootstrap_ci(vs)
    return {'n': len(vs), 'mean': round(mean, 4), 'std': round(std, 4),
            'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4), 'values': vs}

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0-7',
                    help='Seed range, e.g. "0-7" or "0,1,2"')
    ap.add_argument('--report-dir', default='artifacts/reports')
    ap.add_argument('--output', default='artifacts/reports/matrix_stats.json')
    args = ap.parse_args()

    rdir = Path(args.report_dir)

    # Parse seed range
    if '-' in args.seeds:
        lo_s, hi_s = args.seeds.split('-')
        seeds = list(range(int(lo_s), int(hi_s) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]

    METRICS = ['prim', 'r2p', 'lm', 'ent_ref', 'ent_ref_precision', 'ent_ref_f1',
               'idk', 'support', 'verifier', 'proto_role', 'arg_role',
               'contra_prim', 'contra_prim_precision', 'contra_prim_f1',
               'contra_r2p', 'contra_r2p_precision', 'contra_r2p_f1',
               'abduct_prim', 'syllog_prim', 'mp_prim']

    results: dict = {}
    for tag in ['b', 'g', 'c', 'd']:
        for ds in ['shallow', 'deep']:
            key = f'{tag}_{ds}'
            rows = [parse_eval(rdir, tag, ds, s) for s in seeds]
            present = sum(1 for r in rows if r)
            results[key] = {
                'seeds_present': present,
                'seeds_requested': len(seeds),
            }
            for m in METRICS:
                results[key][m] = stats([r.get(m) for r in rows])

    # Paired deltas B→C and C→D (matched by seed)
    for ds in ['shallow', 'deep']:
        for delta_tag, src, dst in [
            ('B_to_G', 'b', 'g'),
            ('G_to_C', 'g', 'c'),
            ('B_to_C', 'b', 'c'),
            ('C_to_D', 'c', 'd'),
        ]:
            key = f'{delta_tag}_{ds}'
            results[key] = {}
            for m in METRICS:
                b_vals = results[f'{src}_{ds}'][m]['values']
                c_vals = results[f'{dst}_{ds}'][m]['values']
                paired = [(c - b) for b, c in zip(b_vals, c_vals)
                          if b is not None and c is not None]
                results[key][m] = stats(paired) if paired else {'n': 0}

    # Print summary table
    print(f"\n{'='*70}")
    print('PAT-ER CONTRIBUTION MATRIX — STATISTICAL SUMMARY')
    print(f"Seeds analysed: {seeds}  (n={len(seeds)})")
    print(f"{'='*70}\n")

    def fmt(s: dict) -> str:
        if s.get('n', 0) == 0 or s.get('mean') is None:
            return '—'
        return f"{s['mean']:.3f}±{s['std']:.3f} [{s['ci95_lo']:.3f},{s['ci95_hi']:.3f}]"

    header_metrics = ['prim', 'r2p', 'ent_ref', 'contra_prim', 'lm']
    print(f"{'Condition':20s}  " + '  '.join(f'{m:>20s}' for m in header_metrics))
    print('-' * (22 + 22 * len(header_metrics)))
    for tag, label in [('b','B token-state'), ('g','G generic-reg'), ('c','C PAT-ER reg'), ('d','D Stage5B')]:
        for ds in ['shallow', 'deep']:
            row = results[f'{tag}_{ds}']
            n = row['seeds_present']
            cells = '  '.join(f"{fmt(row[m]):>20s}" for m in header_metrics)
            print(f"{label+' '+ds:20s}  {cells}  (n={n})")
        print()

    for dtag in ['B_to_G', 'G_to_C', 'B_to_C', 'C_to_D']:
        print(f'\nPAIRED DELTAS ({dtag.replace("_to_", "→")}, shallow OWA):')
        row = results[f'{dtag}_shallow']
        for m in ['prim', 'r2p', 'ent_ref', 'contra_prim', 'contra_prim_f1']:
            s = row[m]
            if s.get('n', 0):
                print(f"  {m:20s}: {s['mean']:+.3f} ± {s['std']:.3f}  "
                      f"95%CI [{s['ci95_lo']:+.3f}, {s['ci95_hi']:+.3f}]  (n={s['n']})")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'\nFull stats → {args.output}')


if __name__ == '__main__':
    main()
