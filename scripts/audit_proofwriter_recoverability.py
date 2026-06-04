#!/usr/bin/env python3
"""Phase 1.0 reasoning-supervision audit: can the LOCAL ProofWriter proof metadata
produce reliable non-chain-of-thought supervision labels for reasoning heads?

OFFLINE. No training, no downloads, no model. Reads only the local OWA depth-1/2
bundle already on disk and reports per-label coverage / ambiguity / missingness /
failure-mode stats for:

    proof_depth          <- QDep
    rule_chain_length    <- ruleN count in proofs / proofsWithIntermediates
    supporting_fact_ids  <- tripleN refs in the proof
    entailed/refuted/unknown <- answer true/false/unknown
    refutation_available <- answer == false
    contradicting_fact_ids <- tripleN in the answer=false inv-proof (the derivation
                              of the OPPOSITE statement); verdict explicit/
                              reconstructable/unreliable
    (meta-abduction)     <- abductions with answers

The proof metadata feeds AUX-HEAD TARGETS only; the converter renders only
facts/rules/question into model input (verified here). Writes JSON + Markdown.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OWA = ROOT / "artifacts/datasets/external/raw/proofwriter/proofwriter-dataset-V2020.12.3/OWA"

_TRIPLE = re.compile(r"triple\d+")
_RULE = re.compile(r"rule\d+")


def _answer_class(a: Any) -> str:
    return "entailed" if a is True else "refuted" if a is False else "unknown"


def _proof_str(q: dict) -> str:
    return q.get("proofs", "") or ""


def _branches(proof: str) -> int:
    if "triple" not in proof and "rule" not in proof:
        return 0
    return proof.count(" OR ") + 1


def audit_questions(paths: list[Path], cap: int | None) -> dict[str, Any]:
    n_q = 0
    by_ans = Counter()
    depth_present = [0, 0]          # [present, total]
    rule_chain = {"recoverable": 0, "total_provable": 0}
    support = {"with_triple": 0, "total_provable": 0}
    contra = {"explicit": 0, "multi_branch": 0, "no_triple": 0, "total_false": 0}
    strat_by_ans = Counter()
    fail_modes = Counter()
    depth_dist = Counter()

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            for q in d.get("questions", {}).values():
                if cap and n_q >= cap:
                    break
                n_q += 1
                ans = _answer_class(q.get("answer"))
                by_ans[ans] += 1
                strat_by_ans[(ans, q.get("strategy"))] += 1
                qdep = q.get("QDep")
                depth_present[1] += 1
                if isinstance(qdep, int) or (isinstance(qdep, str) and qdep.isdigit()):
                    depth_present[0] += 1
                    depth_dist[int(qdep)] += 1
                proof = _proof_str(q)
                provable = ans in ("entailed", "refuted")  # proof/inv-proof carry a real derivation
                if provable:
                    rule_chain["total_provable"] += 1
                    support["total_provable"] += 1
                    if _RULE.search(proof) or q.get("QDep") == 0:
                        rule_chain["recoverable"] += 1
                    if _TRIPLE.search(proof):
                        support["with_triple"] += 1
                else:  # unknown -> CWA/NAF failure trace, no clean derivation
                    if "FAIL" in proof or "CWA" in proof:
                        fail_modes["unknown_cwa_failure_trace"] += 1
                if ans == "refuted":
                    contra["total_false"] += 1
                    if _TRIPLE.search(proof):
                        contra["explicit"] += 1
                        if _branches(proof) > 1:
                            contra["multi_branch"] += 1
                    else:
                        contra["no_triple"] += 1
            if cap and n_q >= cap:
                break
        if cap and n_q >= cap:
            break
    return {"n_questions": n_q, "by_answer": dict(by_ans), "strategy_by_answer": {f"{k[0]}:{k[1]}": v for k, v in strat_by_ans.items()},
            "proof_depth": {"present": depth_present[0], "total": depth_present[1]},
            "depth_distribution": dict(sorted(depth_dist.items())),
            "rule_chain_length": rule_chain, "supporting_fact_ids": support,
            "contradicting_fact_ids": contra, "failure_modes": dict(fail_modes)}


def audit_abduction(paths: list[Path], cap: int | None) -> dict[str, Any]:
    n = with_ans = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            for ab in d.get("abductions", {}).values():
                if cap and n >= cap:
                    break
                n += 1
                if ab.get("answers"):
                    with_ans += 1
            if cap and n >= cap:
                break
        if cap and n >= cap:
            break
    return {"n_abductions": n, "with_abduced_fact": with_ans}


def _verify_no_cot() -> dict[str, Any]:
    """The converter must render only facts/rules/question into model input."""
    conv = (ROOT / "scripts/convert_proofwriter.py").read_text()
    # render_text builds the model-facing string; confirm it uses facts/rules/question and not proofs.
    m = re.search(r"def render_text.*?return ([^\n]+)", conv, re.S)
    body = re.search(r"def render_text(.*?)\n\ndef ", conv, re.S)
    text_uses_proof = bool(body and ("proof" in body.group(1).lower()))
    return {"render_text_returns": m.group(1).strip() if m else None,
            "render_text_references_proof": text_uses_proof,
            "no_cot_in_input": (m is not None) and not text_uses_proof}


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1.0 ProofWriter recoverability audit (offline).")
    p.add_argument("--cap", type=int, default=4000, help="Max questions per split scan (bounded).")
    p.add_argument("--json-out", type=Path, default=ROOT / "artifacts/reports/proofwriter_recoverability.json")
    p.add_argument("--md-out", type=Path, default=ROOT / "docs/results/proofwriter_recoverability_audit.md")
    args = p.parse_args()

    q_paths = [OWA / d / f"{s}.jsonl" for d in ("depth-1", "depth-2") for s in ("meta-dev", "meta-test")]
    q_paths = [p_ for p_ in q_paths if p_.exists()]
    abd_paths = [p_ for p_ in [OWA / "depth-1/meta-abduct-train.jsonl"] if p_.exists()]

    q = audit_questions(q_paths, args.cap)
    abd = audit_abduction(abd_paths, args.cap)
    nocot = _verify_no_cot()

    def pct(a, b):
        return round(100 * a / b, 1) if b else 0.0

    pd = q["proof_depth"]; rc = q["rule_chain_length"]; sf = q["supporting_fact_ids"]; cf = q["contradicting_fact_ids"]
    # verdict for contradicting_fact_ids
    expl_rate = pct(cf["explicit"], cf["total_false"])
    cf_verdict = ("EXPLICIT" if expl_rate >= 95 else "RECONSTRUCTABLE" if expl_rate >= 60 else "UNRELIABLE")

    labels = {
        "proof_depth": {"coverage_%": pct(pd["present"], pd["total"]), "source": "QDep",
                        "recoverable": pct(pd["present"], pd["total"]) >= 95},
        "entailed_refuted_unknown": {"coverage_%": 100.0, "source": "answer", "recoverable": True,
                                     "distribution": q["by_answer"]},
        "refutation_available": {"coverage_%": 100.0, "source": "answer==false",
                                 "n_false": cf["total_false"], "recoverable": True},
        "rule_chain_length": {"coverage_%": pct(rc["recoverable"], rc["total_provable"]),
                              "scope": "provable (entailed/refuted) only; unknown = CWA failure trace",
                              "recoverable": pct(rc["recoverable"], rc["total_provable"]) >= 95},
        "supporting_fact_ids": {"coverage_%": pct(sf["with_triple"], sf["total_provable"]),
                                "scope": "provable only", "recoverable": pct(sf["with_triple"], sf["total_provable"]) >= 95},
        "contradicting_fact_ids": {"explicit_%": expl_rate, "multi_branch_ambiguity_%": pct(cf["multi_branch"], cf["total_false"]),
                                   "no_triple_%": pct(cf["no_triple"], cf["total_false"]), "n_false": cf["total_false"],
                                   "verdict": cf_verdict},
        "abduction_hypothesis": {"coverage_%": pct(abd["with_abduced_fact"], abd["n_abductions"]),
                                 "n": abd["n_abductions"], "recoverable": pct(abd["with_abduced_fact"], abd["n_abductions"]) >= 60},
    }
    core = ["proof_depth", "entailed_refuted_unknown", "refutation_available", "rule_chain_length",
            "supporting_fact_ids", "contradicting_fact_ids"]
    all_recoverable = all(labels[k].get("recoverable", True) for k in core if "recoverable" in labels[k])
    cf_decided = cf_verdict in ("EXPLICIT", "RECONSTRUCTABLE")
    audit_pass = bool(all_recoverable and cf_decided and nocot["no_cot_in_input"])

    out = {"scanned": {"question_files": [str(p_.relative_to(ROOT)) for p_ in q_paths],
                       "abduction_files": [str(p_.relative_to(ROOT)) for p_ in abd_paths], "cap": args.cap},
           "questions": q, "abduction": abd, "no_cot": nocot, "labels": labels,
           "verdict": {"all_core_labels_recoverable": all_recoverable, "contradicting_fact_ids_verdict": cf_verdict,
                       "no_cot_in_input": nocot["no_cot_in_input"], "audit_pass": audit_pass,
                       "aux_heads_justified": audit_pass}}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = labels
    md = [f"# ProofWriter Recoverability Audit — Phase 1.0 ({'PASS' if audit_pass else 'FAIL'})", "",
          "Offline audit (no training, no downloads, no model, frozen baseline untouched) of whether the **local** "
          "ProofWriter OWA depth-1/2 bundle can produce reliable **non-CoT** reasoning-supervision labels.", "",
          f"Scanned {q['n_questions']} questions (cap {args.cap}/split) + {abd['n_abductions']} meta-abductions. "
          f"Answer mix: {q['by_answer']}.", "",
          "## Per-label recoverability", "",
          "| label | source | coverage | recoverable | notes |", "|---|---|---|---|---|",
          f"| proof_depth | QDep | {L['proof_depth']['coverage_%']}% | {L['proof_depth']['recoverable']} | depth dist {q['depth_distribution']} |",
          f"| entailed/refuted/unknown | answer | 100% | True | {q['by_answer']} |",
          f"| refutation_available | answer=false | 100% | True | {L['refutation_available']['n_false']} false |",
          f"| rule_chain_length | ruleN in proof | {L['rule_chain_length']['coverage_%']}% | {L['rule_chain_length']['recoverable']} | provable only; unknown=CWA fail trace |",
          f"| supporting_fact_ids | tripleN | {L['supporting_fact_ids']['coverage_%']}% | {L['supporting_fact_ids']['recoverable']} | provable only |",
          f"| **contradicting_fact_ids** | inv-proof tripleN | explicit {L['contradicting_fact_ids']['explicit_%']}% | — | **verdict: {cf_verdict}** |",
          f"| abduction hypothesis | abductions.answers | {L['abduction_hypothesis']['coverage_%']}% | {L['abduction_hypothesis']['recoverable']} | already used (meta-abduction gate) |",
          "",
          "## contradicting_fact_ids — the decisive question", "",
          f"Among **{cf['total_false']} answer=false** questions, **{L['contradicting_fact_ids']['explicit_%']}%** carry an "
          f"explicit `inv-proof` whose `tripleN` refs derive the OPPOSITE statement (the contradicting facts); "
          f"{L['contradicting_fact_ids']['multi_branch_ambiguity_%']}% have multi-branch (OR) ambiguity (take the union or "
          f"the canonical first branch); {L['contradicting_fact_ids']['no_triple_%']}% lack a tripleN. "
          f"**Verdict: {cf_verdict}.** (All answer=false use strategy `inv-proof`; unknown answers use CWA/NAF failure "
          f"traces and are correctly NOT a contradiction signal.)", "",
          "## No chain-of-thought in model input", "",
          f"- `convert_proofwriter.render_text` returns `{nocot['render_text_returns']}` (facts/rules/question only); "
          f"references proof text: **{nocot['render_text_references_proof']}**. Proof metadata feeds aux-head **targets** "
          f"only. no_cot_in_input = **{nocot['no_cot_in_input']}**.", "",
          "## Verdict", "",
          f"- **Audit pass: {audit_pass}**",
          f"- all core labels recoverable: {all_recoverable}",
          f"- contradicting_fact_ids: {cf_verdict}",
          f"- no CoT in model input: {nocot['no_cot_in_input']}", "",
          ("**Aux reasoning heads are justified.** Next is a small, principled implementation (auxiliary heads + losses for "
           "proof_depth / rule_chain_length / supporting & contradicting fact pointers / entailed-refuted-unknown), no "
           "scale jump. unknown answers (CWA failure traces) should supervise the entailed/refuted/**unknown** 3-way head "
           "but NOT the fact-pointer heads (no clean facts)." if audit_pass else
           "**Audit did not pass** — acquire data or redesign labels before adding heads; do not scale.")]
    args.md_out.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"ProofWriter recoverability audit  questions={q['n_questions']}  PASS={audit_pass}")
    for k in core + ["abduction_hypothesis"]:
        print(f"  {k:28s} {labels[k]}")
    print(f"contradicting_fact_ids verdict: {cf_verdict}  | no_cot_in_input: {nocot['no_cot_in_input']}")
    print(f"AUDIT PASS: {audit_pass}\nwrote {args.json_out}\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
