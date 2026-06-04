#!/usr/bin/env python3
"""Convert ProofWriter-style rule-reasoning examples into PAT-ER records (offline).

Input fixture schema (one JSON object per line under --input):

    {
      "id": "pw_train_0001",          # unique example id
      "theory_id": "t_0001",           # optional; defaults to id (split-group key)
      "split": "train",                # train | dev/val | test
      "license": "...",                # license note for the source
      "facts": ["tests passed"],
      "rules": [{"id": "r1", "if": ["tests passed"], "then": "release ok"}],
      "question": "release ok",
      "answer": "true",                # true | false | unknown (sanity check only)
      "trap": false                    # optional: abduction-as-proof hard negative
    }

Mapping (structural, not by copying the label):
- the question is forward-chained from facts+rules:
  - derivable in 1 rule        -> modus_ponens (proof, support proof);
  - derivable in >=2 rules     -> syllogism (chain, support proof);
- not derivable, but the question is the antecedent of a rule whose consequent is
  observed/derivable           -> abduction (hypothesis, never proof);
- answer "false"               -> contradiction (conflict / abstain);
- otherwise (unknown)          -> contingency (needs_evidence).

The proof is used only to build the event_graph/formula; no chain-of-thought text
is emitted. No downloads, no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import external_common as ec


def _theory_split(theory_id: str, val_ratio: float = 0.1, test_ratio: float = 0.1) -> str:
    """Deterministic per-theory split (stable hash). Used for meta-abduction, which
    ships train-only upstream; theory-level holdout is preserved (all questions of
    one theory share a split)."""

    h = (int(hashlib.sha1(theory_id.encode("utf-8")).hexdigest(), 16) % 1000) / 1000.0
    if h < test_ratio:
        return "test"
    if h < test_ratio + val_ratio:
        return "val"
    return "train"

FAMILY = "ext_proofwriter"
PW_LICENSE_NOTE = ("Allen Institute for AI ProofWriter V2020.12.3 (aristo-data-public S3); "
                   "license not stated on the download page (possible AI2 ImpACT terms) - confirm before "
                   "redistribution; cite Tafjord et al. 2021 arXiv:2012.13048")


def _path_meta(path: Path) -> dict[str, Any]:
    """Parse ProofWriter release / world-assumption / depth / task-subset from the
    file path (e.g. .../proofwriter-dataset-V2020.12.3/OWA/depth-1/meta-abduct-train.jsonl)."""

    # Resolve symlinks so a staging dir still recovers the real OWA/depth/release path.
    try:
        parts = path.resolve().parts
    except OSError:
        parts = path.parts
    release = next((p for p in parts if p.startswith("proofwriter-dataset")), None)
    world = next((p for p in parts if p in ("OWA", "CWA")), None)
    depth = next((p for p in parts if p.startswith("depth-")), None)
    depth_n = int(depth.split("-")[1]) if depth and depth.split("-")[1].isdigit() else None
    subset = "meta-abduct" if "abduct" in path.stem else "meta"
    return {"pw_release": release, "world_assumption": world, "depth": depth_n, "task_subset": subset}


def _resolve_triple_texts(proof: str, triples: dict[str, Any]) -> list[str]:
    ids = list(dict.fromkeys(re.findall(r"triple\d+", proof or "")))
    out = []
    for t in ids:
        node = triples.get(t)
        if isinstance(node, dict) and node.get("text"):
            out.append(node["text"])
    return out


def _reasoning_labels(question: dict[str, Any], triples: dict[str, Any]) -> dict[str, Any]:
    """Non-CoT reasoning-supervision targets from ProofWriter proof metadata
    (audit: docs/results/proofwriter_recoverability_audit.md). For answer=false the
    inv-proof derives the OPPOSITE statement, so its facts are the contradicting set."""

    ans = question.get("answer")
    state = "entailed" if ans is True else "refuted" if ans is False else "unknown"
    qdep = question.get("QDep")
    depth = int(qdep) if (isinstance(qdep, int) or (isinstance(qdep, str) and str(qdep).isdigit())) else None
    proof = question.get("proofs", "") or ""
    n_rules = len(set(re.findall(r"rule\d+", proof)))  # rule-chain length (0 at depth-0)
    facts = _resolve_triple_texts(proof, triples)
    return {"entailment_state": state, "proof_depth": depth,
            "rule_chain_length": (n_rules if state in ("entailed", "refuted") else None),
            "supporting_fact_texts": facts if state == "entailed" else [],
            "contradicting_fact_texts": facts if state == "refuted" else []}


def infer_split(filename: str) -> str | None:
    name = filename.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return None


def _proof_antecedents(question: dict[str, Any], triples: dict[str, Any], rules: dict[str, Any]) -> tuple[list[str], list[str], str | None, bool]:
    """Extract the true antecedent facts/rules and the middle term from the proof.

    ProofWriter proofs reference tripleN / ruleN ids (e.g. "[(((triple1) -> rule5))]")
    and proofsWithIntermediates carries the chain. A FAIL proof (unknown) has no
    triple -> the binding is approximate."""

    proof = question.get("proofs", "") or ""
    triple_ids = list(dict.fromkeys(re.findall(r"triple\d+", proof)))
    rule_ids = list(dict.fromkeys(re.findall(r"rule\d+", proof)))
    antecedent_facts = [triples[t]["text"] for t in triple_ids if t in triples]
    antecedent_rules = [rules[r]["text"] for r in rule_ids if r in rules]
    middle = None
    pwi = question.get("proofsWithIntermediates") or []
    if pwi:
        for step in (pwi[0].get("intermediates") or {}).values():
            text = step.get("text")
            if text and text != question.get("question"):
                middle = text
                break
    return antecedent_facts, antecedent_rules, middle, bool(antecedent_facts)


def expand_theory(theory: dict[str, Any], split: str, max_questions: int | None) -> list[dict[str, Any]]:
    """Expand one upstream ProofWriter theory into per-question example dicts.

    Rules are kept as natural-language text (variable/first-order, not
    propositional), so the converter classifies by the provided answer + QDep and
    binds the event-role argument to the antecedent recovered from the proof."""

    triples = theory.get("triples", {})
    rules = theory.get("rules", {})
    facts = [t["text"] for t in triples.values()]
    rule_texts = [r["text"] for r in rules.values()]
    theory_id = theory.get("id", "t")
    out = []
    for qkey, question in list(theory.get("questions", {}).items())[: max_questions or None]:
        raw = question.get("answer")
        answer = "true" if raw is True else "false" if raw is False else "unknown"
        ant_facts, ant_rules, middle, exact = _proof_antecedents(question, triples, rules)
        out.append({
            "id": f"pw_{theory_id}_{qkey}",
            "theory_id": f"pw_{theory_id}",
            "split": split,
            "license": PW_LICENSE_NOTE,
            "facts": facts,
            "rules": rule_texts,
            "question": question["question"],
            "answer": answer,
            "qdep": question.get("QDep"),
            "antecedent_facts": ant_facts,
            "antecedent_rules": ant_rules,
            "middle": middle,
            "binding_exact": exact,
            "reasoning": _reasoning_labels(question, triples),
        })
    return out


def expand_abduction_theory(theory: dict[str, Any], split: str, max_questions: int | None) -> list[dict[str, Any]]:
    """Expand one meta-abduction theory into abduction examples.

    Each abduction asks what missing fact would make `question` provable; the
    first listed answer is the abduced premise (a real proposition, identifiable).
    Abductions with no answer (nothing abducible) are skipped."""

    triples = theory.get("triples", {})
    rules = theory.get("rules", {})
    facts = [t["text"] for t in triples.values()]
    rule_texts = [r["text"] for r in rules.values()]
    theory_id = theory.get("id", "t")
    # meta-abduction ships train-only; re-split by theory so hypothesis/abduction
    # appears in val/test too (holdout preserved -- whole theory to one split).
    resplit = _theory_split(f"pw_{theory_id}")
    out = []
    for qkey, ab in list(theory.get("abductions", {}).items())[: max_questions or None]:
        answers = ab.get("answers") or []
        if not answers:
            continue  # nothing abducible -> no hypothesis signal
        abduced = answers[0].get("text")
        if not abduced:
            continue
        out.append({
            "id": f"pw_{theory_id}_{qkey}",
            "theory_id": f"pw_{theory_id}",
            "split": resplit,
            "license": PW_LICENSE_NOTE,
            "facts": facts,
            "rules": rule_texts,
            "question": ab["question"],     # the goal to explain
            "abduced_fact": abduced,         # the hypothesised missing premise
            "qdep": answers[0].get("QDep"),
            "task": "abduction",
        })
    return out


def build_abduction(example: dict[str, Any], render_mode: str) -> tuple[dict[str, Any], str]:
    """Map a meta-abduction example to an abduction/hypothesis record, mirroring the
    existing abduction convention: arg0 = goal (observed), arg1 = abduced premise
    (the hypothesis); bridge_binding = arg0; support = hypothesis, never proof."""

    question = example["question"]
    abduced = example["abduced_fact"]
    facts = example.get("facts", [])
    args = [ec.arg("<arg:0>", question, ["<role:undergoes_change>", "<role:affected>"]),
            ec.arg("<arg:1>", abduced, ["<role:source>"])]
    spans = [question, abduced]
    if render_mode == "compact":
        text = "query " + "; ".join(dict.fromkeys(spans))
        evidence = ec.evidence_items(example["id"], facts[:2])
        formula = None
    else:
        text = render_text(facts, example.get("rules", []), question)
        evidence = ec.evidence_items(example["id"], facts)
        formula = f"<atom> {abduced} </atom> IMPLIES <atom> {question} </atom>"
    record = ec.to_pater_record(
        source_dataset="proofwriter", source_split=example.get("split", "train"),
        source_id=example["id"], theory_id=example.get("theory_id", example["id"]),
        license_note=example.get("license", PW_LICENSE_NOTE), family=FAMILY,
        primitive_class="abduction", support_status="hypothesis", idk_action="needs_verification",
        verifier_accept=False, text=text, target="<support:hypothesis>", evidence=evidence, formula=formula,
        event_graph=ec.single_event("abduce", args, "<prim:abduction>", "<support:hypothesis>"),
        bridge_binding="<arg:0>", binding="approximate", extra_provenance=example.get("provenance_extra"),
        source_payload={"facts": facts, "rules": example.get("rules", []), "question": question,
                        "abduced": abduced, "qdep": example.get("qdep")},
        rationale="meta-abduction: abduce a missing premise (arg1) to explain the goal (arg0); hypothesis, never proof")
    return record, "abduction"


def forward_chain(facts: list[str], rules: list[dict[str, Any]]) -> set[str]:
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule["then"] not in known and all(a in known for a in rule["if"]):
                known.add(rule["then"])
                changed = True
    return known


def proof_rules(question: str, facts: list[str], rules: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Rules used to derive the question (backward over the forward closure), or None."""

    known = forward_chain(facts, rules)
    if question not in known:
        return None
    needed = [question]
    used: list[dict[str, Any]] = []
    seen: set[str] = set()
    while needed:
        atom = needed.pop(0)
        if atom in facts or atom in seen:
            continue
        seen.add(atom)
        for rule in rules:
            if rule["then"] == atom and all(a in known for a in rule["if"]):
                used.append(rule)
                needed.extend(rule["if"])
                break
    return used


def find_abduction(question: str, facts: list[str], rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    known = forward_chain(facts, rules)
    if question in known:
        return None
    for rule in rules:
        if rule["then"] in known and question in rule["if"] and question not in known:
            return rule
    return None


def _formula(rules_used: list[dict[str, Any]]) -> str:
    parts = [f"( <atom> {' AND '.join(r['if'])} </atom> IMPLIES <atom> {r['then']} </atom> )" for r in rules_used]
    return " AND ".join(parts) if parts else ""


def render_text(facts: list[str], rules: list[Any], question: str) -> str:
    fact_s = "; ".join(facts) if facts else "none"
    rule_parts = [r if isinstance(r, str) else f"if {' and '.join(r['if'])} then {r['then']}" for r in rules]
    rule_s = "; ".join(rule_parts) if rule_parts else "none"
    return f"facts {fact_s} rules {rule_s} question {question}"


_PRIM_TOKEN = {"observation": "<prim:observation>", "modus_ponens": "<prim:modus_ponens>",
               "syllogism": "<prim:syllogism>", "contradiction": "<prim:contradiction>",
               "contingency": "<prim:contingency>"}
_EVENT_PREDICATE = {"observation": "observe", "modus_ponens": "entail", "syllogism": "entail",
                    "contradiction": "contradict", "contingency": "determine"}


def build_pw(example: dict[str, Any], render_mode: str) -> tuple[dict[str, Any], str]:
    """Map a real ProofWriter question by answer + proof depth, binding the
    event-role argument to the proof's antecedent (exact) where available."""

    question = example["question"]
    facts = example.get("facts", [])
    answer = str(example.get("answer", "")).lower()
    qdep = example.get("qdep")
    qdep = int(qdep) if qdep is not None else None
    antecedent_facts = example.get("antecedent_facts") or []
    antecedent_rules = example.get("antecedent_rules") or []
    middle = example.get("middle")
    exact = bool(example.get("binding_exact"))
    relevant = antecedent_facts if antecedent_facts else facts[:2]

    if answer == "true" and qdep is not None and qdep <= 0:
        primitive, support, target = "observation", "belief", "<support:belief>"
        idk, verifier = "answer", True
        args = [ec.arg("<arg:0>", question, ["<role:undergoes_change>", "<role:affected>"])]
        spans = [question]
    elif answer == "true" and qdep == 1:
        primitive, support, target = "modus_ponens", "proof", "<support:proof>"
        idk, verifier = "answer", True
        a0 = relevant[0] if relevant else question
        args = [ec.arg("<arg:0>", a0, ["<role:source>", "<role:causes_change>"]),
                ec.arg("<arg:1>", question, ["<role:goal>", "<role:undergoes_change>"])]
        spans = [a0, question]
    elif answer == "true":  # qdep >= 2
        primitive, support, target = "syllogism", "proof", "<support:proof>"
        idk, verifier = "answer", True
        mid = middle or (relevant[0] if relevant else question)
        leaf = relevant[0] if relevant else question
        args = [ec.arg("<arg:0>", mid, ["<role:source>", "<role:path>"]),
                ec.arg("<arg:1>", leaf, ["<role:source>"]),
                ec.arg("<arg:2>", question, ["<role:goal>"])]
        spans = [mid, leaf, question]
    elif answer == "false":
        primitive, support, target = "contradiction", "conflict", "<conflict>"
        idk, verifier = "needs_verification", False
        a0 = relevant[0] if relevant else question
        args = [ec.arg("<arg:0>", a0, ["<role:exists_independently>"]),
                ec.arg("<arg:1>", question, ["<role:exists_independently>"])]
        spans = [a0, question]
    else:  # unknown
        primitive, support, target = "contingency", "unknown", "<needs_evidence>"
        idk, verifier = "needs_evidence", False
        args = [ec.arg("<arg:0>", question, ["<role:undergoes_change>"])]
        spans = [question]
        relevant = []  # undetermined: no supporting context

    if render_mode == "compact":
        text = "query " + "; ".join(dict.fromkeys(spans))
        evidence = ec.evidence_items(example["id"], relevant)
        formula = " ; ".join(antecedent_rules) if antecedent_rules else None
    else:
        text = render_text(facts, example.get("rules", []), question)
        evidence = ec.evidence_items(example["id"], facts)
        formula = None

    record = ec.to_pater_record(
        source_dataset="proofwriter", source_split=example.get("split", "train"),
        source_id=example["id"], theory_id=example.get("theory_id", example["id"]),
        license_note=example.get("license", PW_LICENSE_NOTE), family=FAMILY,
        primitive_class=primitive, support_status=support, idk_action=idk, verifier_accept=verifier,
        text=text, target=target, evidence=evidence, formula=formula,
        event_graph=ec.single_event(_EVENT_PREDICATE[primitive], args, _PRIM_TOKEN[primitive], f"<support:{support}>"),
        bridge_binding="<arg:0>", binding="exact" if exact else "approximate",
        extra_provenance=example.get("provenance_extra"),
        source_payload={"facts": facts, "rules": example.get("rules", []), "question": question,
                        "answer": answer, "qdep": qdep, "proof_facts": antecedent_facts, "proof_rules": antecedent_rules},
        rationale=f"{primitive} at proof depth {qdep} ({'exact' if exact else 'approximate'} antecedent binding)")
    if example.get("reasoning") is not None:
        record["reasoning"] = example["reasoning"]
    return record, primitive


def convert_example(example: dict[str, Any], render_mode: str = "full") -> tuple[dict[str, Any], str]:
    if example.get("task") == "abduction":
        return build_abduction(example, render_mode)
    facts = list(example.get("facts", []))
    rules = list(example.get("rules", []))
    question = example["question"]
    answer = str(example.get("answer", "")).lower()
    source_id = example["id"]
    theory_id = example.get("theory_id", source_id)
    license_note = example.get("license", "unknown; verify source license before redistribution")

    # Real ProofWriter (variable/first-order rules) is classified by answer + proof
    # depth with antecedent binding from the proof; our propositional fixtures are
    # derived structurally via forward chaining.
    propositional = bool(rules) and all(isinstance(r, dict) and "if" in r and "then" in r for r in rules)
    if not propositional:
        return build_pw(example, render_mode)

    text = render_text(facts, rules, question)
    evidence = ec.evidence_items(source_id, facts)
    common = dict(source_dataset="proofwriter", source_split=example.get("split", "train"),
                  source_id=source_id, theory_id=theory_id, license_note=license_note, family=FAMILY,
                  text=text, evidence=evidence)
    used = proof_rules(question, facts, rules)
    abd = None if used else find_abduction(question, facts, rules)

    if used:
        primitive = "modus_ponens" if len(used) == 1 else "syllogism"
        # arg0 = the observed premise that fires the chain; argN = conclusion.
        antecedent = used[-1]["if"][0]
        args = [ec.arg("<arg:0>", antecedent, ["<role:source>", "<role:causes_change>"]),
                ec.arg("<arg:1>", question, ["<role:goal>", "<role:undergoes_change>"])]
        if primitive == "syllogism":
            middle = used[0]["if"][0]
            args = [ec.arg("<arg:0>", middle, ["<role:source>", "<role:path>"]),
                    ec.arg("<arg:1>", antecedent, ["<role:source>"]),
                    ec.arg("<arg:2>", question, ["<role:goal>"])]
        prim_token = "<prim:modus_ponens>" if primitive == "modus_ponens" else "<prim:syllogism>"
        record = ec.to_pater_record(
            **common, primitive_class=primitive, support_status="proof", idk_action="answer",
            verifier_accept=True, target="<support:proof>", formula=_formula(used),
            event_graph=ec.single_event("entail", args, prim_token, "<support:proof>"),
            bridge_binding="<arg:0>", rationale="question is forward-derivable from facts and rules")
        return record, primitive

    if abd is not None:
        observed = abd["then"]
        trap = bool(example.get("trap"))
        args = [ec.arg("<arg:0>", observed, ["<role:undergoes_change>", "<role:affected>"]),
                ec.arg("<arg:1>", question, ["<role:source>"])]
        record = ec.to_pater_record(
            **common, primitive_class="abduction", support_status="hypothesis",
            idk_action="needs_verification", verifier_accept=False, target="<support:hypothesis>",
            formula=f"<atom> {question} </atom> IMPLIES <atom> {observed} </atom>",
            event_graph=ec.single_event("abduce", args, "<prim:abduction>", "<support:hypothesis>"),
            bridge_binding="<arg:0>", is_hard_negative=trap,
            hard_negative_type="abduction_as_proof" if trap else None,
            rationale="observed consequent (arg0) -> question is a hypothesis, never proof")
        return record, "abduction"

    if answer == "false":
        args = [ec.arg("<arg:0>", question, ["<role:exists_independently>"]),
                ec.arg("<arg:1>", f"not {question}", ["<role:exists_independently>"])]
        record = ec.to_pater_record(
            **common, primitive_class="contradiction", support_status="conflict",
            idk_action="needs_verification", verifier_accept=False, target="<conflict>",
            formula=f"<atom> {question} </atom> CONTRADICTS theory",
            event_graph=ec.single_event("contradict", args, "<prim:contradiction>", "<support:conflict>"),
            bridge_binding="<arg:1>", rationale="question is refuted by the theory")
        return record, "contradiction"

    args = [ec.arg("<arg:0>", question, ["<role:undergoes_change>"])]
    record = ec.to_pater_record(
        **common, primitive_class="contingency", support_status="unknown",
        idk_action="needs_evidence", verifier_accept=False, target="<needs_evidence>",
        event_graph=ec.single_event("determine", args, "<prim:contingency>", "<support:unknown>"),
        bridge_binding="<arg:0>", rationale="question is undetermined by the theory")
    return record, "contingency"


def _iter_examples(input_dir: Path, max_theories: int | None, max_questions: int | None):
    """Yield example dicts: upstream ProofWriter theories are expanded into
    per-question examples; our propositional fixtures pass through unchanged."""

    for path in sorted(input_dir.glob("*.jsonl")):
        split = infer_split(path.name)
        pmeta = _path_meta(path)
        theories = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "abductions" in record and "triples" in record:  # meta-abduction theory
                if max_theories is not None and theories >= max_theories:
                    break
                theories += 1
                for ex in expand_abduction_theory(record, split or "train", max_questions):
                    ex["provenance_extra"] = pmeta
                    yield ex
            elif "questions" in record and "triples" in record:  # upstream ProofWriter theory
                if max_theories is not None and theories >= max_theories:
                    break
                theories += 1
                for ex in expand_theory(record, split or "train", max_questions):
                    ex["provenance_extra"] = pmeta
                    yield ex
            else:  # our fixture schema
                record.setdefault("split", split or "train")
                yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ProofWriter (real or fixture) to PAT-ER JSONL (offline).")
    parser.add_argument("--input", type=Path, required=True, help="Directory of *.jsonl files.")
    parser.add_argument("--output", type=Path, required=True, help="Output PAT-ER JSONL path.")
    parser.add_argument("--render-mode", choices=["full", "compact"], default="full",
                        help="compact = query-first text + antecedent-only context (fits a small context).")
    parser.add_argument("--max-theories", type=int, default=60, help="Cap theories per upstream file (None for all).")
    parser.add_argument("--max-questions", type=int, default=None, help="Cap questions per theory.")
    args = parser.parse_args()

    records = []
    counts: dict[str, int] = {}
    binding = {"exact": 0, "approximate": 0}
    for example in _iter_examples(Path(args.input), args.max_theories, args.max_questions):
        record, primitive = convert_example(example, args.render_mode)
        records.append(record)
        counts[primitive] = counts.get(primitive, 0) + 1
        binding[record["provenance"].get("binding", "approximate")] += 1

    ec.write_jsonl(args.output, records)
    total = len(records) or 1
    print(f"converted {len(records)} ProofWriter examples -> {args.output} (render={args.render_mode})")
    print(f"primitive mapping: {dict(sorted(counts.items()))}")
    print(f"binding: exact={binding['exact']} ({100*binding['exact']//total}%), approximate={binding['approximate']}")


if __name__ == "__main__":
    main()
