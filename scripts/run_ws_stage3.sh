#!/usr/bin/env bash
# Phase 2 Stage 3 — frozen-backbone PAT-ER adaptation.
# Warm-start the Qwen3-0.6B backbone, FREEZE it, train ONLY the PAT-ER
# side-state (registers/cross-attn/adapters/pressure/aux heads) on the shallow
# reasoning mix (repaired synthetic + shallow OWA ProofWriter + FOLIO; no
# deeper/CWA). Saves a warmstart-init (BEFORE) checkpoint + the trained (AFTER)
# checkpoint, and evals both on val.
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml
OUT=artifacts/checkpoints/qwen_ws_stage3_s${SEED}
OUT0=artifacts/checkpoints/qwen_ws_stage3_init_s${SEED}
export CUDA_VISIBLE_DEVICES="${GPU}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(--config "$CFG" --tokenizer "$TOK" --warmstart-qwen auto --freeze-backbone
        --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768
        --batch-size 2 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0
        --seed "$SEED" --input-only --balanced-sampler --use-reasoning-heads --log-interval 100
        --loss-weight event_arg=0 argument_start=0 argument_end=0 argument_span=0
                      event_token=0 evidence_pointer=0
                      primitive=0.01 support=0.01 tool_intent=0.01 schema=0.01
                      idk=0.01 role_to_primitive=0.01 role_ambiguity=0.01
                      verifier=0.005 predicate_event=0.01
                      proto_role=0.01 arg_role=0.01
                      entailment_state=0.01 proof_depth=0.005 rule_chain_length=0.005)

echo "=== BEFORE: warmstart-init (0 steps) ==="
python3 -u scripts/train_lm_aux.py "${COMMON[@]}" --steps 0 --warmup-steps 0 \
  --out-dir "$OUT0" --tag "ws_stage3_init_s${SEED}"
# Eval on CPU (batch 1) to avoid OOM when another job occupies the GPU between steps.
python3 -u scripts/eval_lm_aux.py --checkpoint "$OUT0/latest.pt" --tokenizer "$TOK" \
  --dataset "$DATA" --split val --device cuda --dtype fp32 --batch-size 4 --seed "$SEED" \
  --input-only --json-out "artifacts/reports/qwen_ws_stage3_init_eval_s${SEED}.json"

echo "=== TRAIN: ${STEPS} steps, frozen backbone ==="
python3 -u scripts/train_lm_aux.py "${COMMON[@]}" --steps "$STEPS" --warmup-steps 140 \
  --out-dir "$OUT" --tag "ws_stage3_s${SEED}"
echo "TRAIN_EXIT=$?"
python3 -u scripts/eval_lm_aux.py --checkpoint "$OUT/latest.pt" --tokenizer "$TOK" \
  --dataset "$DATA" --split val --device cuda --dtype fp32 --batch-size 4 --seed "$SEED" \
  --input-only --json-out "artifacts/reports/qwen_ws_stage3_eval_s${SEED}.json"
echo "=== real-text LM + generation (BEFORE vs AFTER) ==="
python3 -u scripts/report_warmstart_realtext.py --init "$OUT0/latest.pt" --trained "$OUT/latest.pt" \
  --tokenizer "$TOK" --json-out "artifacts/reports/qwen_ws_stage3_realtext_s${SEED}.json"
echo "EVAL_EXIT=$?  DONE seed=${SEED}"
