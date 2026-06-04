#!/usr/bin/env bash
# Phase 2 Stage 6B (Condition E) — PAT-ER aux/register refresh on Stage-5B backbone.
#
# Completes the ablation matrix:
#   B: Qwen + token-pooled labels (no registers)
#   C: PAT-ER registers, frozen backbone (Stage 3)
#   D: Stage 5B as-is (registers stale from LM-only Stage 4/5)
#   E: Stage 5B + PAT-ER aux/register refresh  ← THIS
#
# Claim being tested: does the adapted warm-start backbone + refreshed PAT-ER
# side-state recover or exceed Stage-3 architecture signal?
#
# Trainable (same set as Stage 3):
#   registers, cross-attn projections, adapter_down, pressure_proj.0, aux heads
# Frozen:
#   Qwen backbone (all 28 layers), role_fuse/primitive_fuse (Stage-4 values),
#   adapter_up (Stage-5A values), pressure_proj.1 (Stage-4 values)
#
# Usage: bash scripts/run_ws_stage6b.sh <seed> <gpu> [steps=1800]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml
S5B=artifacts/checkpoints/qwen_ws_stage5b_s${SEED}
S6B=artifacts/checkpoints/qwen_ws_stage6b_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same loss weights as Stage 3 — categorical aux at 0.01, span/pointer at 0.
# "Full aux losses" = all classification heads active; span excluded for stability.
LW=(--loss-weight event_arg=0 argument_start=0 argument_end=0 argument_span=0
               event_token=0 evidence_pointer=0
               primitive=0.01 support=0.01 tool_intent=0.01 schema=0.01
               idk=0.01 role_to_primitive=0.01 role_ambiguity=0.01
               verifier=0.005 predicate_event=0.01
               proto_role=0.01 arg_role=0.01
               entailment_state=0.01 proof_depth=0.005 rule_chain_length=0.005)

echo "=== Stage 6B (Condition E): aux/register refresh on Stage-5B backbone (${STEPS} steps) ==="
python3 -u scripts/train_lm_aux.py \
  --config "$CFG" --tokenizer "$TOK" \
  --freeze-backbone \
  --resume-from "$S5B/latest.pt" \
  --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768 \
  --batch-size 2 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler --use-reasoning-heads \
  --log-interval 100 --warmup-steps 140 --steps "$STEPS" \
  --out-dir "$S6B" --tag "ws_stage6b_s${SEED}" \
  "${LW[@]}"

echo "=== Stage 6B val eval — shallow OWA ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "$S6B/latest.pt" --tokenizer "$TOK" \
  --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 4 --seed "$SEED" \
  --input-only \
  --json-out "artifacts/reports/matrix_e_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_e_shallow_s${SEED}.log"

echo "=== Stage 6B val eval — deep/CWA ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "$S6B/latest.pt" --tokenizer "$TOK" \
  --dataset artifacts/datasets/mixed/pater_mixed_deep.jsonl --split val \
  --device cuda --dtype fp32 --batch-size 4 --seed "$SEED" \
  --input-only \
  --json-out "artifacts/reports/matrix_e_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_e_deep_s${SEED}.log"

echo "DONE seed=${SEED}"
