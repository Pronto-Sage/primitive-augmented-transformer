#!/usr/bin/env bash
# Phase 2 Stage 5B — top-2 Qwen layer unfreeze with KL guard.
#
# Source: Stage-5A checkpoint (Stage-3 registers, Stage-4 fuse tensors,
# Stage-5A upper-layer adapters).  Unfreezes the TOP 2 Qwen decoder layers
# (layers 26-27) at backbone_lr=2e-7 while keeping all other layers frozen.
# PAT-ER adapters (layers 21-27) continue at adapter_lr=1e-6.
# Injection fuse tensors stay frozen at Stage-4 values.
#
# KL guard: evaluates LM delta and KL(full||clean) on 3 fixed probe sentences
# every --kl-eval-interval steps; stops early if either exceeds threshold.
#
# Rules (from warmstart_modulation_rule.md):
#   - Backbone layers train at backbone_lr << adapter_lr << main_lr
#   - injection-only mode: only backbone + adapter params train
#   - LM-only loss to prevent register/aux drift
#   - KL guard prevents pretrained LM collapse
#
# Usage: bash scripts/run_ws_stage5b.sh <seed> <gpu> [steps=900]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-900}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml

S5A=artifacts/checkpoints/qwen_ws_stage5a_s${SEED}
S5B=artifacts/checkpoints/qwen_ws_stage5b_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# LM-only: prevents PAT-ER-specific drift in backbone parameters.
LW=(--loss-weight \
  event_arg=0 argument_start=0 argument_end=0 argument_span=0 \
  event_token=0 evidence_pointer=0 \
  primitive=0 support=0 tool_intent=0 schema=0 idk=0 \
  role_to_primitive=0 role_ambiguity=0 verifier=0 predicate_event=0 \
  proto_role=0 arg_role=0 entailment_state=0 proof_depth=0 rule_chain_length=0)

echo "=== Stage 5B: top-2 Qwen layers (26-27), ${STEPS} steps ==="
python3 -u scripts/train_lm_aux.py \
  --config "$CFG" --tokenizer "$TOK" \
  --freeze-backbone --injection-only --use-reasoning-heads \
  --resume-from "$S5A/latest.pt" \
  --upper-adapter-layers 7 --adapter-lr 1e-6 \
  --backbone-layers 2  --backbone-lr 2e-7 \
  --kl-guard-threshold 0.5  --lm-guard-threshold 0.4 \
  --kl-eval-interval 50 \
  --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768 \
  --batch-size 2 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler \
  --log-interval 100 --warmup-steps 90 --steps "$STEPS" \
  --out-dir "$S5B" --tag "ws_stage5b_s${SEED}" \
  "${LW[@]}"

echo "=== Stage 5B report ==="
CUDA_VISIBLE_DEVICES="${GPU}" python3 -u scripts/report_stage4.py \
  --stage3  artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt \
  --stage4c artifacts/checkpoints/qwen_ws_stage4c_ro_s${SEED}/latest.pt \
  --stage5a "$S5A/latest.pt" \
  --stage5b "$S5B/latest.pt" \
  --tokenizer "$TOK" \
  --device cuda \
  --json-out "artifacts/reports/qwen_ws_stage5b_s${SEED}.json"

# Intermediate val eval skipped — matrix D eval runs at the pipeline level.

echo "DONE seed=${SEED}"
