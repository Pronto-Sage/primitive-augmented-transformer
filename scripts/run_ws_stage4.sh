#!/usr/bin/env bash
# Phase 2 Stage 4 — controlled injection opening.
# Loads the Stage-3 trained checkpoint (frozen backbone + trained registers/aux),
# then opens PAT-ER injection entry paths in three sub-stages at inject_lr=1e-6:
#
#   4A: role_fuse + primitive_fuse  (token←register injection)
#   4B: + FFN adapter up-projections
#   4C: + vocab-pressure proj.1
#
# Each sub-stage trains for STEPS steps and saves its checkpoint. The final
# report_stage4.py measures LM delta, KL vs backbone, injection norms, and
# generation sanity on real text.
#
# Usage: bash scripts/run_ws_stage4.sh <seed> <gpu> [steps_per_stage=900]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-900}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml

# Use the register_only-migrated Stage-3 checkpoint as the starting point.
# The migration preserves all trained params (registers, cross-attn, aux heads)
# and reshapes role_fuse/primitive_fuse from [h,h*3]→[h,h] (zero either way).
S3=artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}
S4A=artifacts/checkpoints/qwen_ws_stage4a_ro_s${SEED}
S4B=artifacts/checkpoints/qwen_ws_stage4b_ro_s${SEED}
S4C=artifacts/checkpoints/qwen_ws_stage4c_ro_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage 4 loss recipe: LM-only (all aux=0) so that role_fuse/primitive_fuse
# receive ONLY the LM gradient through the injection path.  With aux losses
# present, the aux gradient flows through z_event→role_fuse and drives the
# injection toward primitive/role representation, corrupting the backbone even
# with detached z_event (the LM loss still moves role_fuse at inject_lr pace).
# With LM-only + detach: injection learns "does this register content help
# predict the next token?" — the correct signal for a warm-start modulator.
# Aux metrics are already proven at Stage 3; they are not re-evaluated here.
LW=(--loss-weight \
  event_arg=0 argument_start=0 argument_end=0 argument_span=0 \
  event_token=0 evidence_pointer=0 \
  primitive=0 support=0 tool_intent=0 schema=0 idk=0 \
  role_to_primitive=0 role_ambiguity=0 verifier=0 predicate_event=0 \
  proto_role=0 arg_role=0 entailment_state=0 proof_depth=0 rule_chain_length=0)

COMMON=(--config "$CFG" --tokenizer "$TOK" --freeze-backbone --injection-only
        --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768
        --batch-size 2 --lr 8e-5 --inject-lr 1e-8
        --weight-decay 0.01 --grad-clip 1.0
        --seed "$SEED" --input-only --balanced-sampler --use-reasoning-heads
        --log-interval 100 --warmup-steps 90 "${LW[@]}")

echo "=== Stage 4A: open role_fuse + primitive_fuse  (${STEPS} steps) ==="
python3 -u scripts/train_lm_aux.py "${COMMON[@]}" \
  --resume-from "$S3/latest.pt" \
  --inject-paths role_fuse,primitive_fuse \
  --steps "$STEPS" --out-dir "$S4A" --tag "ws_stage4a_s${SEED}"

echo "=== Stage 4B: + adapter up-projections  (${STEPS} steps) ==="
python3 -u scripts/train_lm_aux.py "${COMMON[@]}" \
  --resume-from "$S4A/latest.pt" \
  --inject-paths role_fuse,primitive_fuse,adapter_up \
  --steps "$STEPS" --out-dir "$S4B" --tag "ws_stage4b_s${SEED}"

echo "=== Stage 4C: + vocab-pressure proj.1  (${STEPS} steps) ==="
python3 -u scripts/train_lm_aux.py "${COMMON[@]}" \
  --resume-from "$S4B/latest.pt" \
  --inject-paths role_fuse,primitive_fuse,adapter_up,pressure_proj1 \
  --steps "$STEPS" --out-dir "$S4C" --tag "ws_stage4c_s${SEED}"

# Injection report skipped for multi-seed hardening runs — loads 4 checkpoints
# simultaneously and exhausts allocator after Stage 4 training. Seed 0 injection
# report is already documented in docs/results/warmstart_stage4_gate.md.

# Intermediate val eval skipped — matrix C/D evals run at the pipeline level.
echo "DONE seed=${SEED}"
