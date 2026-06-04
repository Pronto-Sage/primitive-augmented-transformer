#!/usr/bin/env bash
# Stage 3 training only — no intermediate evals.
# Used for multi-seed matrix hardening where only the checkpoint matters.
# The C-condition eval is run separately by the pipeline.
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml
OUT=artifacts/checkpoints/qwen_ws_stage3_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
# expandable_segments only for training — clear it before any eval to prevent
# the allocator from caching 40+ GB across a long eval loop.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 3 train-only seed=${SEED} gpu=${GPU} steps=${STEPS} ==="
python3 -u scripts/train_lm_aux.py \
  --config "$CFG" --tokenizer "$TOK" --warmstart-qwen auto --freeze-backbone \
  --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768 \
  --batch-size 2 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler --use-reasoning-heads \
  --log-interval 100 --warmup-steps 140 --steps "$STEPS" \
  --out-dir "$OUT" --tag "ws_stage3_s${SEED}" \
  --loss-weight event_arg=0 argument_start=0 argument_end=0 argument_span=0 \
                event_token=0 evidence_pointer=0 \
                primitive=0.01 support=0.01 tool_intent=0.01 schema=0.01 \
                idk=0.01 role_to_primitive=0.01 role_ambiguity=0.01 \
                verifier=0.005 predicate_event=0.01 \
                proto_role=0.01 arg_role=0.01 \
                entailment_state=0.01 proof_depth=0.005 rule_chain_length=0.005
echo "DONE Stage3-train seed=${SEED}"
