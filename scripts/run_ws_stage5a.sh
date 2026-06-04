#!/usr/bin/env bash
# Phase 2 Stage 5A — upper-layer PAT-ER adapter opening.
#
# Source: Stage-4C checkpoint (backbone frozen, Stage-3 registers, Stage-4 fuse).
# Opens FFN adapter (up+down) weights in the top UPPER_LAYERS (default 7) layers.
# Backbone (self-attn, base FFN, norms), Stage-4 fuse tensors, Stage-3 registers,
# cross-attn, aux heads, gate, and lower-layer adapters all remain frozen.
#
# Zero-init on up.weight provides natural gradient isolation (no detach needed):
#   adapter(x) = up(tanh(down(x)))
#   d(loss)/d(down) = up.weight^T * upstream = 0 when up.weight=0
#   d(loss)/d(up)   = upstream * tanh(down(x))^T  (small from LM at pretrained optimum)
#
# As up.weight grows from zero at adapter_lr, the adapter activates gradually.
# Backbone residual stream is protected by the same injection-only + LM-only discipline
# as Stage 4 (docs/warmstart_modulation_rule.md).
#
# Usage: bash scripts/run_ws_stage5a.sh <seed> <gpu> [upper_layers=7] [steps=900]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; UPPER="${3:-7}"; STEPS="${4:-900}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml

# Source: Stage-4C (Stage-4 fuse tensors are frozen at Stage-4 trained values).
S4C=artifacts/checkpoints/qwen_ws_stage4c_ro_s${SEED}
S5A=artifacts/checkpoints/qwen_ws_stage5a_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# LM-only loss: all aux=0 to prevent adapter drift toward PAT-ER distribution.
LW=(--loss-weight \
  event_arg=0 argument_start=0 argument_end=0 argument_span=0 \
  event_token=0 evidence_pointer=0 \
  primitive=0 support=0 tool_intent=0 schema=0 idk=0 \
  role_to_primitive=0 role_ambiguity=0 verifier=0 predicate_event=0 \
  proto_role=0 arg_role=0 entailment_state=0 proof_depth=0 rule_chain_length=0)

echo "=== Stage 5A: upper-layer adapters (top ${UPPER} layers, ${STEPS} steps) ==="
python3 -u scripts/train_lm_aux.py \
  --config "$CFG" --tokenizer "$TOK" \
  --freeze-backbone --injection-only --use-reasoning-heads \
  --resume-from "$S4C/latest.pt" \
  --upper-adapter-layers "$UPPER" \
  --adapter-lr 1e-6 \
  --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768 \
  --batch-size 2 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler \
  --log-interval 100 --warmup-steps 90 --steps "$STEPS" \
  --out-dir "$S5A" --tag "ws_stage5a_s${SEED}" \
  "${LW[@]}"

echo "=== Stage 5A injection + adapter report ==="
CUDA_VISIBLE_DEVICES="${GPU}" python3 -u scripts/report_stage4.py \
  --stage3  artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt \
  --stage4c "$S4C/latest.pt" \
  --stage5a "$S5A/latest.pt" \
  --tokenizer "$TOK" \
  --device cuda \
  --json-out "artifacts/reports/qwen_ws_stage5a_s${SEED}.json"

# Intermediate val eval skipped — matrix D eval runs at the pipeline level.

echo "DONE seed=${SEED} upper_layers=${UPPER}"
