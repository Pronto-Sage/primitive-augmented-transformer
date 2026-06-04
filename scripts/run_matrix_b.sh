#!/usr/bin/env bash
# Condition B: Qwen3-0.6B + aux-from-token-state (no PAT-ER registers).
# Warm-start the backbone, train aux heads with token-pooled state only.
# Used for contribution matrix statistical hardening (3 seeds).
#
# Usage: bash scripts/run_matrix_b.sh <seed> <gpu>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml
OUT=artifacts/checkpoints/qwen_ws_tokenstate_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Condition B seed=${SEED} gpu=${GPU}: aux-from-token-state ==="
python3 -u scripts/train_lm_aux.py \
  --config "$CFG" --tokenizer "$TOK" \
  --warmstart-qwen auto --freeze-backbone \
  --aux-from-token-state \
  --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768 \
  --batch-size 4 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler --use-reasoning-heads \
  --log-interval 200 --warmup-steps 140 --steps 1800 \
  --out-dir "$OUT" --tag "ws_tokenstate_s${SEED}"

echo "=== Condition B eval — shallow OWA ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "$OUT/latest.pt" --tokenizer "$TOK" \
  --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_b_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_b_shallow_s${SEED}.log"

echo "=== Condition B eval — deep/CWA ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "$OUT/latest.pt" --tokenizer "$TOK" \
  --dataset artifacts/datasets/mixed/pater_mixed_deep.jsonl --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_b_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_b_deep_s${SEED}.log"

echo "DONE B seed=${SEED}"
