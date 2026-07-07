#!/usr/bin/env bash
# Condition G: Qwen3-0.6B + generic learned registers.
#
# This is the reviewer-control baseline between:
#   B = token-pooled aux labels, no side-state;
#   C = typed PAT-ER event-role -> primitive registers.
#
# G keeps the same learned register count and the same cross-attention/fuse
# modules as PAT-ER, but updates all registers as one homogeneous latent bank.
# The bank is split only for the existing aux-head shapes. If C beats G, the
# typed PAT-ER flow contributes beyond "any learned register memory".
#
# Usage: bash scripts/run_matrix_g.sh <seed> <gpu>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
DEEP=artifacts/datasets/mixed/pater_mixed_deep.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended
CFG=configs/pat_er_qwen3_warmstart.yaml
OUT=artifacts/checkpoints/qwen_ws_genericreg_s${SEED}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Condition G seed=${SEED} gpu=${GPU}: generic learned registers ==="
python3 -u scripts/train_lm_aux.py \
  --config "$CFG" --tokenizer "$TOK" \
  --warmstart-qwen auto --freeze-backbone \
  --generic-register-stream \
  --dataset "$DATA" --device cuda --dtype fp32 --max-seq-len 768 \
  --batch-size 4 --lr 8e-5 --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler --use-reasoning-heads \
  --log-interval 200 --warmup-steps 140 --steps 1800 \
  --out-dir "$OUT" --tag "ws_genericreg_s${SEED}"

echo "=== Condition G eval — shallow OWA ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "$OUT/latest.pt" --tokenizer "$TOK" \
  --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_g_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_g_shallow_s${SEED}.log"

echo "=== Condition G eval — deep/CWA ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "$OUT/latest.pt" --tokenizer "$TOK" \
  --dataset "$DEEP" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_g_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_g_deep_s${SEED}.log"

echo "DONE G seed=${SEED}"
