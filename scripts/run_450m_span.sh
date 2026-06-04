#!/usr/bin/env bash
# Span-END repair gate: 450M, 25/75 mix, extended Qwen, ctx 1024, input-only,
# balanced sampler + span-focused sampler + boundary-aware joint span loss.
# Usage: run_450m_span.sh <seed> <gpu> <steps>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-400}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_25_75.jsonl
OUT=artifacts/checkpoints/qwen_450m_span_s${SEED}
EVAL=artifacts/reports/qwen_450m_span_eval_s${SEED}.json

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN span-repair seed=${SEED} gpu=${GPU} steps=${STEPS} (joint span loss + span sampler) ==="
python3 -u scripts/train_lm_aux.py \
  --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr 1e-4 --warmup-steps 40 --weight-decay 0.01 --seed "${SEED}" \
  --input-only --balanced-sampler --span-sampler --span-sampler-alpha 1.0 \
  --tag "450m_span_s${SEED}" --log-interval 25
echo "TRAIN_EXIT=$?"

echo "=== EVAL span-repair seed=${SEED} (val split, input-only) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL}"
echo "EVAL_EXIT=$?"
echo "DONE seed=${SEED}"
