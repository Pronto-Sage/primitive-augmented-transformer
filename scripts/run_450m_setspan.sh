#!/usr/bin/env bash
# Set-valued span target gate: 450M, 25/75 mix, extended Qwen, ctx 1024, input-only,
# balanced sampler. Set-valued (multi-positive) argument span targets credit any
# co-referent occurrence; boundary-aware joint span loss kept; span sampler OFF.
# Usage: run_450m_setspan.sh <seed> <gpu> <steps>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-400}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_25_75.jsonl
OUT=artifacts/checkpoints/qwen_450m_set_s${SEED}
EVAL=artifacts/reports/qwen_450m_set_eval_s${SEED}.json

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN set-valued span seed=${SEED} gpu=${GPU} steps=${STEPS} (multi-positive span targets) ==="
python3 -u scripts/train_lm_aux.py \
  --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr 1e-4 --warmup-steps 40 --weight-decay 0.01 --seed "${SEED}" \
  --input-only --balanced-sampler \
  --tag "450m_set_s${SEED}" --log-interval 25
echo "TRAIN_EXIT=$?"

echo "=== EVAL set-valued span seed=${SEED} (val split, input-only) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL}"
echo "EVAL_EXIT=$?"
echo "DONE seed=${SEED}"
