#!/usr/bin/env bash
# Train + eval one seed of the 450M PAT-ER gate on the 25/75 mixed dataset,
# extended Qwen tokenizer, ctx 1024, input-only, balanced sampler.
# Usage: run_450m_seed.sh <seed> <gpu> <steps>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-250}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_25_75.jsonl
OUT=artifacts/checkpoints/qwen_450m_s${SEED}
EVAL=artifacts/reports/qwen_450m_eval_s${SEED}.json

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN seed=${SEED} gpu=${GPU} steps=${STEPS} (450M, ext-qwen, ctx1024, input-only, balanced) ==="
python3 -u scripts/train_lm_aux.py \
  --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr 1e-4 --warmup-steps 40 --weight-decay 0.01 --seed "${SEED}" \
  --input-only --balanced-sampler --tag "450m_s${SEED}" --log-interval 25
echo "TRAIN_EXIT=$?"

echo "=== EVAL seed=${SEED} (val split, input-only) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL}"
echo "EVAL_EXIT=$?"
echo "DONE seed=${SEED}"
