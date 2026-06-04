#!/usr/bin/env bash
# 450M stability gate on the surface-DIVERSIFIED 25/75 mix. Identical recipe to
# run_450m_stability.sh (1200 steps, warmup+cosine, peak LR 8e-5, fp32, set-valued
# span objective, input-only, balanced sampler); only the synthetic argument
# surfaces differ (natural, context-bearing gold spans). Evals val + train splits.
# Usage: run_450m_stability_divsurf.sh <seed> <gpu> [steps] [peak_lr] [warmup]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1200}"; LR="${4:-8e-5}"; WARMUP="${5:-100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_25_75_divsurf.jsonl
OUT=artifacts/checkpoints/qwen_450m_divsurf_s${SEED}
EVAL_VAL=artifacts/reports/qwen_450m_divsurf_eval_s${SEED}.json
EVAL_TRAIN=artifacts/reports/qwen_450m_divsurf_traineval_s${SEED}.json

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN divsurf seed=${SEED} gpu=${GPU} steps=${STEPS} lr=${LR} warmup=${WARMUP} ==="
python3 -u scripts/train_lm_aux.py \
  --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr "${LR}" --warmup-steps "${WARMUP}" \
  --weight-decay 0.01 --grad-clip 1.0 --seed "${SEED}" \
  --input-only --balanced-sampler \
  --tag "450m_divsurf_s${SEED}" --log-interval 50
echo "TRAIN_EXIT=$?"

echo "=== EVAL divsurf seed=${SEED} (val split) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL_VAL}"
echo "EVAL_VAL_EXIT=$?"

echo "=== EVAL divsurf seed=${SEED} (train split, for gap) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split train --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL_TRAIN}"
echo "EVAL_TRAIN_EXIT=$?"
echo "DONE seed=${SEED}"
