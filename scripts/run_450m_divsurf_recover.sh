#!/usr/bin/env bash
# 450M diversified RECOVERY gate: identical data/config/tokenizer/recipe as
# run_450m_stability_divsurf.sh, only a longer schedule (default 1800 steps,
# warmup 140) to let the support/idk calibration heads finish converging on the
# harder diversified-text LM target. No loss-weight changes. Evals val + train.
# Usage: run_450m_divsurf_recover.sh <seed> <gpu> [steps] [peak_lr] [warmup]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"; LR="${4:-8e-5}"; WARMUP="${5:-140}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_25_75_divsurf.jsonl
OUT=artifacts/checkpoints/qwen_450m_divrec_s${SEED}
EVAL_VAL=artifacts/reports/qwen_450m_divrec_eval_s${SEED}.json
EVAL_TRAIN=artifacts/reports/qwen_450m_divrec_traineval_s${SEED}.json

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN divrec seed=${SEED} gpu=${GPU} steps=${STEPS} lr=${LR} warmup=${WARMUP} ==="
python3 -u scripts/train_lm_aux.py \
  --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr "${LR}" --warmup-steps "${WARMUP}" \
  --weight-decay 0.01 --grad-clip 1.0 --seed "${SEED}" \
  --input-only --balanced-sampler \
  --tag "450m_divrec_s${SEED}" --log-interval 50
echo "TRAIN_EXIT=$?"

echo "=== EVAL divrec seed=${SEED} (val split) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL_VAL}"
echo "EVAL_VAL_EXIT=$?"

echo "=== EVAL divrec seed=${SEED} (train split, for gap) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split train --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL_TRAIN}"
echo "EVAL_TRAIN_EXIT=$?"
echo "DONE seed=${SEED}"
