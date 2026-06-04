#!/usr/bin/env bash
# 450M external-coverage gate: frozen diversified+support recipe on the 3-way mix
# (real external 25% / derived external-style augmentation 25% / synthetic 50%).
# Real external records are untouched; derived augmentation is the synthetic_ext
# origin bucket. Same recipe: 1800 steps, warmup 140, peak LR 8e-5, fp32,
# input-only, balanced sampler. Evals val + train. Usage: <seed> <gpu> [steps] [lr] [warmup]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"; LR="${4:-8e-5}"; WARMUP="${5:-140}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_extaug.jsonl
OUT=artifacts/checkpoints/qwen_450m_extaug_s${SEED}
EVAL_VAL=artifacts/reports/qwen_450m_extaug_eval_s${SEED}.json
EVAL_TRAIN=artifacts/reports/qwen_450m_extaug_traineval_s${SEED}.json
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== TRAIN extaug seed=${SEED} gpu=${GPU} steps=${STEPS} lr=${LR} warmup=${WARMUP} ==="
python3 -u scripts/train_lm_aux.py --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr "${LR}" --warmup-steps "${WARMUP}" \
  --weight-decay 0.01 --grad-clip 1.0 --seed "${SEED}" --input-only --balanced-sampler \
  --tag "450m_extaug_s${SEED}" --log-interval 50
echo "TRAIN_EXIT=$?"
echo "=== EVAL extaug seed=${SEED} (val) ==="
python3 -u scripts/eval_lm_aux.py --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" --input-only --json-out "${EVAL_VAL}"
echo "EVAL_VAL_EXIT=$?"
echo "=== EVAL extaug seed=${SEED} (train) ==="
python3 -u scripts/eval_lm_aux.py --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split train --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" --input-only --json-out "${EVAL_TRAIN}"
echo "EVAL_TRAIN_EXIT=$?"
echo "DONE seed=${SEED}"
