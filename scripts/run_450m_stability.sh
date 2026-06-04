#!/usr/bin/env bash
# 450M stability gate: same 450M config / 25/75 mix / extended Qwen / ctx 1024,
# input-only, balanced sampler, set-valued (any-occurrence) span objective kept.
# Longer schedule (default 1200 steps) with warmup+cosine and a slightly lower
# peak LR to reduce seed variance and let support/idk/verifier recover.
#
# Evaluates BOTH the val split (headline) and the train split (for train/val gap).
# Usage: run_450m_stability.sh <seed> <gpu> [steps] [peak_lr] [warmup]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1200}"; LR="${4:-8e-5}"; WARMUP="${5:-100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_25_75.jsonl
OUT=artifacts/checkpoints/qwen_450m_stab_s${SEED}
EVAL_VAL=artifacts/reports/qwen_450m_stab_eval_s${SEED}.json
EVAL_TRAIN=artifacts/reports/qwen_450m_stab_traineval_s${SEED}.json

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN stability seed=${SEED} gpu=${GPU} steps=${STEPS} lr=${LR} warmup=${WARMUP} (set-valued span kept) ==="
python3 -u scripts/train_lm_aux.py \
  --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr "${LR}" --warmup-steps "${WARMUP}" \
  --weight-decay 0.01 --grad-clip 1.0 --seed "${SEED}" \
  --input-only --balanced-sampler \
  --tag "450m_stab_s${SEED}" --log-interval 50
echo "TRAIN_EXIT=$?"

echo "=== EVAL stability seed=${SEED} (val split, input-only) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL_VAL}"
echo "EVAL_VAL_EXIT=$?"

echo "=== EVAL stability seed=${SEED} (train split, for train/val gap) ==="
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split train --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" \
  --input-only --json-out "${EVAL_TRAIN}"
echo "EVAL_TRAIN_EXIT=$?"
echo "DONE seed=${SEED}"
