#!/usr/bin/env bash
# Phase 1.1 gate: reasoning heads off, same reasoning-labeled mix, frozen recipe.
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA=artifacts/datasets/mixed/pater_mixed_reason.jsonl
OUT=artifacts/checkpoints/qwen_450m_reasonoff_s${SEED}
export CUDA_VISIBLE_DEVICES="${GPU}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u scripts/train_lm_aux.py --config configs/pat_er_450m_qwen.yaml --tokenizer artifacts/tokenizers/qwen_pater_extended   --dataset "$DATA" --out-dir "$OUT" --device cuda --dtype fp32 --max-seq-len 768   --steps 1800 --batch-size 4 --lr 8e-5 --warmup-steps 140 --weight-decay 0.01 --grad-clip 1.0   --seed "$SEED" --input-only --balanced-sampler  --tag "450m_reasonoff_s${SEED}" --log-interval 100
echo "TRAIN_EXIT=$?"
python3 -u scripts/eval_lm_aux.py --checkpoint "$OUT/latest.pt" --tokenizer artifacts/tokenizers/qwen_pater_extended   --dataset "$DATA" --split val --device cuda --dtype fp32 --batch-size 4 --seed "$SEED" --input-only   --json-out "artifacts/reports/qwen_450m_reasonoff_eval_s${SEED}.json"
echo "EVAL_EXIT=$?  DONE seed=${SEED}"
