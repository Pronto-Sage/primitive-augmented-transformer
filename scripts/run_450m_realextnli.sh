#!/usr/bin/env bash
# 450M explicit-conflict gate: frozen recipe on repaired realext + SNLI external.
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"; LR="${4:-8e-5}"; WARMUP="${5:-140}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA=artifacts/datasets/mixed/pater_mixed_realext_nli.jsonl
OUT=artifacts/checkpoints/qwen_450m_realextnli_s${SEED}
export CUDA_VISIBLE_DEVICES="${GPU}"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== TRAIN realextnli seed=${SEED} gpu=${GPU} ==="
python3 -u scripts/train_lm_aux.py --config configs/pat_er_450m_qwen.yaml --tokenizer artifacts/tokenizers/qwen_pater_extended \
  --dataset "$DATA" --out-dir "$OUT" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "$STEPS" --batch-size 4 --lr "$LR" --warmup-steps "$WARMUP" --weight-decay 0.01 --grad-clip 1.0 \
  --seed "$SEED" --input-only --balanced-sampler --tag "450m_realextnli_s${SEED}" --log-interval 50
echo "TRAIN_EXIT=$?"
for SP in val test; do
  python3 -u scripts/eval_lm_aux.py --checkpoint "$OUT/latest.pt" --tokenizer artifacts/tokenizers/qwen_pater_extended \
    --dataset "$DATA" --split "$SP" --device cuda --dtype fp32 --batch-size 4 --seed "$SEED" --input-only \
    --json-out "artifacts/reports/qwen_450m_realextnli_${SP}_s${SEED}.json"
  echo "EVAL_${SP}_EXIT=$?"
done
echo "DONE seed=${SEED}"
