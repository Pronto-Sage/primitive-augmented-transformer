#!/usr/bin/env bash
# 450M real-external candidate gate (step 6): frozen divsup recipe on the
# real-data candidate mix (existing external + ProofWriter meta-abduction
# extension + divsup synthetic; NO synthetic_ext). Recipe unchanged: 1800 steps,
# warmup 140, peak LR 8e-5, fp32, input-only, balanced sampler, set-valued span.
# Usage: <seed> <gpu> [steps] [lr] [warmup]
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; STEPS="${3:-1800}"; LR="${4:-8e-5}"; WARMUP="${5:-140}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=configs/pat_er_450m_qwen.yaml
TOK=artifacts/tokenizers/qwen_pater_extended
DATA=artifacts/datasets/mixed/pater_mixed_realext_b.jsonl
OUT=artifacts/checkpoints/qwen_450m_realextb_s${SEED}
EVAL_VAL=artifacts/reports/qwen_450m_realextb_eval_s${SEED}.json
EVAL_TEST=artifacts/reports/qwen_450m_realextb_test_s${SEED}.json
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== TRAIN realext seed=${SEED} gpu=${GPU} steps=${STEPS} ==="
python3 -u scripts/train_lm_aux.py --config "${CFG}" --tokenizer "${TOK}" --dataset "${DATA}" \
  --out-dir "${OUT}" --device cuda --dtype fp32 --max-seq-len 768 \
  --steps "${STEPS}" --batch-size 4 --lr "${LR}" --warmup-steps "${WARMUP}" \
  --weight-decay 0.01 --grad-clip 1.0 --seed "${SEED}" --input-only --balanced-sampler \
  --tag "450m_realextb_s${SEED}" --log-interval 50
echo "TRAIN_EXIT=$?"
echo "=== EVAL realext seed=${SEED} (val) ==="
python3 -u scripts/eval_lm_aux.py --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split val --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" --input-only --json-out "${EVAL_VAL}"
echo "EVAL_VAL_EXIT=$?"
echo "=== EVAL realext seed=${SEED} (test) ==="
python3 -u scripts/eval_lm_aux.py --checkpoint "${OUT}/latest.pt" --tokenizer "${TOK}" --dataset "${DATA}" \
  --split test --device cuda --dtype fp32 --batch-size 4 --seed "${SEED}" --input-only --json-out "${EVAL_TEST}"
echo "EVAL_TEST_EXIT=$?"
echo "DONE seed=${SEED}"
