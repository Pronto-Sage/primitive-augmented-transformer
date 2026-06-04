#!/usr/bin/env bash
# Resume CD pipeline from Stage 3 migrate step (Stage 3 training already complete).
# Use when stage3_s${SEED}/latest.pt is valid but stage3_ro is missing/corrupted.
# Usage: bash scripts/run_ws_pipeline_cd_resume.sh <seed> <gpu>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
DEEP=artifacts/datasets/mixed/pater_mixed_deep.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended

export CUDA_VISIBLE_DEVICES="${GPU}"
# Do NOT set expandable_segments here — only set it for training stages below.

echo "=== CD resume pipeline seed=${SEED} gpu=${GPU} (starting from migrate) ==="

# Migrate Stage 3 → register_only shape (stage3 training already done)
python3 scripts/migrate_stage3_register_only.py \
  --in-ckpt  "artifacts/checkpoints/qwen_ws_stage3_s${SEED}/latest.pt" \
  --out-ckpt "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
  --config   configs/pat_er_qwen3_warmstart.yaml

# Condition C eval (Stage3 register_only, frozen backbone)
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "${SEED}" --input-only \
  --json-out "artifacts/reports/matrix_c_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_c_shallow_s${SEED}.log"

python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DEEP" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "${SEED}" --input-only \
  --json-out "artifacts/reports/matrix_c_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_c_deep_s${SEED}.log"

# Stage 4 → 5A → 5B (set expandable_segments only during training, clear before evals)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_ws_stage4.sh  "${SEED}" "${GPU}" 900
bash scripts/run_ws_stage5a.sh "${SEED}" "${GPU}" 7 900
bash scripts/run_ws_stage5b.sh "${SEED}" "${GPU}" 900
unset PYTORCH_CUDA_ALLOC_CONF

# Condition D eval (Stage5B full warm-start PAT-ER)
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "${SEED}" --input-only \
  --json-out "artifacts/reports/matrix_d_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_d_shallow_s${SEED}.log"

python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DEEP" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "${SEED}" --input-only \
  --json-out "artifacts/reports/matrix_d_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_d_deep_s${SEED}.log"

echo "DONE CD resume seed=${SEED}"
