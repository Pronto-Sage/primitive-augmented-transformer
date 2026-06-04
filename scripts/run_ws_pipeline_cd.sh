#!/usr/bin/env bash
# Full warm-start pipeline producing Conditions C and D for the contribution matrix.
#
#   C = Stage 3 (PAT-ER registers on frozen backbone, no injection)
#   D = Stage 5B (full warm-start stack: Stage 3→4→5A→5B)
#
# Usage: bash scripts/run_ws_pipeline_cd.sh <seed> <gpu>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
DEEP=artifacts/datasets/mixed/pater_mixed_deep.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== CD pipeline seed=${SEED} gpu=${GPU} ==="

# ── Stage 3: freeze backbone, train registers + aux heads ─────────────────────
echo "--- Stage 3 ---"
bash scripts/run_ws_stage3.sh "$SEED" "$GPU" 1800

# ── Migrate Stage-3 checkpoint to register_only fuse shape ────────────────────
echo "--- migrate to register_only ---"
python3 scripts/migrate_stage3_register_only.py \
  --in-ckpt  "artifacts/checkpoints/qwen_ws_stage3_s${SEED}/latest.pt" \
  --out-ckpt "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
  --config   configs/pat_er_qwen3_warmstart.yaml

# ── Condition C eval (Stage 3 = PAT-ER registers, no injection) ───────────────
echo "--- Condition C eval ---"
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_c_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_c_shallow_s${SEED}.log"

python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DEEP" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_c_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_c_deep_s${SEED}.log"

# ── Stage 4A/B/C: controlled injection opening ────────────────────────────────
echo "--- Stage 4 ---"
bash scripts/run_ws_stage4.sh "$SEED" "$GPU" 900

# ── Stage 5A: upper-layer adapter opening ─────────────────────────────────────
echo "--- Stage 5A ---"
bash scripts/run_ws_stage5a.sh "$SEED" "$GPU" 7 900

# ── Stage 5B: top-2 Qwen layer unfreeze with KL guard ─────────────────────────
echo "--- Stage 5B ---"
bash scripts/run_ws_stage5b.sh "$SEED" "$GPU" 900

# ── Condition D eval (Stage 5B = full warm-start PAT-ER) ──────────────────────
echo "--- Condition D eval ---"
python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DATA" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_d_shallow_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_d_shallow_s${SEED}.log"

python3 -u scripts/eval_lm_aux.py \
  --checkpoint "artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt" \
  --tokenizer "$TOK" --dataset "$DEEP" --split val \
  --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
  --json-out "artifacts/reports/matrix_d_deep_s${SEED}.json" \
  2>&1 | tee "artifacts/reports/matrix_d_deep_s${SEED}.log"

echo "DONE CD pipeline seed=${SEED}"
