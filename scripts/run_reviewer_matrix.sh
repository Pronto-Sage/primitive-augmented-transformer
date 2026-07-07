#!/usr/bin/env bash
# Reviewer-targeted matrix hardening:
#   1. train missing Condition B checkpoints (token-state baseline);
#   2. train missing Condition G checkpoints (generic registers);
#   3. re-evaluate B/G/C/D checkpoints with the updated contradiction
#      precision/recall/F1 JSON reporting;
#   4. aggregate B/G/C/D statistics.
#
# Assumes the prior C/D checkpoints already exist:
#   C: artifacts/checkpoints/qwen_ws_stage3_ro_s{seed}/latest.pt
#   D: artifacts/checkpoints/qwen_ws_stage5b_s{seed}/latest.pt
#
# Usage:
#   bash scripts/run_reviewer_matrix.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

mkdir -p artifacts/reports

DATA=artifacts/datasets/mixed/pater_mixed_reason_big.jsonl
DEEP=artifacts/datasets/mixed/pater_mixed_deep.jsonl
TOK=artifacts/tokenizers/qwen_pater_extended

missing_b=()
missing_g=()
missing_cd=()
for SEED in 0 1 2 3 4 5 6 7; do
  [[ -f "artifacts/checkpoints/qwen_ws_tokenstate_s${SEED}/latest.pt" ]] || missing_b+=("$SEED")
  [[ -f "artifacts/checkpoints/qwen_ws_genericreg_s${SEED}/latest.pt" ]] || missing_g+=("$SEED")
  [[ -f "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" ]] || missing_cd+=("C:s${SEED}")
  [[ -f "artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt" ]] || missing_cd+=("D:s${SEED}")
done

if (( ${#missing_cd[@]} > 0 )); then
  echo "ERROR: required C/D checkpoints are missing: ${missing_cd[*]}" >&2
  echo "Run the C/D warm-start pipeline first; this script will not substitute checkpoints." >&2
  exit 2
fi

if (( ${#missing_b[@]} > 0 )); then
  echo "=== Reviewer matrix: train missing B seeds: ${missing_b[*]} ==="
  for SEED in "${missing_b[@]}"; do
    GPU="$SEED"
    (
      bash scripts/run_matrix_b.sh "$SEED" "$GPU"
    ) > "artifacts/reports/run_matrix_b_s${SEED}.outer.log" 2>&1 &
  done
  wait
else
  echo "=== Condition B checkpoints present; skipping B training ==="
fi

if (( ${#missing_g[@]} > 0 )); then
  echo "=== Reviewer matrix: train missing G seeds: ${missing_g[*]} ==="
  for SEED in "${missing_g[@]}"; do
    GPU="$SEED"
    (
      bash scripts/run_matrix_g.sh "$SEED" "$GPU"
    ) > "artifacts/reports/run_matrix_g_s${SEED}.outer.log" 2>&1 &
  done
  wait
else
  echo "=== Condition G checkpoints present; skipping G training ==="
fi

echo "=== Verify B/G/C/D checkpoints before eval ==="
missing_all=()
for SEED in 0 1 2 3 4 5 6 7; do
  [[ -f "artifacts/checkpoints/qwen_ws_tokenstate_s${SEED}/latest.pt" ]] || missing_all+=("B:s${SEED}")
  [[ -f "artifacts/checkpoints/qwen_ws_genericreg_s${SEED}/latest.pt" ]] || missing_all+=("G:s${SEED}")
  [[ -f "artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" ]] || missing_all+=("C:s${SEED}")
  [[ -f "artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt" ]] || missing_all+=("D:s${SEED}")
done
if (( ${#missing_all[@]} > 0 )); then
  echo "ERROR: still missing checkpoints after training: ${missing_all[*]}" >&2
  exit 3
fi

echo "=== Re-evaluate B/G/C/D checkpoints with PRF metrics ==="
for SEED in 0 1 2 3 4 5 6 7; do
  GPU="$SEED"
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    for SPEC in \
      "b artifacts/checkpoints/qwen_ws_tokenstate_s${SEED}/latest.pt" \
      "g artifacts/checkpoints/qwen_ws_genericreg_s${SEED}/latest.pt" \
      "c artifacts/checkpoints/qwen_ws_stage3_ro_s${SEED}/latest.pt" \
      "d artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt"; do
      set -- $SPEC
      TAG="$1"; CKPT="$2"
      python3 -u scripts/eval_lm_aux.py \
        --checkpoint "$CKPT" --tokenizer "$TOK" \
        --dataset "$DATA" --split val \
        --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
        --json-out "artifacts/reports/matrix_${TAG}_shallow_s${SEED}.json" \
        2>&1 | tee "artifacts/reports/matrix_${TAG}_shallow_s${SEED}.log"

      python3 -u scripts/eval_lm_aux.py \
        --checkpoint "$CKPT" --tokenizer "$TOK" \
        --dataset "$DEEP" --split val \
        --device cuda --dtype fp32 --batch-size 2 --seed "$SEED" --input-only \
        --json-out "artifacts/reports/matrix_${TAG}_deep_s${SEED}.json" \
        2>&1 | tee "artifacts/reports/matrix_${TAG}_deep_s${SEED}.log"
    done
  ) > "artifacts/reports/reeval_bcd_s${SEED}.outer.log" 2>&1 &
done
wait

echo "=== Aggregate matrix ==="
python3 scripts/compute_matrix_stats.py \
  --seeds 0-7 \
  --output artifacts/reports/reviewer_matrix_stats.json \
  2>&1 | tee artifacts/reports/reviewer_matrix_stats.log

python3 scripts/report_contradiction_prf.py \
  --seeds 0-7 \
  --output-md artifacts/reports/contradiction_prf.md \
  --output-json artifacts/reports/contradiction_prf.json

echo "DONE reviewer matrix"
