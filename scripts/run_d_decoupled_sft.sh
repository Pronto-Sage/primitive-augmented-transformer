#!/usr/bin/env bash
# Decoupled interface SFT via mode-gated top-K layers (option 3) — makes Condition D
# usable with ZERO side-state cost. Trains COPIES of the top-2 decoder blocks used
# only in interface_mode; base mode runs the original frozen blocks, so the aux heads
# read exactly the Condition-D representation (r2p regression = 0, asserted at train
# time via base_drift==0). Interface mode gets adapted attention → tool-name copying.
#
# Locked recipe: adapt-layers 2, lr 5e-5, 1000 steps, bs 8, warmup 100.
# Base mode is provably protected (base_drift==0), so interface mode is trained hard
# for robust tool-name copying across seeds at zero side-state cost.
# Corpus (build_interface_sft.py): tool preamble + dual phrasings + varied JSON keys.
#
# Usage: bash scripts/run_d_decoupled_sft.sh <seed> <gpu>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TOK=artifacts/tokenizers/qwen_pater_extended
BASE="artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt"
OUT="artifacts/checkpoints/qwen_ws_d_decoupled_s${SEED}"

export CUDA_VISIBLE_DEVICES="${GPU}"
python3 -u scripts/train_interface_decoupled_sft.py \
  --checkpoint "$BASE" --tokenizer "$TOK" \
  --train artifacts/datasets/interface/train.jsonl \
  --val   artifacts/datasets/interface/val.jsonl \
  --adapt-layers 2 --steps 1000 --batch-size 8 --warmup-steps 100 --lr 5e-5 \
  --device cuda --dtype fp32 --log-interval 225 \
  --seed "$SEED" --out-dir "$OUT" --tag "d_decoupled_s${SEED}"
echo "DONE decoupled interface SFT seed=${SEED} -> ${OUT}/latest.pt"
