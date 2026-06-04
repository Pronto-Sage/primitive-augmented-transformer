#!/usr/bin/env bash
# D-interface SFT — make Condition D (Stage 5B) usable without touching the
# architecture. Conservative: top-2 backbone layers + top-6 FFN adapters, NO
# vocab-pressure (it globally biases logits and blows the KL guard), response-
# masked LM loss, KL-guarded against base D.
#
# Locked recipe (selected by calibration A–F, see docs/results/d_interface_sft_gate.md):
#   backbone-layers 2 (lr 1e-5)   adapters top-6 (lr 3e-5)   no pressure
#   aux-anchored replay (PAT-ER reason_big, frac 0.5) — holds primitive/r2p
#   500 steps   bs 8   warmup 60   KL guard 0.6
#
# Calibration findings encoded here:
#   - vocab-pressure heads globally bias logits → blow the KL guard; --no-pressure.
#   - adapters-only cannot learn generation; top-2 backbone layers are needed.
#   - leading with the <tool_call> special token is unreachable with a frozen tied
#     LM head → the corpus uses a short NL preamble (build_interface_sft.py).
#   - retraining the top-2 layers undoes Stage 5B's PAT-ER specialization, so the
#     replay stream runs aux losses (primitive/r2p/…@1.0) to anchor the side-state.
#
# Usage: bash scripts/run_d_interface_sft.sh <seed> <gpu>
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TOK=artifacts/tokenizers/qwen_pater_extended
BASE="artifacts/checkpoints/qwen_ws_stage5b_s${SEED}/latest.pt"
OUT="artifacts/checkpoints/qwen_ws_d_iface_s${SEED}"

export CUDA_VISIBLE_DEVICES="${GPU}"
python3 -u scripts/train_interface_sft.py \
  --checkpoint "$BASE" --tokenizer "$TOK" \
  --train artifacts/datasets/interface/train.jsonl \
  --val   artifacts/datasets/interface/val.jsonl \
  --replay artifacts/datasets/mixed/pater_mixed_reason_big.jsonl --replay-frac 0.5 \
  --device cuda --dtype fp32 \
  --backbone-layers 2 --backbone-lr 1e-5 --adapter-layers 6 --no-pressure \
  --lr 3e-5 --steps 500 --batch-size 8 --warmup-steps 60 \
  --log-interval 125 --kl-eval-interval 250 --kl-guard-threshold 0.6 \
  --seed "$SEED" --out-dir "$OUT" --tag "d_iface_s${SEED}"
echo "DONE d-interface SFT seed=${SEED} -> ${OUT}/latest.pt"
