# Reproducibility

This package keeps code, configs, fixtures, manuscript source, and compact result summaries in git. Large checkpoints, converted datasets, tokenizer caches, and raw training artifacts are intentionally excluded.

## External Data

The experiments use upstream datasets through conversion scripts:

- ProofWriter V2020.12.3
- FOLIO
- SNLI/e-SNLI/ANLI-style NLI fixtures for converter validation and rejected-variant evaluation

Converted external records are not redistributed. Reproduce them by obtaining the upstream data from its provider, then running the corresponding converter:

```bash
python scripts/convert_proofwriter.py --input <proofwriter_dir> --output artifacts/datasets/external/proofwriter_pater.jsonl
python scripts/convert_folio.py --input <folio_dir> --output artifacts/datasets/external/folio_pater.jsonl
python scripts/convert_nli.py --input <nli_jsonl> --output artifacts/datasets/external/nli_pater.jsonl
```

Validate converted records:

```bash
python scripts/validate_pater_dataset.py artifacts/datasets/external/*.jsonl
python scripts/validate_external_conversion.py artifacts/datasets/external/*.jsonl
```

## Synthetic And Mixed Data

```bash
python scripts/build_synthetic_pater_dataset.py \
  --output-dir artifacts/datasets/pater_synthetic \
  --diversify-surfaces \
  --support-coverage

python scripts/mix_pater_datasets.py \
  --external artifacts/datasets/external/*.jsonl \
  --synthetic artifacts/datasets/pater_synthetic/*.jsonl \
  --output artifacts/datasets/mixed/pater_mixed.jsonl \
  --external-fraction 0.25 \
  --balance-primitive
```

## Warm-Start Path

The Qwen3 warm-start path requires an upstream `Qwen/Qwen3-0.6B` checkpoint and the extended PAT-ER tokenizer.

```bash
python scripts/prepare_hf_tokenizer.py \
  --base Qwen/Qwen3-0.6B \
  --output artifacts/tokenizers/qwen_pater_extended

python scripts/validate_warmstart_real.py \
  --config configs/pat_er_qwen3_warmstart.yaml \
  --tokenizer artifacts/tokenizers/qwen_pater_extended \
  --qwen Qwen/Qwen3-0.6B
```

The staged training drivers are:

```bash
bash scripts/run_ws_pipeline_cd_full.sh <seed> <gpu>
python scripts/compute_matrix_stats.py --seeds 0-7
```

## Interface Evaluation

```bash
python scripts/build_interface_sft.py --out-dir artifacts/datasets/interface
bash scripts/run_d_decoupled_sft.sh <seed> <gpu>
python scripts/build_product_eval.py --output artifacts/datasets/product_eval_big.jsonl
python scripts/eval_product.py --checkpoint <checkpoint> --prompts artifacts/datasets/product_eval_big.jsonl
```

Compact published result summaries are in `results/`.

