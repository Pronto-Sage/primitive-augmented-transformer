# Primitive-Augmented Transformer with Event-Role Stream

This repository contains the code and reproducibility package for PAT-ER, a decoder-only transformer augmented with event-role and logical-primitive side-state registers.

## Contents

- `src/pat_er/` — model, configuration, side-state modules, auxiliary heads, serialization.
- `scripts/` — dataset conversion, synthetic data generation, training, evaluation, warm-start, and interface-evaluation scripts.
- `configs/` — tiny, 450M, 760M-shape, and Qwen3 warm-start configs.
- `fixtures/` — small validation fixtures for external converters.
- `results/` — compact result tables used by the manuscript.
- `paper/mdpi_pat_er/` — MDPI LaTeX manuscript source and compiled PDF.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Smoke Checks

```bash
python scripts/smoke_forward.py --dtype fp32
python scripts/smoke_generate.py
python scripts/inspect_shapes.py
python scripts/smoke_760m.py
python -m compileall -q scripts src
```

## Reproducibility

See [`docs/reproducibility.md`](docs/reproducibility.md). External datasets are not redistributed here; use the cited upstream datasets and the conversion scripts in `scripts/`.

## Paper

The current article package is in [`paper/mdpi_pat_er/main.tex`](paper/mdpi_pat_er/main.tex). A compiled PDF is included at [`paper/mdpi_pat_er/main.pdf`](paper/mdpi_pat_er/main.pdf).

