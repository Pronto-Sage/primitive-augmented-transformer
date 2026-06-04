# MDPI PAT-ER Article Package

This directory is an MDPI-ready article package for PAT-ER review and
submission adaptation.
It is not submitted anywhere.

## Recommended Journal

Primary recommendation: **Machine Learning and Knowledge Extraction (MAKE)**.

Rationale: MAKE is the closest MDPI scope fit for a machine-learning architecture
paper whose central contribution is a learned representation and evaluation matrix.
The current manuscript uses the MDPI `make` class option. Fallbacks are recorded in
`journal_recommendation.md`.

## Structure

- `main.tex` -- end-to-end MDPI manuscript assembled from section files.
- `sections/*.tex` -- major manuscript sections, split for submission-form mapping.
- `tables/*.tex` -- reusable table assets. `T2`--`T6` are included in the
  manuscript; `T1` supports journal-selection review.
- `figures/*.tex` -- TikZ figure/graph assets.
- `refs.bib` -- stable citation-key bibliography.
- `journal_recommendation.md` -- MDPI venue comparison and recommendation.
- `figures/FIGURE_NOTES.md` and `tables/TABLE_NOTES.md` -- provenance notes.
- `submission_release_checklist.md` -- required public-release steps before
  final submission.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The package includes a local copy of the MDPI `Definitions/` directory copied
from the existing submitted MDPI package for template compatibility.
