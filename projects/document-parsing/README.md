<!--
Path: projects/document-parsing/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# PDF Page Classifier

Applied document-intelligence project that classifies PDF pages as **document-like** or **diagram-like** using interpretable geometric/visual heuristics plus an adaptive unsupervised calibration layer.

## Engineering highlights

- deterministic baseline and adaptive classifier kept side-by-side for comparison;
- explainable features returned with every decision;
- anomaly detection reduces confidence instead of creating an untestable hidden fallback class;
- reproducible offline evaluation with per-model metrics, bootstrap intervals, confusion matrices and per-page error analysis;
- regression tests cover the hybrid voting direction and adaptive calibration;
- packaged CLI instead of a hard-coded local PDF path.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
pdf-page-classifier /path/to/document.pdf
```

The previous repository copy bundled a sample PDF directly in source. The hardened version does **not** redistribute input documents; provide your own PDF at runtime.

## Evaluation

The classifier ships with a reproducible offline evaluation:

```bash
make eval-doc-parsing   # from the repository root
```

It builds deterministic synthetic pages, uses the six curated pages in
[`evals/fixtures/curated`](evals/fixtures/curated), and writes
[`evals/results.md`](evals/results.md) with accuracy, macro F1, bootstrap
intervals, confusion matrices and every misclassified page.

The baseline is deliberately unflattering: the hybrid reproduces the layout
branch, and confidence collapses to two values (`0.180` / `0.660`). The label
rubric, split policy and limitations are documented in
[`evals/README.md`](evals/README.md).

## Architecture

```text
PDF page
├── LayoutClassifier ── text blocks / occupancy / area
├── VisualLayoutClassifier ── edges / transitions
└── HybridClassifier ── consistent signed voting

Representative pages
└── ImprovedPDFClassifier
    ├── adaptive thresholds
    ├── DBSCAN block structure
    └── IsolationForest confidence adjustment
```
