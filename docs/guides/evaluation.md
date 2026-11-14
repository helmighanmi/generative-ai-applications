# AI evaluation

Tests answer "does the code do what it says?". Evaluation answers "is the model
good enough?". They are different questions, with different failure modes, and
this repository keeps them separate.

See [`testing.md`](testing.md) for the software-testing side.

## Where evaluation lives

Evaluation belongs to the project that owns the model or retrieval pipeline:

```text
projects/<name>/
└── evals/
    ├── README.md        # rubric, dataset policy, limitations
    ├── manifest.json    # labelled cases and splits
    ├── <generator>.py   # only when fixtures are synthetic
    ├── evaluate.py      # runner; writes the report
    └── results.md       # committed, generated report
```

Do not put project-specific evaluation data at the repository root. The root
only describes the convention, which is this page.

## Required artefacts

1. **A manifest with stable case IDs.** Every case states its label and a short
   rationale, so a reviewer can disagree with the label instead of guessing at
   it.
2. **A declared split policy.** Calibration and test cases must be disjoint, and
   the runner must fail if they overlap. A metric measured on a page the model
   was tuned on is not a metric.
3. **A deterministic runner.** Fixed seeds, no timestamps, no absolute paths.
   Re-running on the same commit must produce the same report.
4. **A committed report.** The report is generated, never edited by hand. It
   records the command that produced it.
5. **A limitations section.** State the sample size, the sampling method, and
   what the numbers do *not* estimate.

## Metrics

Accuracy alone hides class imbalance and label ambiguity. A report should
include, at minimum:

- accuracy and macro F1;
- per-class precision, recall and support;
- the confusion matrix;
- a confidence interval when the sample is small;
- every misclassified case, with the features that explain the decision.

If the model exposes a confidence value, state whether it is a calibrated
probability or a decision margin. Do not report a Brier score for a number that
was never calibrated.

## Gating

A gate is only useful when it is tied to a stated threshold. The evaluation
runner accepts `--fail-under-accuracy` and `--fail-under-macro-f1` and exits
non-zero when either fails. CI wiring for those gates is tracked separately from
the evaluation itself.

## Reference implementation

[`projects/document-parsing/evals`](../../projects/document-parsing/evals/README.md)
is the reference: a manifest, a deterministic synthetic generator, six curated
PDF pages, bootstrap intervals, confusion matrices and per-case error analysis.
