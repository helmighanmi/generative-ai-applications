"""Offline evaluation for the PDF page classifier.

The script builds deterministic synthetic fixtures, reads the curated PDFs from
``fixtures/curated``, fits the adaptive classifier on the calibration split, and
writes a Markdown report with per-model metrics, confidence intervals, confusion
matrices and misclassified cases.

Run with ``make eval-doc-parsing`` from the repository root, or directly:

    python projects/document-parsing/evals/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
# Allow running as a script without an editable install of the project.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVALS_DIR))

from make_fixtures import build_fixtures  # noqa: E402
from pdf_page_classifier import (  # noqa: E402
    HybridClassifier,
    ImprovedPDFClassifier,
    LayoutClassifier,
    VisualLayoutClassifier,
)
from pdf_page_classifier.metrics import (  # noqa: E402
    DEFAULT_LABELS,
    ClassificationMetrics,
    assert_disjoint_splits,
    bootstrap_accuracy_ci,
    compute_metrics,
)

MODEL_NAMES = ("layout", "visual", "hybrid", "adaptive")
DISPLAY_NAMES = {
    "layout": "layout-only",
    "visual": "visual-only",
    "hybrid": "hybrid",
    "adaptive": "adaptive (calibrated)",
}
CONFIDENCE_MODELS = ("hybrid", "adaptive")


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled page in the evaluation set."""

    id: str
    source: str
    split: str
    label: str
    rationale: str
    page_index: int
    generator: str | None = None
    variant: int = 0
    document: str | None = None
    difficulty: str = "standard"


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Metrics and a bootstrap interval for one model."""

    metrics: ClassificationMetrics
    confidence_interval: tuple[float, float]


def load_cases(manifest_path: Path) -> tuple[list[EvalCase], str]:
    """Load and validate the evaluation manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if not raw_cases:
        raise ValueError(f"manifest contains no cases: {manifest_path}")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        case = EvalCase(
            id=str(raw["id"]),
            source=str(raw["source"]),
            split=str(raw["split"]),
            label=str(raw["label"]),
            rationale=str(raw.get("rationale", "")),
            page_index=int(raw["page"]),
            generator=raw.get("generator"),
            variant=int(raw.get("variant", 0)),
            document=raw.get("document"),
            difficulty=str(raw.get("difficulty", "standard")),
        )
        if case.id in seen:
            raise ValueError(f"duplicate case id: {case.id}")
        seen.add(case.id)
        if case.source not in {"synthetic", "curated"}:
            raise ValueError(f"{case.id}: unsupported source {case.source!r}")
        if case.split not in {"calibration", "test"}:
            raise ValueError(f"{case.id}: unsupported split {case.split!r}")
        if case.label not in DEFAULT_LABELS:
            raise ValueError(f"{case.id}: unsupported label {case.label!r}")
        if (case.generator is None) == (case.document is None):
            raise ValueError(f"{case.id}: set exactly one of generator or document")
        cases.append(case)

    calibration_ids = [case.id for case in cases if case.split == "calibration"]
    test_ids = [case.id for case in cases if case.split == "test"]
    if not calibration_ids or not test_ids:
        raise ValueError("manifest needs at least one calibration and one test case")
    assert_disjoint_splits(calibration_ids, test_ids)

    return cases, str(payload.get("label_rubric", ""))


def _path_for(case: EvalCase, synthetic_paths: dict[tuple[str, int], Path]) -> Path:
    if case.generator is not None:
        return synthetic_paths[(case.generator, case.variant)]
    if case.document is None:
        raise ValueError(f"{case.id}: no document path")
    return EVALS_DIR / case.document


def _open_page(
    case: EvalCase,
    synthetic_paths: dict[tuple[str, int], Path],
) -> tuple[pymupdf.Document, pymupdf.Page]:
    document = pymupdf.open(_path_for(case, synthetic_paths))
    if case.page_index >= document.page_count:
        document.close()
        raise ValueError(f"{case.id}: page {case.page_index} is missing")
    return document, document[case.page_index]


def predict(
    cases: list[EvalCase],
    synthetic_paths: dict[tuple[str, int], Path],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Return predictions, confidences and features for every test case."""
    calibration = [case for case in cases if case.split == "calibration"]
    adaptive = ImprovedPDFClassifier()
    calibration_documents: list[pymupdf.Document] = []
    try:
        calibration_pages = []
        for case in calibration:
            document, page = _open_page(case, synthetic_paths)
            calibration_documents.append(document)
            calibration_pages.append(page)
        adaptive.calibrate(calibration_pages)
    finally:
        for document in calibration_documents:
            document.close()

    layout = LayoutClassifier()
    visual = VisualLayoutClassifier()
    hybrid = HybridClassifier()

    predictions: dict[str, dict[str, str]] = {name: {} for name in MODEL_NAMES}
    confidences: dict[str, dict[str, float]] = {name: {} for name in CONFIDENCE_MODELS}
    features: dict[str, dict[str, float]] = {}

    for case in cases:
        if case.split != "test":
            continue
        document, page = _open_page(case, synthetic_paths)
        try:
            layout_result = layout.classify(page)
            visual_result = visual.classify(page)
            hybrid_result = hybrid.classify(page)
            adaptive_result = adaptive.classify(page)
            features[case.id] = adaptive.extract_features(page)
        finally:
            document.close()

        predictions["layout"][case.id] = layout_result.page_type.value
        predictions["visual"][case.id] = visual_result.page_type.value
        predictions["hybrid"][case.id] = hybrid_result.page_type.value
        predictions["adaptive"][case.id] = adaptive_result.page_type.value
        confidences["hybrid"][case.id] = hybrid_result.confidence
        confidences["adaptive"][case.id] = adaptive_result.confidence

    return predictions, confidences, features


def evaluate_models(
    cases: list[EvalCase],
    predictions: dict[str, dict[str, str]],
) -> dict[str, ModelResult]:
    """Compute metrics for every model on the supplied cases."""
    if not cases:
        raise ValueError("cannot evaluate an empty case list")
    truth = [case.label for case in cases]
    results: dict[str, ModelResult] = {}
    for model in MODEL_NAMES:
        predicted = [predictions[model][case.id] for case in cases]
        results[model] = ModelResult(
            metrics=compute_metrics(truth, predicted),
            confidence_interval=bootstrap_accuracy_ci(truth, predicted),
        )
    return results


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _dataset_table(cases: list[EvalCase]) -> list[str]:
    counts = Counter((case.source, case.split, case.label) for case in cases)
    groups = sorted({(source, split) for source, split, _ in counts})
    rows: list[list[str]] = []
    for source, split in groups:
        document = counts[(source, split, "document")]
        diagram = counts[(source, split, "diagram")]
        rows.append([source, split, str(document), str(diagram), str(document + diagram)])
    return _markdown_table(
        ["Source", "Split", "document", "diagram", "Total"],
        rows,
    )


def _metrics_table(results: dict[str, ModelResult], include_ci: bool) -> list[str]:
    rows: list[list[str]] = []
    for model in MODEL_NAMES:
        result = results[model]
        row = [
            DISPLAY_NAMES[model],
            _format_float(result.metrics.accuracy),
            _format_float(result.metrics.macro_f1),
        ]
        if include_ci:
            lower, upper = result.confidence_interval
            row.append(f"[{_format_float(lower)}, {_format_float(upper)}]")
        row.append(str(result.metrics.n_samples))
        rows.append(row)
    headers = ["Model", "Accuracy", "Macro F1"]
    if include_ci:
        headers.append("95% CI (accuracy)")
    headers.append("N")
    return _markdown_table(headers, rows)


def _per_source_table(
    cases: list[EvalCase],
    predictions: dict[str, dict[str, str]],
) -> list[str]:
    rows: list[list[str]] = []
    for source in ("synthetic", "curated"):
        subset = [case for case in cases if case.source == source]
        if not subset:
            continue
        results = evaluate_models(subset, predictions)
        for model in MODEL_NAMES:
            result = results[model]
            rows.append(
                [
                    source,
                    DISPLAY_NAMES[model],
                    _format_float(result.metrics.accuracy),
                    _format_float(result.metrics.macro_f1),
                    str(result.metrics.n_samples),
                ]
            )
    return _markdown_table(["Source", "Model", "Accuracy", "Macro F1", "N"], rows)


def _confusion_table(metrics: ClassificationMetrics) -> list[str]:
    headers = ["actual \\ predicted", *metrics.labels, "support"]
    rows: list[list[str]] = []
    for index, label in enumerate(metrics.labels):
        row = [label]
        row.extend(str(int(value)) for value in metrics.confusion[index])
        row.append(str(int(metrics.confusion[index].sum())))
        rows.append(row)
    return _markdown_table(headers, rows)


def _feature_summary(features: dict[str, float]) -> str:
    return (
        f"word_count={int(features.get('word_count', 0))}, "
        f"occ_20={features.get('occ_20', 0.0):.2f}, "
        f"edges_200={features.get('edges_200', 0.0):.4f}, "
        f"n_clusters={int(features.get('n_clusters', 0))}"
    )


def _confidence_table(confidences: dict[str, dict[str, float]]) -> list[str]:
    rows: list[list[str]] = []
    for model in CONFIDENCE_MODELS:
        counter = Counter(round(value, 3) for value in confidences[model].values())
        distribution = ", ".join(
            f"{value:.3f} × {count}" for value, count in sorted(counter.items())
        )
        rows.append([DISPLAY_NAMES[model], str(len(counter)), distribution])
    return _markdown_table(["Model", "Distinct values", "Distribution"], rows)


def render_report(
    cases: list[EvalCase],
    label_rubric: str,
    predictions: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    features: dict[str, dict[str, float]],
) -> str:
    """Render the deterministic Markdown report."""
    test_cases = [case for case in cases if case.split == "test"]
    results = evaluate_models(test_cases, predictions)
    standard_cases = [case for case in test_cases if case.difficulty != "hard"]
    standard_results = evaluate_models(standard_cases, predictions)

    lines: list[str] = [
        "# Page classification evaluation",
        "",
        "Generated by `make eval-doc-parsing`. Do not edit this file by hand;",
        "change the evaluation code and regenerate it.",
        "",
        "## Label rubric",
        "",
        label_rubric or "(no rubric recorded in the manifest)",
        "",
        "## Dataset",
        "",
        *_dataset_table(cases),
        "",
        "Calibration pages are never used as test pages. The evaluation fails if the",
        "two splits overlap.",
        "",
        "## Test-set metrics",
        "",
        f"All test cases (N = {len(test_cases)}):",
        "",
        *_metrics_table(results, include_ci=True),
        "",
        f"Excluding documented hard cases (N = {len(standard_cases)}):",
        "",
        *_metrics_table(standard_results, include_ci=True),
        "",
        "## Metrics per source",
        "",
        *_per_source_table(test_cases, predictions),
        "",
        "The synthetic split measures the mechanics of the method; the curated split",
        "measures behaviour on intentionally designed, mixed pages.",
        "",
        "## Confusion matrices",
        "",
    ]

    for model in MODEL_NAMES:
        lines.extend(
            [
                f"### {DISPLAY_NAMES[model]}",
                "",
                *_confusion_table(results[model].metrics),
                "",
            ]
        )

    lines.extend(
        [
            "## Confidence distribution",
            "",
            "The current `confidence` field is a decision margin, not a calibrated",
            "probability, so no Brier score is reported yet. A single repeated value",
            "means the field carries no information on this set.",
            "",
            *_confidence_table(confidences),
            "",
            "## Misclassified cases",
            "",
        ]
    )

    for model in MODEL_NAMES:
        lines.append(f"### {DISPLAY_NAMES[model]}")
        lines.append("")
        errors = [case for case in test_cases if predictions[model][case.id] != case.label]
        if not errors:
            lines.extend(["No misclassifications.", ""])
            continue
        for case in errors:
            difficulty = "" if case.difficulty == "standard" else f" ({case.difficulty})"
            lines.append(
                f"- `{case.id}`{difficulty} — expected `{case.label}`, "
                f"predicted `{predictions[model][case.id]}` · "
                f"{_feature_summary(features[case.id])}"
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- The curated set contains a small number of intentionally designed",
            "  pages. It is not a random sample of real-world PDFs, so these numbers",
            "  do not estimate production accuracy.",
            "- The test set is small. The bootstrap interval is wide, and one flipped",
            "  page changes accuracy by several percentage points.",
            "- Scanned prose and the short title page are documented hard cases: the",
            "  current features cannot recover text that is absent or very sparse.",
            "- Synthetic pages are generated from code, so they test the decision",
            "  mechanics under controlled conditions rather than visual diversity.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EVALS_DIR / "manifest.json",
        help="Evaluation manifest (default: evals/manifest.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALS_DIR / "results.md",
        help="Markdown report path (default: evals/results.md)",
    )
    parser.add_argument(
        "--gate-model",
        choices=MODEL_NAMES,
        default="hybrid",
        help="Model whose metrics are compared with the failure thresholds",
    )
    parser.add_argument(
        "--fail-under-accuracy",
        type=float,
        default=None,
        help="Exit non-zero if the gated model's accuracy is below this value",
    )
    parser.add_argument(
        "--fail-under-macro-f1",
        type=float,
        default=None,
        help="Exit non-zero if the gated model's macro F1 is below this value",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip cases whose curated document is not present",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases, label_rubric = load_cases(args.manifest)

    missing = [
        case
        for case in cases
        if case.document is not None and not (EVALS_DIR / case.document).is_file()
    ]
    if missing:
        if not args.skip_missing:
            names = ", ".join(case.id for case in missing)
            raise FileNotFoundError(f"missing curated fixtures: {names}")
        cases = [case for case in cases if case not in missing]
        print(f"Skipping {len(missing)} curated cases with missing documents", file=sys.stderr)

    synthetic_specs = {
        (case.generator, case.variant) for case in cases if case.generator is not None
    }
    with tempfile.TemporaryDirectory(prefix="pdf-page-eval-") as temporary:
        synthetic_paths = build_fixtures(Path(temporary), synthetic_specs)
        predictions, confidences, page_features = predict(cases, synthetic_paths)

    report = render_report(cases, label_rubric, predictions, confidences, page_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    test_cases = [case for case in cases if case.split == "test"]
    results = evaluate_models(test_cases, predictions)
    gated = results[args.gate_model].metrics
    print(
        f"Wrote {args.output} | {args.gate_model}: "
        f"accuracy={gated.accuracy:.3f}, macro_f1={gated.macro_f1:.3f}, n={gated.n_samples}"
    )

    failures: list[str] = []
    if args.fail_under_accuracy is not None and gated.accuracy < args.fail_under_accuracy:
        failures.append(f"accuracy {gated.accuracy:.3f} < {args.fail_under_accuracy:.3f}")
    if args.fail_under_macro_f1 is not None and gated.macro_f1 < args.fail_under_macro_f1:
        failures.append(f"macro F1 {gated.macro_f1:.3f} < {args.fail_under_macro_f1:.3f}")
    if failures:
        print("Evaluation gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
