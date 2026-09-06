"""Fail if the README stops agreeing with the recorded results.

    python scripts/check_repository.py

Checks that the files the README points at exist and compile, that the recorded
artifact directories name their own preprocessing, and that every number the
README quotes still matches `results/benchmark.json`.

The last is the one that matters. The transformer run takes forty minutes on a
GPU and is not repeated in continuous integration, so the committed results file
is the only record of it. A README that drifts from that file would be the only
description of a run nobody is going to redo.

Each claim is matched with its surrounding words included, so a value that has
moved into a different sentence does not accidentally satisfy a check.
"""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements-transformer.txt",
    "pyproject.toml",
    "src/benchmark.py",
    "src/artifacts.py",
    "src/error_analysis.py",
    "src/data_loader.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/inference.py",
    "src/utils.py",
    "tests/test_leakage.py",
    "tests/test_artifacts.py",
    "scripts/check_reproducibility.py",
    "scripts/figures/generate_figures.py",
    "notebooks/01_EDA_and_Preprocessing.ipynb",
    "notebooks/02_Training_and_Evaluation.ipynb",
    "docs/diagrams/approach.svg",
    "docs/figures/01_model_comparison.png",
    "docs/figures/02_cost_against_quality.png",
    "docs/figures/03_confusion.png",
)


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    baselines = data["baselines"]
    chosen = baselines["chosen_on_validation"]
    baseline = baselines["models"][chosen]
    transformer = data["transformer"]
    comparison = data["comparison"]
    contamination = data["contamination"]
    configuration = transformer["configuration"]

    out = [
        # Dataset.
        (f"{data['data']['train']:,} train", "the training split size"),
        (f"{data['data']['validation']:,} validation", "the validation split size"),
        (f"{data['data']['test']:,} held-out", "the test split size"),

        # Contamination, stated without overclaiming.
        (f"No exact train and test overlap was found; "
         f"{contamination['normalised_matches_in_train']} normalised text matches",
         "the contamination result"),
        (f"in the {contamination['test_rows']:,}-example published test set",
         "the test set size in the contamination sentence"),

        # DistilBERT configuration.
        (f"`{configuration['base_checkpoint']}`", "the base checkpoint"),
        (f"{configuration['epochs']} epochs", "the epoch count"),
        (f"batch size {configuration['batch_size']}", "the batch size"),
        (f"learning rate {configuration['learning_rate']:g}", "the learning rate"),
        (f"{configuration['optimizer']}", "the optimizer"),
        (f"max sequence length {configuration['max_sequence_length']}",
         "the sequence length"),
        (f"seed {configuration['seed']}", "the seed"),
        (f"{transformer['parameters']:,} parameters", "the parameter count"),
        (f"epoch {transformer['epoch_chosen_on_validation']} was chosen",
         "the chosen epoch"),

        # Headline comparison.
        (f"| Most frequent class | "
         f"{baselines['most_frequent_class_accuracy'] * 100:.2f} |",
         "the most frequent class row"),
        (f"| DistilBERT | {transformer['test']['accuracy'] * 100:.2f} | "
         f"{transformer['test']['macro_f1'] * 100:.2f} |",
         "the DistilBERT results row"),
        (f"{comparison['accuracy_gain_points']:+.2f} points of accuracy",
         "the accuracy gain"),
        (f"{comparison['macro_f1_gain_points']:+.2f} of macro F1",
         "the macro F1 gain"),
        (f"{comparison['errors_removed_fraction'] * 100:.1f} percent of its errors",
         "the share of errors removed"),
        (f"{comparison['training_time_multiple']:,.0f} times the fit time",
         "the fit time multiple"),
        (f"{comparison['inference_time_multiple']:,.0f} times the time per document",
         "the inference multiple"),
        (f"{comparison['model_size_multiple']:,.0f} times the artifact size",
         "the artifact size multiple"),
    ]

    for name, entry in baselines["models"].items():
        pretty = {"logistic_regression": "TF-IDF, logistic regression",
                  "linear_svm": "TF-IDF, linear SVM",
                  "random_forest": "TF-IDF, random forest"}[name]
        out.append((
            f"| {pretty} | {entry['test']['accuracy'] * 100:.2f} | "
            f"{entry['test']['macro_f1'] * 100:.2f} |",
            f"the {name} results row"))

    # Per-class table for both models.
    for label, section in ((chosen, baseline["test"]), ("distilbert",
                                                        transformer["test"])):
        for name, entry in section["per_class"].items():
            out.append((
                f"| {name} | {entry['precision'] * 100:.2f} | "
                f"{entry['recall'] * 100:.2f} | {entry['f1'] * 100:.2f} |",
                f"the {label} {name} per-class row"))

    # Error analysis.
    if "error_analysis" in data:
        overlap = data["error_analysis"]["overlap"]
        out += [
            (f"{overlap['both_wrong']:,} of them", "the shared error count"),
            (f"| Both wrong | {overlap['both_wrong']:,} |", "the both-wrong row"),
            (f"| Only the SVM wrong | {overlap['only_baseline_wrong']:,} |",
             "the baseline-only error row"),
            (f"| Only DistilBERT wrong | {overlap['only_transformer_wrong']:,} |",
             "the transformer-only error row"),
        ]

    worst = transformer["confusion_pairs"][0]
    out.append((f"{worst['true']} read as {worst['predicted']}",
                "the dominant confusion pair"))
    return out


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")

    for source in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(source, doraise=True)
    print(f"  {len(REQUIRED_FILES)} required files present, src/ compiles")

    if not RESULTS.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    # An artifact directory that does not name its own preprocessing is the bug
    # this repository had. The results file records what was on disk.
    for section, required in (("baselines", "vectorizer.joblib"),
                              ("transformer", "tokenizer.json")):
        if section not in data:
            continue
        names = {entry["name"] for entry in data[section]["artifact"]["files"]}
        for needed in (required, "metadata.json"):
            if needed not in names:
                raise SystemExit(
                    f"  the recorded {section} artifact has no {needed}; a model "
                    f"without its preprocessing cannot be evaluated correctly")
    print("  both recorded artifacts name their own preprocessing")

    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    failures = []
    checks = claims(data)
    for expected, description in checks:
        if re.sub(r"\s+", " ", expected) not in flat:
            failures.append(f"  README does not state {description}: expected "
                            f"{re.sub(r'\\s+', ' ', expected).strip()!r}")
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} claims in the README no longer match "
                         f"results/benchmark.json")

    print("  README and results/benchmark.json agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
