"""Check that the recorded results are reproducible and internally consistent.

    python scripts/check_reproducibility.py

The two tracks are treated differently, because one of them costs forty minutes
on a GPU and the other costs a minute on a laptop.

The classical track is rerun in full and compared within a tolerance. The
transformer track is not rerun. Instead every metric recorded for it is
recomputed from the predictions stored alongside, and must agree exactly. That
catches the failure worth catching without a GPU: a number in the results file
that no longer follows from the predictions that produced it, whether through a
hand edit or a partial rerun.

The findings the README argues from are then checked against the recorded values,
and must hold exactly, because if one flips the README is saying something false.

Writes nothing. The recorded results file is not overwritten.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "benchmark.json"
sys.path.insert(0, str(ROOT))

TOLERANCE = 0.02


def recompute(predictions_true, predictions, recorded, label: str) -> list[str]:
    """Recompute every stored metric from the stored predictions."""
    from src.benchmark import scores

    fresh = scores(np.array(predictions_true), np.array(predictions))
    problems = []

    for key in ("accuracy", "macro_f1"):
        if abs(fresh[key] - recorded[key]) > 1e-9:
            problems.append(f"  {label}: recorded {key} is {recorded[key]:.6f} but "
                            f"the stored predictions give {fresh[key]:.6f}")

    if fresh["confusion_matrix"] != recorded["confusion_matrix"]:
        problems.append(f"  {label}: the recorded confusion matrix does not follow "
                        f"from the stored predictions")

    for name, entry in fresh["per_class"].items():
        for key, value in entry.items():
            stored = recorded["per_class"][name][key]
            if abs(value - stored) > 1e-9:
                problems.append(f"  {label}: recorded {name} {key} is {stored} but "
                                f"the stored predictions give {value}")
    return problems


def findings(data: dict) -> dict:
    """The claims the README rests on."""
    baselines = data["baselines"]
    chosen = baselines["chosen_on_validation"]
    baseline = baselines["models"][chosen]
    transformer = data["transformer"]
    comparison = data["comparison"]

    every_test_accuracy = {
        name: entry["test"]["accuracy"] for name, entry in baselines["models"].items()}

    return {
        "the baseline was chosen on validation, not on test":
            chosen == max(baselines["models"],
                          key=lambda name: baselines["models"][name]
                          ["validation"]["accuracy"]),
        "every model beats predicting the most frequent class": all(
            value > baselines["most_frequent_class_accuracy"]
            for value in every_test_accuracy.values()),
        "the linear SVM is the strongest classical model on test": (
            chosen == max(every_test_accuracy, key=every_test_accuracy.get)),
        "DistilBERT is above the strongest classical baseline":
            transformer["test"]["accuracy"] > baseline["test"]["accuracy"],
        "the gain is under five accuracy points":
            comparison["accuracy_gain_points"] < 5.0,
        "the transformer costs more to fit":
            comparison["training_time_multiple"] > 1.0,
        "the classical model answers far more documents per second":
            min(comparison["inference_ratio_range"]) > 100,
        "the transformer artifact is larger on disk":
            comparison["model_size_multiple"] > 1.0,
        "no exact train and test overlap was found":
            data["contamination"]["exact_matches_in_train"] == 0,
        "the normalised overlap is under one percent of the test split":
            data["contamination"]["normalised_match_fraction"] < 0.01,
        "both models confuse Business and Sci/Tech more than any other pair": all(
            {entry["confusion_pairs"][0]["true"],
             entry["confusion_pairs"][0]["predicted"]} == {"Business", "Sci/Tech"}
            for entry in (baseline, transformer)),
        "both artifact directories name their own preprocessing": all(
            any(name in {file["name"] for file in section["artifact"]["files"]}
                for name in required)
            for section, required in ((baselines, ("vectorizer.joblib",)),
                                      (transformer, ("tokenizer.json",)))),
    }


def main() -> int:
    if not RECORDED.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    failures = []

    # The transformer track is not rerun. Its metrics are recomputed from the
    # predictions stored beside them.
    if "transformer" in recorded:
        stored = recorded["transformer"]["predictions"]
        failures += recompute(stored["test_true"], stored["test_predicted"],
                              recorded["transformer"]["test"], "distilbert")
        print("  transformer metrics recomputed from its stored predictions")

    if "baselines" in recorded:
        stored = recorded["baselines"]["predictions"]
        for name, predicted in stored["test_predicted"].items():
            failures += recompute(stored["test_true"], predicted,
                                  recorded["baselines"]["models"][name]["test"], name)
        print(f"  {len(stored['test_predicted'])} baseline metric sets recomputed "
              f"from their stored predictions")

    # The classical track is cheap, so it is rerun in full.
    with tempfile.TemporaryDirectory() as directory:
        fresh_path = Path(directory) / "benchmark.json"
        print("  rerunning the classical track into a temporary file")
        completed = subprocess.run(
            [sys.executable, "-m", "src.benchmark", "--track", "baselines",
             "--out", str(fresh_path)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
            raise SystemExit(f"  the benchmark failed with exit code "
                             f"{completed.returncode}")
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))

    drifted = 0
    for name, entry in recorded["baselines"]["models"].items():
        was = entry["test"]["accuracy"]
        now = fresh["baselines"]["models"][name]["test"]["accuracy"]
        if abs(now - was) > TOLERANCE:
            drifted += 1
            failures.append(f"  {name} accuracy moved from {was:.4f} to {now:.4f}")
    if fresh["baselines"]["chosen_on_validation"] != \
            recorded["baselines"]["chosen_on_validation"]:
        failures.append("  the rerun selected a different baseline on validation")
    print(f"  {len(recorded['baselines']['models'])} baseline accuracies compared, "
          f"{len(recorded['baselines']['models']) - drifted} within {TOLERANCE:.0%}")

    if "comparison" in recorded:
        checks = findings(recorded)
        for description, holds in checks.items():
            if not holds:
                failures.append(f"  the recorded results no longer support: "
                                f"{description}")
        print(f"  {len(checks)} findings checked, "
              f"{sum(1 for value in checks.values() if value)} hold")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} checks failed")

    print("  the recorded results are consistent and still support the same "
          "conclusions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
