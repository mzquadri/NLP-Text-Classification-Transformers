"""Render the README figures from results/benchmark.json.

    python scripts/figures/generate_figures.py

Each figure answers one question:

  01  how do all five models compare on the same held-out test split
  02  what does DistilBERT's extra accuracy cost against the strongest baseline
  03  which classes each model confuses, side by side on the same scale

Everything is read from the recorded results rather than recomputed, so a figure
cannot disagree with the numbers in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import portfolio_style as ps

RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"

NICE = {
    "most_frequent": "Most frequent\nclass",
    "random_forest": "TF-IDF\nrandom forest",
    "logistic_regression": "TF-IDF\nlogistic regression",
    "linear_svm": "TF-IDF\nlinear SVM",
    "distilbert": "DistilBERT\nfine-tuned",
}
ORDER = ("most_frequent", "random_forest", "logistic_regression", "linear_svm",
         "distilbert")


def load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for section in ("baselines", "transformer"):
        if section not in data:
            raise SystemExit(
                f"results/benchmark.json has no '{section}' section; run "
                f"`python -m src.benchmark` to produce both tracks")
    return data


def figure_01_model_comparison(data: dict) -> None:
    """How do all five compare on the same held-out test split?"""
    baselines = data["baselines"]["models"]
    transformer = data["transformer"]
    floor = data["baselines"]["most_frequent_class_accuracy"]

    accuracy = {"most_frequent": floor * 100}
    macro_f1 = {"most_frequent": None}
    for name, entry in baselines.items():
        accuracy[name] = entry["test"]["accuracy"] * 100
        macro_f1[name] = entry["test"]["macro_f1"] * 100
    accuracy["distilbert"] = transformer["test"]["accuracy"] * 100
    macro_f1["distilbert"] = transformer["test"]["macro_f1"] * 100

    chosen = data["baselines"]["chosen_on_validation"]

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.subplots_adjust(left=0.085, right=0.955, top=0.745, bottom=0.245)

    positions = np.arange(len(ORDER))
    width = 0.36
    colours = [ps.AMBER if name == "most_frequent"
               else ps.BLUE if name == "distilbert"
               else ps.TEAL if name == chosen else ps.SLATE
               for name in ORDER]

    ax.bar(positions - width / 2, [accuracy[name] for name in ORDER], width,
           color=colours, label="Accuracy")
    ax.bar(positions + width / 2,
           [macro_f1[name] if macro_f1[name] is not None else 0 for name in ORDER],
           width, color=[ps.SLATE_SOFT if name == "most_frequent" else colour
                         for name, colour in zip(ORDER, colours, strict=True)],
           alpha=0.55, label="Macro F1")

    for index, name in enumerate(ORDER):
        ax.text(index - width / 2, accuracy[name] + 1.1, f"{accuracy[name]:.2f}",
                ha="center", fontsize=10.2, color=ps.INK, fontweight="600")

    ax.set_xticks(positions)
    ax.set_xticklabels([NICE[name] for name in ORDER], fontsize=10.2, color=ps.INK)
    ax.set_ylabel("Score on the 7,600 held-out documents", fontsize=11,
                  color=ps.MUTED)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", frameon=False, fontsize=10.4, labelcolor=ps.MUTED)
    ps.clean(ax, grid_axis="y")

    ps.title_block(
        fig, "All five, scored on the same held-out documents",
        "AG News, four classes. Model choice was made on a validation split; "
        "the test split was scored\nonce per model. Teal is the baseline "
        "validation selected.")

    ps.footnote(fig, [
        "The most frequent class scores 25 because the four classes are balanced, "
        "and it has no macro F1 worth plotting since it",
        "never predicts three of them. Everything above it is a real result on "
        "the published test split, not a validation number.",
    ])
    ps.save(fig, FIGURES, "01_model_comparison")


def figure_02_cost_against_quality(data: dict) -> None:
    """What does the extra accuracy cost?"""
    comparison = data["comparison"]
    chosen = data["baselines"]["chosen_on_validation"]
    baseline = data["baselines"]["models"][chosen]
    transformer = data["transformer"]

    panels = [
        ("Accuracy", "percent", baseline["test"]["accuracy"] * 100,
         transformer["test"]["accuracy"] * 100, "{:.2f}", False),
        ("Fit time", "seconds", baseline["total_fit_seconds"],
         transformer["train_seconds"], "{:,.0f}", True),
        ("Artifact on disk", "megabytes",
         data["baselines"]["artifact"]["total_bytes"] / 1e6,
         transformer["artifact"]["total_bytes"] / 1e6, "{:,.0f}", True),
        ("Time per document", "milliseconds", baseline["milliseconds_per_document"],
         transformer["milliseconds_per_document"], "{:.3f}", True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 6.0))
    fig.subplots_adjust(left=0.055, right=0.975, top=0.695, bottom=0.245, wspace=0.42)

    for axis, (title, unit, left_value, right_value, fmt, log) in zip(
            axes, panels, strict=True):
        bars = axis.bar([0, 1], [left_value, right_value], width=0.58,
                        color=[ps.TEAL, ps.BLUE])
        if log:
            axis.set_yscale("log")
            axis.set_ylim(min(left_value, right_value) * 0.35,
                          max(left_value, right_value) * 4.5)
        else:
            axis.set_ylim(0, max(left_value, right_value) * 1.22)
        for bar, value in zip(bars, (left_value, right_value), strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2,
                      value * (1.35 if log else 1.02), fmt.format(value),
                      ha="center", fontsize=10.6, color=ps.INK, fontweight="600")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["linear\nSVM", "Distil\nBERT"], fontsize=10,
                             color=ps.INK)
        axis.set_title(title, fontsize=12, color=ps.INK, pad=10, loc="left")
        axis.set_ylabel(unit, fontsize=10, color=ps.MUTED)
        ps.clean(axis, grid_axis="y")

    ps.title_block(
        fig, "What the extra accuracy costs",
        "DistilBERT against the strongest classical baseline, on the same "
        "held-out documents. Three of the four\npanels use a log scale, because "
        "the cost differences are much larger than the accuracy difference.")

    ps.footnote(fig, [
        f"{comparison['accuracy_gain_points']:+.2f} points of accuracy and "
        f"{comparison['macro_f1_gain_points']:+.2f} of macro F1, for "
        f"{comparison['training_time_multiple']:,.0f} times the fit time, "
        f"{comparison['model_size_multiple']:,.0f} times the artifact size and "
        f"{comparison['inference_time_multiple']:,.0f} times the",
        "time per document. The fit time is hardware-specific: the SVM was fitted "
        "on a CPU and the transformer on a GPU.",
    ])
    ps.save(fig, FIGURES, "02_cost_against_quality")


def figure_03_confusion(data: dict) -> None:
    """Which classes does each model confuse?"""
    chosen = data["baselines"]["chosen_on_validation"]
    matrices = {
        NICE[chosen].replace("\n", " "):
            np.array(data["baselines"]["models"][chosen]["test"]["confusion_matrix"]),
        NICE["distilbert"].replace("\n", " "):
            np.array(data["transformer"]["test"]["confusion_matrix"]),
    }
    classes = list(data["transformer"]["test"]["per_class_f1"])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.705, bottom=0.235, wspace=0.28)

    # One shared colour scale, so the two panels are directly comparable. Only
    # the off-diagonal cells are coloured: the diagonal is two orders of
    # magnitude larger and would flatten everything else to white.
    off_diagonal = max(
        int((matrix - np.diag(np.diag(matrix))).max()) for matrix in matrices.values())

    for axis, (title, matrix) in zip(axes, matrices.items(), strict=True):
        shown = matrix - np.diag(np.diag(matrix))
        axis.imshow(shown, cmap="Blues", vmin=0, vmax=off_diagonal)
        for row in range(len(classes)):
            for column in range(len(classes)):
                value = int(matrix[row][column])
                diagonal = row == column
                axis.text(column, row, f"{value:,}", ha="center", va="center",
                          fontsize=10.4,
                          color=ps.FAINT if diagonal
                          else "white" if shown[row][column] > off_diagonal * 0.55
                          else ps.INK,
                          fontweight="400" if diagonal else "600")
        axis.set_xticks(range(len(classes)), classes, fontsize=9.6, rotation=20,
                        ha="right")
        axis.set_yticks(range(len(classes)), classes, fontsize=9.6)
        axis.set_xlabel("predicted", fontsize=10.4, color=ps.MUTED)
        axis.set_title(title, fontsize=12.5, color=ps.INK, pad=11, loc="left")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)
    axes[0].set_ylabel("true", fontsize=10.4, color=ps.MUTED)

    worst = data["transformer"]["confusion_pairs"][0]
    ps.title_block(
        fig, "Both models confuse the same two classes",
        "Counts on the 7,600 held-out documents, same class order and same colour "
        "scale. Correct counts on the\ndiagonal are greyed, because colouring them "
        "would flatten every error to white.")

    ps.footnote(fig, [
        f"The largest single error cell for both models is {worst['true']} read as "
        f"{worst['predicted']}. Business and Sci/Tech overlap in AG News: a story",
        "about a technology company's results belongs to either, and the label "
        "depends on the desk that filed it rather than the words.",
    ])
    ps.save(fig, FIGURES, "03_confusion")


def main() -> int:
    ps.apply()
    data = load()
    print(f"  reading {RESULTS.relative_to(ROOT).as_posix()}")
    figure_01_model_comparison(data)
    figure_02_cost_against_quality(data)
    figure_03_confusion(data)
    print(f"  figures in {FIGURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
