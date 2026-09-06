"""Select representative mistakes from the recorded predictions.

    python -m src.error_analysis

Aggregate accuracy says how often a model is wrong. It does not say what it is
wrong about, and on AG News the errors are not spread evenly: one class pair
accounts for a large share of them. This module picks examples by rule rather
than by eye, so the ones shown are the ones the confusion matrix says matter
rather than the ones that read well.

Categories are derived from the confusion matrix that was actually produced. A
category with no supporting mistakes is dropped rather than illustrated with
something that nearly fits.

Reads and updates results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import LABEL_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"

#: Length below which a headline is treated as short. AG News items are a
#: headline plus a description; the short tail is mostly headline alone, which
#: is where there is least to go on.
SHORT_WORDS = 12
EXAMPLES_PER_CATEGORY = 3
EXCERPT_CHARACTERS = 190


def excerpt(text: str) -> str:
    text = " ".join(text.replace("\\", " ").split())
    return text if len(text) <= EXCERPT_CHARACTERS else (
        text[:EXCERPT_CHARACTERS].rstrip() + "...")


def categorise(texts, actual, predicted, matrix) -> dict:
    """Group the mistakes by what kind of mistake they are.

    The confusion pairs come first and in order of size, because those are the
    categories the data supports. The two text-shape categories are applied only
    to rows that are already wrong.
    """
    wrong = np.flatnonzero(actual != predicted)
    categories = {}

    ordered_pairs = sorted(
        ((row, column, matrix[row][column])
         for row in range(len(LABEL_NAMES))
         for column in range(len(LABEL_NAMES)) if row != column),
        key=lambda entry: -entry[2])

    for row, column, count in ordered_pairs[:3]:
        if count == 0:
            continue
        indices = [index for index in wrong
                   if actual[index] == row and predicted[index] == column]
        categories[f"{LABEL_NAMES[row]} called {LABEL_NAMES[column]}"] = {
            "kind": "confusion pair",
            "count": len(indices),
            "share_of_errors": round(len(indices) / len(wrong), 4),
            "indices": indices,
        }

    short = [index for index in wrong
             if len(texts[index].split()) < SHORT_WORDS]
    if short:
        short_rate = len(short) / max(
            sum(1 for text in texts if len(text.split()) < SHORT_WORDS), 1)
        overall_rate = len(wrong) / len(texts)
        categories["short items"] = {
            "kind": "text shape",
            "count": len(short),
            "share_of_errors": round(len(short) / len(wrong), 4),
            "error_rate_among_short_items": round(short_rate, 4),
            "error_rate_overall": round(overall_rate, 4),
            "indices": short,
        }

    # Entity-heavy: many capitalised tokens that are not sentence-initial. A
    # headline that is mostly names carries little topical vocabulary for a
    # bag-of-words model to use.
    def capitalised_run(text: str) -> int:
        tokens = text.split()
        return sum(1 for token in tokens[1:] if token[:1].isupper())

    entity_heavy = [index for index in wrong if capitalised_run(texts[index]) >= 6]
    if entity_heavy:
        categories["entity-heavy items"] = {
            "kind": "text shape",
            "count": len(entity_heavy),
            "share_of_errors": round(len(entity_heavy) / len(wrong), 4),
            "indices": entity_heavy,
        }

    return categories


def build(data: dict, texts) -> dict:
    """Error analysis for the strongest classical baseline and the transformer."""
    chosen = data["baselines"]["chosen_on_validation"]
    actual = np.array(data["baselines"]["predictions"]["test_true"])

    sources = {
        chosen: np.array(data["baselines"]["predictions"]["test_predicted"][chosen]),
        "distilbert": np.array(data["transformer"]["predictions"]["test_predicted"]),
    }

    analysis = {}
    for name, predicted in sources.items():
        matrix = (data["baselines"]["models"][chosen]["test"]["confusion_matrix"]
                  if name == chosen else
                  data["transformer"]["test"]["confusion_matrix"])
        categories = categorise(texts, actual, predicted, matrix)

        analysis[name] = {
            "total_errors": int((actual != predicted).sum()),
            "categories": {
                label: {
                    key: value for key, value in entry.items() if key != "indices"
                } | {
                    "examples": [
                        {
                            "text": excerpt(texts[index]),
                            "true": LABEL_NAMES[actual[index]],
                            "predicted": LABEL_NAMES[predicted[index]],
                        }
                        for index in entry["indices"][:EXAMPLES_PER_CATEGORY]
                    ]
                }
                for label, entry in categories.items()
            },
        }

    # Where the two models disagree, and where they fail together.
    baseline_wrong = actual != sources[chosen]
    transformer_wrong = actual != sources["distilbert"]
    analysis["overlap"] = {
        "baseline_errors": int(baseline_wrong.sum()),
        "transformer_errors": int(transformer_wrong.sum()),
        "both_wrong": int((baseline_wrong & transformer_wrong).sum()),
        "only_baseline_wrong": int((baseline_wrong & ~transformer_wrong).sum()),
        "only_transformer_wrong": int((~baseline_wrong & transformer_wrong).sum()),
        "note": "items both models miss are the ones a bigger model of the same "
                "kind is least likely to fix",
    }
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    if not arguments.out.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    data = json.loads(arguments.out.read_text(encoding="utf-8"))
    for section in ("baselines", "transformer"):
        if section not in data or "predictions" not in data[section]:
            raise SystemExit(
                f"results has no recorded predictions for '{section}'; rerun "
                f"`python -m src.benchmark --track {section.rstrip('s')}`")

    from .benchmark import load_splits
    texts = load_splits()["test"][0]

    data["error_analysis"] = build(data, texts)
    arguments.out.write_text(json.dumps(data, indent=2) + "\n",
                             encoding="utf-8", newline="\n")

    overlap = data["error_analysis"]["overlap"]
    print(f"  baseline wrong on {overlap['baseline_errors']}, transformer wrong on "
          f"{overlap['transformer_errors']}")
    print(f"  both wrong on {overlap['both_wrong']}, which is "
          f"{overlap['both_wrong'] / overlap['transformer_errors'] * 100:.0f} percent "
          f"of the transformer's errors")
    for name, entry in data["error_analysis"].items():
        if name == "overlap":
            continue
        print(f"\n  {name}: {entry['total_errors']} errors")
        for label, category in entry["categories"].items():
            print(f"    {label:<34} {category['count']:>4}  "
                  f"{category['share_of_errors'] * 100:5.1f} percent of errors")
    print(f"\n  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
