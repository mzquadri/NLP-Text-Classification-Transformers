"""Measure what fine-tuning a transformer buys over TF-IDF on AG News.

    python -m src.benchmark --track baselines     # about a minute
    python -m src.benchmark --track transformer   # about half an hour on a GPU
    python -m src.benchmark                       # both

This repository compared "classical machine learning against a fine-tuned
transformer" without ever reporting a number for either. AG News is public and
downloads in seconds, so there was nothing stopping the comparison being run. It
is run here, on the published test split, with the cost of each side recorded
next to its accuracy.

Model choice happens on a validation split carved out of the training data. The
published test split is scored once per model and never used to choose anything,
which is what the repository's own reproducibility note asks for.

Writes results/benchmark.json. The two tracks write into the same file and each
leaves the other's section alone, so the slow one need not be rerun to redo the
fast one.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"

SEED = 42
VAL_FRACTION = 0.1
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")

#: The canonical Hub id, with its namespace. The bare "ag_news" alias resolves
#: from a warm local cache but not from a clean machine: current huggingface_hub
#: requires namespace/name and raises HfUriError otherwise. That failure is
#: invisible to anyone who has already downloaded the dataset, which is why it
#: reached continuous integration before it was noticed. The two ids return
#: byte-identical text and labels for both splits.
DATASET = "fancyzhx/ag_news"

#: DistilBERT settings. Sequence length covers the AG News text at the 99th
#: percentile; padding is dynamic so short items do not pay for the tail.
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 5e-5

#: Repeats used to time classical prediction, reduced to the median.
PREDICT_REPEATS = 7


def load_splits():
    """AG News, with a validation split carved out of the training data."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    data = load_dataset(DATASET)
    texts = np.array(data["train"]["text"])
    labels = np.array(data["train"]["label"])
    train_text, val_text, train_label, val_label = train_test_split(
        texts, labels, test_size=VAL_FRACTION, random_state=SEED, stratify=labels)
    return {
        "train": (train_text, train_label),
        "validation": (val_text, val_label),
        "test": (np.array(data["test"]["text"]), np.array(data["test"]["label"])),
    }


def scores(actual, predicted) -> dict:
    """Accuracy, macro F1, and precision, recall and F1 for each class.

    Aggregate accuracy on a balanced four-class problem hides which class a model
    is failing on, and the confusion matrix is carried alongside because on AG
    News the errors are concentrated in one pair rather than spread evenly.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        actual, predicted, labels=range(len(LABEL_NAMES)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(LABEL_NAMES)
        },
        "per_class_f1": {
            name: float(per_class_f1[index])
            for index, name in enumerate(LABEL_NAMES)},
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=range(len(LABEL_NAMES))).tolist(),
        "confusion_matrix_rows": "true label",
        "confusion_matrix_columns": "predicted label",
        "n": len(actual),
    }


def confusion_pairs(matrix, top: int = 4) -> list[dict]:
    """The off-diagonal cells that account for the most errors."""
    pairs = [
        {"true": LABEL_NAMES[row], "predicted": LABEL_NAMES[column],
         "count": int(matrix[row][column])}
        for row in range(len(LABEL_NAMES))
        for column in range(len(LABEL_NAMES))
        if row != column
    ]
    pairs.sort(key=lambda entry: -entry["count"])
    total = sum(entry["count"] for entry in pairs)
    for entry in pairs:
        entry["share_of_errors"] = round(entry["count"] / total, 4) if total else 0.0
    return pairs[:top]


def contamination(splits) -> dict:
    """Does the published test split overlap the training data?

    Worth checking before trusting any number from a public benchmark, and worth
    recording when the answer is no. A held-out set that shares rows with
    training measures memorisation, and nothing in the reported accuracy would
    say so.
    """
    import re

    def normalise(text: str) -> str:
        text = re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("\\", " "))
        return re.sub(r"\s+", " ", text).strip()

    train_text = splits["train"][0]
    test_text, test_label = splits["test"]

    exact = set(train_text)
    normalised = {}
    for text, label in zip(train_text, splits["train"][1], strict=True):
        normalised.setdefault(normalise(text), []).append(int(label))

    exact_hits = sum(1 for text in test_text if text in exact)
    normalised_hits = sum(1 for text in test_text if normalise(text) in normalised)
    disagreeing = sum(
        1 for text, label in zip(test_text, test_label, strict=True)
        if normalise(text) in normalised
        and int(label) not in normalised[normalise(text)])

    return {
        "test_rows": len(test_text),
        "exact_matches_in_train": exact_hits,
        "normalised_matches_in_train": normalised_hits,
        "normalised_match_fraction": round(normalised_hits / len(test_text), 5),
        "of_those_the_train_label_disagrees": disagreeing,
        "duplicate_rows_within_train_after_normalising": int(
            len(train_text) - len(normalised)),
        "note": "normalising lowercases, strips punctuation and collapses "
                "whitespace, so it catches reformatting that an exact match misses",
    }


def run_baselines(splits) -> dict:
    """TF-IDF and three linear or tree models, selected on validation."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    (train_text, train_label) = splits["train"]
    (val_text, val_label) = splits["validation"]
    (test_text, test_label) = splits["test"]

    started = time.perf_counter()
    vectoriser = TfidfVectorizer(max_features=200_000, ngram_range=(1, 2),
                                 sublinear_tf=True, strip_accents="unicode",
                                 min_df=3)
    x_train = vectoriser.fit_transform(train_text)
    x_val = vectoriser.transform(val_text)
    x_test = vectoriser.transform(test_text)
    vectorise_seconds = time.perf_counter() - started
    print(f"  TF-IDF: {x_train.shape[1]:,} features in {vectorise_seconds:.1f}s")

    builders = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=4.0),
        "linear_svm": lambda: LinearSVC(C=0.5),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    }

    results, models = {}, {}
    for name, build in builders.items():
        started = time.perf_counter()
        model = build().fit(x_train, train_label)
        train_seconds = time.perf_counter() - started
        models[name] = model

        # Timed several times and reduced to the median. A single measurement of
        # the SVM is unreliable: predicting 7,600 documents from a precomputed
        # sparse matrix takes a few milliseconds in total, so ordinary scheduling
        # noise moved the reported cost multiple by nearly half between two runs
        # that produced identical predictions.
        predicted_test = model.predict(x_test)
        timings = []
        for _ in range(PREDICT_REPEATS):
            started = time.perf_counter()
            model.predict(x_test)
            timings.append(time.perf_counter() - started)
        predict_seconds = float(np.median(timings))
        throughput = [len(test_label) / value for value in timings]

        results[name] = {
            "validation": scores(val_label, model.predict(x_val)),
            "test": scores(test_label, predicted_test),
            "confusion_pairs": confusion_pairs(
                scores(test_label, predicted_test)["confusion_matrix"]),
            "train_seconds": round(train_seconds, 2),
            "vectorise_seconds": round(vectorise_seconds, 2),
            "total_fit_seconds": round(train_seconds + vectorise_seconds, 2),
            "milliseconds_per_document": round(
                predict_seconds / len(test_label) * 1000, 5),
            "documents_per_second": round(float(np.median(throughput))),
            "documents_per_second_slowest": round(min(throughput)),
            "documents_per_second_fastest": round(max(throughput)),
            "prediction_timing_repeats": PREDICT_REPEATS,
            "timing_note": "model forward only, on an already vectorised matrix. "
                           "Predicting 7,600 rows from a sparse matrix takes a few "
                           "milliseconds in total, close enough to the timer floor "
                           "that repeated measurement spans a threefold range. The "
                           "spread is recorded rather than hidden behind a median",
            "_predicted_test": predicted_test.tolist(),
        }
        print(f"  {name:<20} val {results[name]['validation']['accuracy'] * 100:5.2f}"
              f"   test {results[name]['test']['accuracy'] * 100:5.2f}"
              f"   macro F1 {results[name]['test']['macro_f1'] * 100:5.2f}"
              f"   fit {train_seconds:6.1f}s")

    majority = np.bincount(test_label).max() / len(test_label)
    chosen = max(results, key=lambda name: results[name]["validation"]["accuracy"])
    print(f"  validation chooses {chosen}")

    # The chosen model is saved together with the vectoriser that produced its
    # features. Saving one without the other is what made src/evaluate.py refit a
    # fresh vectoriser and score the model against a different vocabulary.
    import joblib

    from .artifacts import BASELINE_DIR, describe, write_metadata

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[chosen], BASELINE_DIR / "model.joblib")
    joblib.dump(vectoriser, BASELINE_DIR / "vectorizer.joblib")
    write_metadata(BASELINE_DIR, {
        "kind": "baseline",
        "estimator": type(models[chosen]).__name__,
        "estimator_name": chosen,
        "vectoriser": type(vectoriser).__name__,
        "vectoriser_parameters": {
            "max_features": 200_000, "ngram_range": [1, 2], "sublinear_tf": True,
            "strip_accents": "unicode", "min_df": 3,
        },
        "features": int(x_train.shape[1]),
        "trained_on": {"dataset": DATASET, "split": "train",
                       "rows": len(train_label), "seed": SEED},
        "labels": list(LABEL_NAMES),
        "note": "model.joblib and vectorizer.joblib belong together; the "
                "classifier is indexed by this vectoriser's vocabulary",
    })
    artifact = describe("baseline", BASELINE_DIR)
    print(f"  saved {artifact['directory']} "
          f"({artifact['total_bytes'] / 1e6:.1f} MB)")

    predictions = {
        "test_true": test_label.tolist(),
        "test_predicted": {name: results[name].pop("_predicted_test")
                           for name in results},
    }

    return {
        "models": results,
        "chosen_on_validation": chosen,
        "features": int(x_train.shape[1]),
        "most_frequent_class_accuracy": float(majority),
        "artifact": artifact,
        "predictions": predictions,
        "vectoriser": {
            "max_features": 200_000, "ngram_range": [1, 2], "sublinear_tf": True,
            "strip_accents": "unicode", "min_df": 3,
            "vocabulary_size": int(x_train.shape[1]),
        },
    }


def run_transformer(splits) -> dict:
    """Fine-tune DistilBERT, choosing the epoch on validation."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    device = torch.device(
        "xpu" if getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    torch.manual_seed(SEED)
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokeniser)

    def encode(texts, labels):
        encoded = tokeniser(list(texts), truncation=True, max_length=MAX_LENGTH)
        return [{"input_ids": encoded["input_ids"][index],
                 "attention_mask": encoded["attention_mask"][index],
                 "labels": int(labels[index])}
                for index in range(len(labels))]

    loaders = {
        name: DataLoader(encode(*splits[name]), batch_size=BATCH_SIZE,
                         shuffle=(name == "train"), collate_fn=collator)
        for name in ("train", "validation", "test")
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loaders["train"]) * EPOCHS
    schedule = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    def predict(loader):
        model.eval()
        predicted, actual = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                predicted.append(model(**batch).logits.argmax(-1).cpu().numpy())
                actual.append(labels.numpy())
        return np.concatenate(actual), np.concatenate(predicted)

    history, best = [], None
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimiser.zero_grad()
            model(**batch).loss.backward()
            optimiser.step()
            schedule.step()
            if step % 200 == 0:
                print(f"    epoch {epoch} step {step}/{len(loaders['train'])}",
                      flush=True)

        validation = scores(*predict(loaders["validation"]))
        history.append({"epoch": epoch, "validation": validation,
                        "elapsed_seconds": round(time.perf_counter() - started, 1)})
        print(f"  epoch {epoch}: validation accuracy "
              f"{validation['accuracy'] * 100:.2f}")

        # The epoch is chosen on validation. The test split is scored once, after.
        if best is None or validation["accuracy"] > best["validation"]["accuracy"]:
            best = history[-1]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    train_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model.to(device)

    started = time.perf_counter()
    actual, predicted = predict(loaders["test"])
    predict_seconds = time.perf_counter() - started

    # The fine-tuned weights and the tokeniser that produced their inputs are
    # saved into one directory. Evaluating a transformer with a tokeniser rebuilt
    # from the base checkpoint usually agrees for DistilBERT, so a mismatch would
    # never raise; it would only be a wrong number.
    from .artifacts import TRANSFORMER_DIR, describe, write_metadata

    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    model.config.id2label = dict(enumerate(LABEL_NAMES))
    model.config.label2id = {name: index for index, name in enumerate(LABEL_NAMES)}
    model.save_pretrained(TRANSFORMER_DIR)
    tokeniser.save_pretrained(TRANSFORMER_DIR)

    peak_bytes = None
    if device.type == "xpu":
        peak_bytes = int(torch.xpu.max_memory_allocated())
    elif device.type == "cuda":
        peak_bytes = int(torch.cuda.max_memory_allocated())

    parameters = sum(p.numel() for p in model.parameters())
    configuration = {
        "base_checkpoint": MODEL_NAME,
        "tokenizer": MODEL_NAME,
        "head": "sequence classification, 4 labels, randomly initialised",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "optimizer_defaults": {
            "betas": list(optimiser.defaults["betas"]),
            "eps": optimiser.defaults["eps"],
            "weight_decay": optimiser.defaults["weight_decay"],
        },
        "scheduler": "LinearLR, factor 1.0 to 0.0 over all steps",
        "max_sequence_length": MAX_LENGTH,
        "padding": "dynamic, to the longest item in each batch",
        "seed": SEED,
        "total_optimiser_steps": total_steps,
    }

    write_metadata(TRANSFORMER_DIR, {
        "kind": "transformer",
        "configuration": configuration,
        "trained_on": {"dataset": DATASET, "split": "train",
                       "rows": len(splits["train"][1])},
        "epoch_chosen_on_validation": best["epoch"],
        "labels": list(LABEL_NAMES),
        "note": "the tokenizer files in this directory belong to these weights; "
                "evaluating with a tokeniser rebuilt from the base checkpoint "
                "would not raise, it would only be wrong",
    })
    artifact = describe("transformer", TRANSFORMER_DIR)
    print(f"  saved {artifact['directory']} "
          f"({artifact['total_bytes'] / 1e6:.0f} MB)")

    return {
        "model": MODEL_NAME,
        "device": str(device),
        "device_name": (torch.xpu.get_device_name(0) if device.type == "xpu"
                        else torch.cuda.get_device_name(0) if device.type == "cuda"
                        else platform.processor() or "cpu"),
        "configuration": configuration,
        "epochs_run": EPOCHS,
        "epoch_chosen_on_validation": best["epoch"],
        "history": history,
        "validation": best["validation"],
        "test": scores(actual, predicted),
        "confusion_pairs": confusion_pairs(scores(actual, predicted)["confusion_matrix"]),
        "train_seconds": round(train_seconds, 1),
        "milliseconds_per_document": round(
            predict_seconds / len(actual) * 1000, 4),
        "documents_per_second": round(len(actual) / predict_seconds, 1),
        "parameters": int(parameters),
        "peak_device_memory_bytes": peak_bytes,
        "artifact": artifact,
        "predictions": {"test_true": actual.tolist(),
                        "test_predicted": predicted.tolist()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["baselines", "transformer", "both"],
                        default="both")
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    payload = {}
    if arguments.out.exists():
        payload = json.loads(arguments.out.read_text(encoding="utf-8"))

    print("  loading AG News")
    splits = load_splits()
    print(f"  train {len(splits['train'][1]):,}   "
          f"validation {len(splits['validation'][1]):,}   "
          f"test {len(splits['test'][1]):,}")

    payload["environment"] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
    }
    payload["data"] = {
        "dataset": DATASET,
        "source": "Hugging Face Datasets, the published splits",
        "classes": list(LABEL_NAMES),
        "train": len(splits["train"][1]),
        "validation": len(splits["validation"][1]),
        "test": len(splits["test"][1]),
        "seed": SEED,
        "note": "the validation split is carved out of the training data; the "
                "published test split is scored once per model and chooses nothing",
    }

    if "contamination" not in payload or arguments.track in ("baselines", "both"):
        print("\n  checking whether the test split overlaps training")
        payload["contamination"] = contamination(splits)
        found = payload["contamination"]
        print(f"    exact matches {found['exact_matches_in_train']}, "
              f"normalised {found['normalised_matches_in_train']} "
              f"({found['normalised_match_fraction'] * 100:.2f} percent)")

    if arguments.track in ("baselines", "both"):
        print("\n  TF-IDF track")
        payload["baselines"] = run_baselines(splits)

    if arguments.track in ("transformer", "both"):
        print("\n  DistilBERT track")
        payload["transformer"] = run_transformer(splits)

    if "baselines" in payload and "transformer" in payload:
        best_baseline = payload["baselines"]["chosen_on_validation"]
        baseline = payload["baselines"]["models"][best_baseline]
        transformer = payload["transformer"]
        payload["comparison"] = {
            "note": "the comparison the README leads with is DistilBERT against "
                    "the strongest classical baseline, which validation selected "
                    "before any test score was looked at",
            "baseline_chosen_on_validation": best_baseline,
            "baseline_test_macro_f1": baseline["test"]["macro_f1"],
            "transformer_test_macro_f1": transformer["test"]["macro_f1"],
            "macro_f1_gain_points": round(
                (transformer["test"]["macro_f1"] - baseline["test"]["macro_f1"])
                * 100, 3),
            "baseline_artifact_bytes": payload["baselines"]["artifact"]["total_bytes"],
            "transformer_artifact_bytes": transformer["artifact"]["total_bytes"],
            "model_size_multiple": round(
                transformer["artifact"]["total_bytes"]
                / payload["baselines"]["artifact"]["total_bytes"], 1),
            "baseline_test_accuracy": baseline["test"]["accuracy"],
            "transformer_test_accuracy": transformer["test"]["accuracy"],
            "accuracy_gain_points": round(
                (transformer["test"]["accuracy"] - baseline["test"]["accuracy"])
                * 100, 3),
            "errors_removed_fraction": round(
                1 - (1 - transformer["test"]["accuracy"])
                / (1 - baseline["test"]["accuracy"]), 4),
            "training_time_multiple": round(
                transformer["train_seconds"] / baseline["total_fit_seconds"], 1),
            # Reported as a range. The classical side is close enough to the
            # timer floor that a single figure, a median, or even the order of
            # magnitude moved between identical runs. What is stable is that it
            # is thousands of times faster, so that is what is recorded.
            "baseline_documents_per_second": baseline["documents_per_second"],
            "baseline_documents_per_second_range": [
                baseline["documents_per_second_slowest"],
                baseline["documents_per_second_fastest"]],
            "transformer_documents_per_second": transformer["documents_per_second"],
            "inference_ratio_range": [
                round(baseline["documents_per_second_slowest"]
                      / transformer["documents_per_second"]),
                round(baseline["documents_per_second_fastest"]
                      / transformer["documents_per_second"])],
        }
        summary = payload["comparison"]
        print(f"\n  {best_baseline} {baseline['test']['accuracy'] * 100:.2f} against "
              f"DistilBERT {transformer['test']['accuracy'] * 100:.2f}")
        print(f"  {summary['accuracy_gain_points']:+.2f} points, removing "
              f"{summary['errors_removed_fraction'] * 100:.1f} percent of the errors")
        low, high = summary["inference_ratio_range"]
        print(f"  for {summary['training_time_multiple']}x the training time, and "
              f"{low:,} to {high:,} times the throughput")
        print(f"  {summary['baseline_documents_per_second']:,} documents per second "
              f"against {summary['transformer_documents_per_second']:,}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"\n  wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
