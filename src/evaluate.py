"""
Evaluation Module
=================
Comprehensive evaluation utilities: classification metrics, confusion matrix,
per-class analysis, and model comparison.

Usage:
    python src/evaluate.py --model_path models/distilbert-ag-news
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# ---------------------------------------------------------------------------
# Hugging Face Trainer compute_metrics callback
# ---------------------------------------------------------------------------


def compute_metrics_fn(eval_pred: Any) -> Dict[str, float]:
    """
    Compute metrics for the Hugging Face Trainer.

    Used as the ``compute_metrics`` argument to ``Trainer``.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
    }


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------


class TextClassificationEvaluator:
    """Evaluate a trained model with detailed metrics and plots."""

    def __init__(self, label_names: Optional[List[str]] = None) -> None:
        self.label_names = label_names or LABEL_NAMES

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute full evaluation metrics."""
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=self.label_names, output_dict=True
            ),
        }

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.label_names):
            mask = y_true == i
            if mask.sum() > 0:
                per_class[name] = {
                    "support": int(mask.sum()),
                    "accuracy": float((y_pred[mask] == i).mean()),
                    "f1": float(f1_score(y_true == i, y_pred == i, average="binary")),
                }
        metrics["per_class"] = per_class

        logger.info(
            "Accuracy: %.4f | F1 (macro): %.4f | F1 (weighted): %.4f",
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["f1_weighted"],
        )
        return metrics

    def measure_inference_latency(
        self,
        predict_fn: Any,
        sample_inputs: list,
        n_runs: int = 3,
    ) -> Dict[str, float]:
        """Measure average inference latency per sample."""
        times = []
        for _ in range(n_runs):
            for inp in sample_inputs:
                t0 = time.perf_counter()
                predict_fn(inp)
                times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": round(float(np.mean(times)), 3),
            "median_ms": round(float(np.median(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "total_samples": len(times),
        }

    # ---- Plots -----------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        """Plot a confusion matrix heatmap."""
        fmt = ".2f" if normalize else "d"
        norm = "true" if normalize else None
        cm = confusion_matrix(y_true, y_pred, normalize=norm)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved → %s", save_path)
        plt.close(fig)

    def plot_per_class_metrics(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart of per-class F1, precision, recall."""
        report = metrics["classification_report"]
        classes = self.label_names
        f1s = [report[c]["f1-score"] for c in classes]
        precs = [report[c]["precision"] for c in classes]
        recs = [report[c]["recall"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, precs, width, label="Precision", color="#4C72B0")
        ax.bar(x, recs, width, label="Recall", color="#55A868")
        ax.bar(x + width, f1s, width, label="F1-Score", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.legend()
        ax.set_ylim(0.7, 1.0)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Per-class metrics plot saved → %s", save_path)
        plt.close(fig)

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart comparing accuracy and F1 across models."""
        models = list(results.keys())
        accs = [results[m].get("accuracy", 0) for m in models]
        f1s = [results[m].get("f1_macro", 0) for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
        ax.bar(x + width / 2, f1s, width, label="F1 (macro)", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_ylim(0.8, 1.0)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Model comparison plot saved → %s", save_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_evaluation_report(
    metrics: Dict[str, Any],
    output_dir: str = "results",
) -> str:
    """Save evaluation metrics as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "evaluation_report.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Evaluation report saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_path", type=str, default="models/distilbert-ag-news")
    parser.add_argument(
        "--model_type", choices=["baseline", "transformer"], default="transformer"
    )
    args = parser.parse_args()

    from src.data_loader import create_tfidf_features, load_ag_news

    train_df, test_df = load_ag_news()
    evaluator = TextClassificationEvaluator()

    if args.model_type == "baseline":
        import joblib

        model = joblib.load(os.path.join(args.model_path, "model.joblib"))
        tfidf_data = create_tfidf_features(train_df, test_df)
        y_pred = model.predict(tfidf_data["X_test"])
        y_true = tfidf_data["y_test"]
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

        from src.data_loader import create_tokenized_datasets

        tok_data = create_tokenized_datasets(train_df, test_df)

        from transformers import Trainer

        trainer = Trainer(model=model)
        preds = trainer.predict(tok_data["test"])
        y_pred = np.argmax(preds.predictions, axis=-1)
        y_true = test_df["label"].values

    metrics = evaluator.evaluate(y_true, y_pred)
    evaluator.plot_confusion_matrix(
        y_true, y_pred, save_path="results/confusion_matrix.png"
    )
    evaluator.plot_per_class_metrics(metrics, save_path="results/per_class_metrics.png")
    generate_evaluation_report(metrics)


if __name__ == "__main__":
    main()
