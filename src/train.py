"""
Training Module
===============
Provides training routines for both baseline (scikit-learn) and transformer
(Hugging Face Trainer) models.

Usage:
    python src/train.py --model distilbert --epochs 3 --batch_size 32
    python src/train.py --model baseline --baseline_type logistic_regression
"""

import argparse
import json
import logging
import os
import time
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------


def train_baseline(
    model: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """
    Train a scikit-learn baseline model and evaluate on test set.

    Returns:
        (fitted_model, metrics_dict)
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import cross_val_score

    logger.info("Training baseline model on %d samples ...", X_train.shape[0])

    # Cross-validation
    t0 = time.time()
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    cv_time = time.time() - t0
    logger.info(
        "CV accuracy: %.4f ± %.4f (%.1fs)", cv_scores.mean(), cv_scores.std(), cv_time
    )

    # Final fit
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "train_time_seconds": round(train_time, 2),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    logger.info(
        "Baseline accuracy: %.4f | F1 (macro): %.4f | train: %.1fs",
        metrics["accuracy"],
        metrics["f1_macro"],
        train_time,
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Transformer training
# ---------------------------------------------------------------------------


def train_transformer(
    model: Any,
    tokenized_datasets: dict[str, Any],
    output_dir: str = "models/distilbert-ag-news",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> tuple[Any, dict[str, Any]]:
    """
    Fine-tune a transformer model using the Hugging Face Trainer API.

    Returns:
        (trained_model, metrics_dict)
    """
    from transformers import Trainer, TrainingArguments

    from src.evaluate import compute_metrics_fn

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics_fn,
    )

    logger.info(
        "Starting transformer training: %d epochs, batch_size=%d, lr=%s",
        num_epochs,
        batch_size,
        learning_rate,
    )
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    # Evaluate on test set
    test_metrics = trainer.evaluate(tokenized_datasets["test"])

    metrics = {
        "train_loss": train_result.training_loss,
        "train_time_seconds": round(train_time, 2),
        "eval_accuracy": test_metrics.get("eval_accuracy", 0),
        "eval_f1_macro": test_metrics.get("eval_f1_macro", 0),
        "eval_f1_weighted": test_metrics.get("eval_f1_weighted", 0),
        "eval_loss": test_metrics.get("eval_loss", 0),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenized_datasets["tokenizer"].save_pretrained(output_dir)
    logger.info("Model saved → %s", output_dir)

    logger.info(
        "Transformer accuracy: %.4f | F1 (macro): %.4f | train: %.1fs",
        metrics["eval_accuracy"],
        metrics["eval_f1_macro"],
        train_time,
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_baseline_model(
    model: Any,
    metrics: dict[str, Any],
    path: str = "models/baseline",
) -> None:
    """Save baseline model and metrics to disk."""
    import joblib

    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.joblib"))
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f,
            indent=2,
        )
    logger.info("Baseline model saved → %s", path)


def save_vectorizer(
    vectorizer: Any, path: str = "models/baseline/vectorizer.joblib"
) -> None:
    """Save TF-IDF vectorizer for inference."""
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info("Vectorizer saved → %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text classification models")
    parser.add_argument(
        "--model", choices=["baseline", "distilbert"], default="distilbert"
    )
    parser.add_argument(
        "--baseline_type",
        default="logistic_regression",
        choices=["logistic_regression", "svm", "random_forest"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic development data if AG News cannot be loaded; not a benchmark.",
    )
    args = parser.parse_args()

    from src.data_loader import (
        create_tfidf_features,
        create_tokenized_datasets,
        load_ag_news,
    )

    train_df, test_df = load_ag_news(allow_synthetic=args.synthetic)

    if args.model == "baseline":
        from src.model import build_baseline_model

        tfidf_data = create_tfidf_features(
            train_df, test_df, max_features=args.max_features
        )
        model = build_baseline_model(args.baseline_type)
        model, metrics = train_baseline(
            model,
            tfidf_data["X_train"],
            tfidf_data["y_train"],
            tfidf_data["X_test"],
            tfidf_data["y_test"],
        )
        save_baseline_model(model, metrics)
        save_vectorizer(tfidf_data["vectorizer"])

    else:
        from src.model import build_transformer_model

        tok_data = create_tokenized_datasets(train_df, test_df)
        model = build_transformer_model(num_labels=4)
        model, metrics = train_transformer(
            model,
            tok_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    logger.info(
        "Training complete.\n%s",
        json.dumps(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
