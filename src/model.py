"""
Model Module
============
Defines model architectures for text classification:
    - TF-IDF baselines (Logistic Regression, Linear SVM)
    - DistilBERT fine-tuning via Hugging Face

Usage:
    from src.model import build_baseline_model, build_transformer_model
"""

import logging
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline models (TF-IDF + classical ML)
# ---------------------------------------------------------------------------


def build_baseline_model(
    model_type: str = "logistic_regression",
    **kwargs: Any,
) -> Any:
    """
    Build a scikit-learn baseline classifier.

    Args:
        model_type: One of ``logistic_regression``, ``svm``, ``random_forest``.
        **kwargs: Passed to the sklearn constructor.

    Returns:
        A fitted-ready sklearn estimator.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    defaults: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "multinomial",
        },
        "svm": {"C": 1.0, "max_iter": 2000},
        "random_forest": {"n_estimators": 200, "max_depth": 30, "n_jobs": -1},
    }

    registry = {
        "logistic_regression": LogisticRegression,
        "svm": LinearSVC,
        "random_forest": RandomForestClassifier,
    }

    if model_type not in registry:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from {list(registry)}"
        )

    params = {**defaults.get(model_type, {}), **kwargs}
    model = registry[model_type](**params)
    logger.info("Built baseline model: %s with %s", model_type, params)
    return model


# ---------------------------------------------------------------------------
# Transformer model (DistilBERT)
# ---------------------------------------------------------------------------


def build_transformer_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 4,
    dropout: float = 0.1,
) -> Any:
    """
    Load a pre-trained DistilBERT model for sequence classification.

    Args:
        model_name: Hugging Face model identifier.
        num_labels: Number of output classes.
        dropout: Classifier dropout probability.

    Returns:
        A ``transformers.AutoModelForSequenceClassification`` instance.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        classifier_dropout=dropout,
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Loaded %s: %s total params (%s trainable)",
        model_name,
        f"{total:,}",
        f"{trainable:,}",
    )
    return model


def freeze_backbone(model: Any, unfreeze_last_n: int = 0) -> None:
    """
    Freeze all transformer layers except the classifier head.

    Optionally unfreeze the last *n* transformer layers for partial
    fine-tuning (feature extraction → gradual unfreezing).

    Args:
        model: A Hugging Face model.
        unfreeze_last_n: Number of transformer layers to keep unfrozen
                         (from the top). 0 = only classifier is trainable.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    if hasattr(model, "pre_classifier"):
        for param in model.pre_classifier.parameters():
            param.requires_grad = True

    # Optionally unfreeze last N transformer layers
    if unfreeze_last_n > 0 and hasattr(model, "distilbert"):
        layers = model.distilbert.transformer.layer
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Backbone frozen. Trainable: %s / %s (%.1f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )


def get_model_summary(model: Any) -> Dict[str, Any]:
    """Return a summary dict of model architecture and parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_info = {}
    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        if top_level not in layer_info:
            layer_info[top_level] = {"total": 0, "trainable": 0}
        layer_info[top_level]["total"] += param.numel()
        if param.requires_grad:
            layer_info[top_level]["trainable"] += param.numel()

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "layers": layer_info,
    }
