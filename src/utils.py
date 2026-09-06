"""
Utilities Module
================
Configuration management, reproducibility helpers, and logging setup.

Usage:
    from src.utils import set_seed, load_config, get_device
"""

import logging
import os
import random
from typing import Any

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def get_device(prefer_gpu: bool = True) -> str:
    """
    Detect the best available compute device.

    Returns:
        ``"cuda"`` if GPU available and preferred, else ``"cpu"``.
    """
    try:
        import torch

        if prefer_gpu and torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info("Using GPU: %s (%.1f GB)", gpu_name, gpu_mem)
        else:
            device = "cpu"
            logger.info("Using CPU")
    except ImportError:
        device = "cpu"
        logger.info("PyTorch not available. Using CPU.")
    return device


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "dataset": "fancyzhx/ag_news",
        "max_length": 128,
        "val_size": 0.1,
        "max_tfidf_features": 50000,
    },
    "model": {
        "transformer": "distilbert-base-uncased",
        "num_labels": 4,
        "dropout": 0.1,
    },
    "training": {
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed": 42,
    },
    "baseline": {
        "model_type": "logistic_regression",
        "tfidf_max_features": 50000,
        "tfidf_ngram_range": [1, 2],
    },
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file, falling back to defaults.

    Args:
        path: Path to YAML config file. If ``None``, uses defaults.
    """
    config = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            user_config = yaml.safe_load(f)
        if user_config:
            _deep_update(config, user_config)
        logger.info("Config loaded from %s", path)
    else:
        logger.info("Using default configuration")
    return config


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update a nested dictionary."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging with optional file output."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def count_parameters(model: Any) -> dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_number(n: int) -> str:
    """Format large numbers with suffixes (e.g. 66M, 1.2K)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
