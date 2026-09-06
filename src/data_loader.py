"""
Data Loader Module
==================
Handles dataset loading, tokenization, and DataLoader creation for both
the TF-IDF baseline and the DistilBERT transformer model.

Supports:
    - Hugging Face ``datasets`` library (AG News)
    - Synthetic fallback data for offline development
    - Tokenization with AutoTokenizer
    - TF-IDF vectorization for baseline comparison

Usage:
    from src.data_loader import load_ag_news, create_tokenized_datasets
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# AG News label mapping
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
NUM_LABELS = len(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_ag_news(
    cache_dir: str = "data",
    allow_synthetic: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the AG News dataset via Hugging Face ``datasets``.

    When ``allow_synthetic`` is true, uses synthetic text if the library is
    unavailable or download fails. This option is for offline code development
    only and must not be used as an AG News benchmark.

    Returns:
        (train_df, test_df) each with columns ``text`` and ``label``.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("ag_news", cache_dir=cache_dir)
        train_df = ds["train"].to_pandas()  # type: ignore[union-attr]
        test_df = ds["test"].to_pandas()  # type: ignore[union-attr]
        logger.info("AG News loaded: %d train, %d test", len(train_df), len(test_df))
    except Exception as exc:
        if not allow_synthetic:
            raise RuntimeError(
                "Could not load AG News. Check network access and dataset availability, "
                "or pass --synthetic for offline development only."
            ) from exc
        logger.warning("Could not load AG News (%s). Generating synthetic development data.", exc)
        train_df, test_df = _generate_synthetic_ag_news()

    return train_df, test_df


def _generate_synthetic_ag_news(
    n_train: int = 4000, n_test: int = 800
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic news-like data for offline development."""
    rng = np.random.default_rng(42)

    templates = {
        0: [  # World
            "President of {country} meets with officials to discuss trade agreements",
            "United Nations announces new humanitarian aid program in {country}",
            "Tensions rise in {country} as elections approach amid political unrest",
            "International summit in Geneva addresses climate change and global policy",
            "Diplomatic efforts continue as {country} seeks peace agreement",
        ],
        1: [  # Sports
            "The {team} defeated their rivals in a thrilling overtime match",
            "Olympic athlete breaks world record in the {event} competition",
            "NBA playoffs heat up as {team} advance to the conference finals",
            "FIFA announces new regulations for the upcoming World Cup qualifiers",
            "Tennis star wins Grand Slam title after five-set marathon final",
        ],
        2: [  # Business
            "Stock market reaches all-time high as tech sector leads gains",
            "Major merger announced between {company} and rival corporation",
            "Federal Reserve signals potential interest rate adjustment next quarter",
            "Quarterly earnings report shows {company} exceeding analyst expectations",
            "Startup raises billion-dollar funding round for AI technology platform",
        ],
        3: [  # Sci/Tech
            "Researchers develop new {tech} algorithm that improves performance by 40 percent",
            "NASA announces discovery of potentially habitable exoplanet",
            "New study reveals breakthrough in quantum computing error correction",
            "Tech giant releases open-source {tech} framework for developers",
            "Scientists achieve fusion energy milestone at national laboratory",
        ],
    }

    fillers = {
        "country": ["France", "Japan", "Brazil", "India", "Nigeria", "Germany"],
        "team": ["Lakers", "Warriors", "Eagles", "Manchester United", "Bayern"],
        "event": ["100m sprint", "high jump", "swimming", "gymnastics"],
        "company": ["Apple", "Google", "Amazon", "Microsoft", "Tesla"],
        "tech": ["machine learning", "deep learning", "neural network", "AI"],
    }

    def _make(n: int) -> pd.DataFrame:
        texts, labels = [], []
        for _ in range(n):
            label = int(rng.integers(0, 4))
            tmpl = rng.choice(templates[label])
            for key, vals in fillers.items():
                tmpl = tmpl.replace(f"{{{key}}}", rng.choice(vals))
            texts.append(tmpl)
            labels.append(label)
        return pd.DataFrame({"text": texts, "label": labels})

    train_df = _make(n_train)
    test_df = _make(n_test)
    logger.info(
        "Synthetic AG News generated: %d train, %d test",
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Tokenization (for Transformer)
# ---------------------------------------------------------------------------


def create_tokenized_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    val_size: float = 0.1,
) -> dict[str, Any]:
    """
    Tokenize text data using a Hugging Face AutoTokenizer.

    Returns a dict with keys ``train``, ``val``, ``test``, each a
    Hugging Face ``Dataset`` object (or dict-of-tensors fallback).
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(
        "Tokenizer loaded: %s (vocab_size=%d)", model_name, tokenizer.vocab_size
    )

    # Split off validation set from training data
    train_sub, val_sub = train_test_split(
        train_df, test_size=val_size, random_state=42, stratify=train_df["label"]
    )

    def _tokenize(batch: dict[str, list]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds_train = Dataset.from_pandas(train_sub.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_sub.reset_index(drop=True))
    ds_test = Dataset.from_pandas(test_df.reset_index(drop=True))

    ds_train = ds_train.map(_tokenize, batched=True, batch_size=256)
    ds_val = ds_val.map(_tokenize, batched=True, batch_size=256)
    ds_test = ds_test.map(_tokenize, batched=True, batch_size=256)

    # Set format for PyTorch
    cols = ["input_ids", "attention_mask", "label"]
    ds_train.set_format("torch", columns=cols)
    ds_val.set_format("torch", columns=cols)
    ds_test.set_format("torch", columns=cols)

    logger.info(
        "Tokenized datasets: train=%d, val=%d, test=%d",
        len(ds_train),
        len(ds_val),
        len(ds_test),
    )

    return {
        "train": ds_train,
        "val": ds_val,
        "test": ds_test,
        "tokenizer": tokenizer,
    }


# ---------------------------------------------------------------------------
# TF-IDF feature extraction (for baseline)
# ---------------------------------------------------------------------------


def create_tfidf_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict[str, Any]:
    """
    Build TF-IDF feature matrices for baseline models.

    Returns dict with keys ``X_train``, ``X_test``, ``y_train``, ``y_test``,
    and the fitted ``vectorizer``.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=3,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    logger.info(
        "TF-IDF features: %d train rows, %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": train_df["label"].values,
        "y_test": test_df["label"].values,
        "vectorizer": vectorizer,
    }
