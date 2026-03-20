"""
Inference Module
================
Production-ready prediction pipeline for both baseline and transformer models.
Supports single-text and batch prediction with confidence scores.

Usage:
    python src/inference.py --text "Apple unveils new MacBook with M4 chip"
    python src/inference.py --model_type baseline --text "Stock market surges"
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# ---------------------------------------------------------------------------
# Baseline inference
# ---------------------------------------------------------------------------


class BaselinePredictor:
    """Inference pipeline for TF-IDF + sklearn models."""

    def __init__(
        self,
        model_path: str = "models/baseline/model.joblib",
        vectorizer_path: str = "models/baseline/vectorizer.joblib",
    ) -> None:
        import joblib

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info("Baseline model loaded from %s", model_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict class for a single text."""
        t0 = time.perf_counter()
        X = self.vectorizer.transform([text])
        label_id = int(self.model.predict(X)[0])
        latency = (time.perf_counter() - t0) * 1000

        # Confidence from decision function or probability
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)[0]
            # Softmax-like normalization
            exp_scores = np.exp(scores - np.max(scores))
            confidence = float(np.max(exp_scores / exp_scores.sum()))

        return {
            "text": text,
            "label": LABEL_NAMES[label_id],
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency, 3),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict classes for multiple texts."""
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# Transformer inference
# ---------------------------------------------------------------------------


class TransformerPredictor:
    """Inference pipeline for fine-tuned DistilBERT."""

    def __init__(
        self,
        model_path: str = "models/distilbert-ag-news",
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()
        logger.info("Transformer model loaded on %s from %s", self.device, model_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict class for a single text with confidence scores."""
        import torch

        t0 = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label_id = int(np.argmax(probs))
        latency = (time.perf_counter() - t0) * 1000

        return {
            "text": text,
            "label": LABEL_NAMES[label_id],
            "label_id": label_id,
            "confidence": round(float(probs[label_id]), 4),
            "all_probabilities": {
                LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
            "latency_ms": round(latency, 3),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict classes for multiple texts (sequential for simplicity)."""
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def get_predictor(
    model_type: str = "transformer",
    model_path: Optional[str] = None,
) -> Union[BaselinePredictor, TransformerPredictor]:
    """
    Factory to get the appropriate predictor.

    Args:
        model_type: ``"baseline"`` or ``"transformer"``.
        model_path: Path to model directory.
    """
    if model_type == "baseline":
        path = model_path or "models/baseline/model.joblib"
        vec_path = os.path.join(os.path.dirname(path), "vectorizer.joblib")
        return BaselinePredictor(model_path=path, vectorizer_path=vec_path)
    else:
        path = model_path or "models/distilbert-ag-news"
        return TransformerPredictor(model_path=path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on text")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument(
        "--model_type", choices=["baseline", "transformer"], default="transformer"
    )
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    predictor = get_predictor(args.model_type, args.model_path)
    result = predictor.predict(args.text)

    print(f"\nInput:      {result['text']}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Latency:    {result['latency_ms']:.1f} ms")

    if "all_probabilities" in result:
        print("\nAll probabilities:")
        for cls, prob in result["all_probabilities"].items():
            bar = "█" * int(prob * 40)
            print(f"  {cls:12s} {prob:.4f} {bar}")


if __name__ == "__main__":
    main()
