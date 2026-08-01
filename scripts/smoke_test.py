"""Exercise the offline TF-IDF baseline without downloading external models or data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import _generate_synthetic_ag_news, create_tfidf_features
from src.model import build_baseline_model
from src.train import train_baseline


def main() -> None:
    train_df, test_df = _generate_synthetic_ag_news(n_train=80, n_test=40)
    features = create_tfidf_features(train_df, test_df, max_features=100)
    model = build_baseline_model("logistic_regression")
    _, metrics = train_baseline(
        model,
        features["X_train"],
        features["y_train"],
        features["X_test"],
        features["y_test"],
    )
    if not 0.0 <= metrics["accuracy"] <= 1.0:
        raise SystemExit("Baseline accuracy is outside the valid range.")
    print("Baseline smoke test passed.")


if __name__ == "__main__":
    main()
