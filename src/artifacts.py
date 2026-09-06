"""Where trained artifacts live, and what has to travel with them.

A saved classifier is not usable on its own. It is indexed by the vocabulary or
the token ids that produced it, so a model file without its preprocessing is a
model that will either raise or, worse, score against columns meaning something
else. `src/evaluate.py` did exactly that: it loaded a saved classifier and fitted
a fresh TF-IDF vectoriser to feed it.

Every artifact directory here therefore holds the model, the preprocessing that
belongs to it, and a metadata file naming both. `describe` records what is
actually on disk so the results file can point at the files that produced it,
and `require` refuses to load a directory that is missing a piece.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

BASELINE_DIR = MODELS / "tfidf-linear-svm"
TRANSFORMER_DIR = MODELS / "distilbert-ag-news"

#: What each kind of artifact directory must contain to be loadable. The
#: preprocessing entry is the point: a model file on its own is an orphan.
REQUIRED = {
    "baseline": ("model.joblib", "vectorizer.joblib", "metadata.json"),
    "transformer": ("model.safetensors", "config.json", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json",
                    "metadata.json"),
}


def digest(path: Path, limit: int = 1 << 20) -> str:
    """A short content hash, so a results file can name the exact bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(limit):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def require(kind: str, path: Path) -> None:
    """Refuse to proceed against an artifact directory that is missing a piece."""
    missing = [name for name in REQUIRED[kind] if not (path / name).is_file()]
    if missing:
        raise SystemExit(
            f"{path} is not a complete {kind} artifact: missing "
            f"{', '.join(missing)}. Evaluating a model without the preprocessing "
            f"it was trained with scores it against a different feature space."
        )


def describe(kind: str, path: Path) -> dict:
    """What is on disk, named and hashed, for the results file to point at."""
    require(kind, path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    # Recorded relative to the repository when it lives there, which is the case
    # for a real run. A directory elsewhere, as in a test, keeps its own name.
    try:
        where = path.relative_to(ROOT).as_posix()
    except ValueError:
        where = path.name
    return {
        "kind": kind,
        "directory": where,
        "total_bytes": directory_size(path),
        "files": [
            {
                "name": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256_prefix": digest(item),
            }
            for item in files
        ],
    }


def write_metadata(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_metadata(path: Path) -> dict:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))
