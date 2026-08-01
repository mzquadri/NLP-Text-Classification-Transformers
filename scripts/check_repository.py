"""Check that documented source and notebook files are present and parseable."""

from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "src/data_loader.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/inference.py",
    "src/utils.py",
    "notebooks/01_EDA_and_Preprocessing.ipynb",
    "notebooks/02_Training_and_Evaluation.ipynb",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")
    for source in (ROOT / "src").glob("*.py"):
        py_compile.compile(source, doraise=True)
    print(f"Repository check passed: {len(REQUIRED_FILES)} required artifacts available.")


if __name__ == "__main__":
    main()
