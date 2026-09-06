"""Regression tests for the two bugs where preprocessing was rebuilt from defaults.

Both had the same shape. A model was loaded from disk and then fed by a fresh
vectoriser or tokeniser constructed from defaults, rather than by the one saved
beside it. The classical case raises, because the vocabulary width differs. The
transformer case does not raise at all, because a tokeniser rebuilt from the base
checkpoint usually agrees with the saved one, so the only symptom would be a
quietly wrong number.

These tests fail if either substitution is reintroduced.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import artifacts


def small_frames():
    train = pd.DataFrame({
        "text": [f"markets moved on {word} news today" for word in
                 ("rates", "trade", "oil", "jobs", "growth")] * 24,
        "label": [0, 1, 2, 3, 0] * 24,
    })
    test = pd.DataFrame({
        "text": [f"a report about {word} was published" for word in
                 ("sport", "science", "trade")] * 8,
        "label": [1, 3, 2] * 8,
    })
    return train, test


class ClassicalPathLoadsItsOwnVectoriser(unittest.TestCase):
    """src/evaluate.py refit a vectoriser instead of loading the saved one."""

    def setUp(self):
        from sklearn.svm import LinearSVC

        from src.data_loader import create_tfidf_features

        self.train, self.test = small_frames()
        built = create_tfidf_features(self.train, self.test)
        self.vectoriser = built["vectorizer"]
        self.model = LinearSVC().fit(built["X_train"], self.train["label"])
        self.create_tfidf_features = create_tfidf_features

    def test_the_saved_vectoriser_produces_a_usable_matrix(self):
        matrix = self.vectoriser.transform(self.test["text"])
        predicted = self.model.predict(matrix)
        self.assertEqual(len(predicted), len(self.test))

    def test_a_vectoriser_refit_on_other_text_breaks_the_model(self):
        """The regression. If evaluate.py ever refits again, this is what happens."""
        other = pd.DataFrame({
            "text": [f"the team won the {word} final last night" for word in
                     ("cup", "league", "derby", "open", "series")] * 24,
            "label": [1] * 120,
        })
        rebuilt = self.create_tfidf_features(other, self.test)["vectorizer"]
        self.assertNotEqual(len(rebuilt.vocabulary_), len(self.vectoriser.vocabulary_))
        with self.assertRaises(ValueError):
            self.model.predict(rebuilt.transform(self.test["text"]))

    def test_evaluate_does_not_build_features_from_the_training_frame(self):
        """Read the source, because the failure it prevents needs a saved model.

        A unit test cannot easily reach the command line path, but the defect was
        a single call, and its absence is checkable.
        """
        source = (Path(__file__).resolve().parents[1] / "src" / "evaluate.py"
                  ).read_text(encoding="utf-8")
        self.assertNotIn("create_tfidf_features(train_df", source)
        self.assertIn("vectorizer.joblib", source)

    def test_evaluate_refuses_when_the_vectoriser_is_missing(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "evaluate.py"
                  ).read_text(encoding="utf-8")
        self.assertIn("No vectorizer at", source)


class TransformerPathLoadsItsOwnTokeniser(unittest.TestCase):
    """The same substitution, in the case that would not have raised."""

    def test_evaluate_tokenises_with_the_saved_model_directory(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "evaluate.py"
                  ).read_text(encoding="utf-8")
        self.assertIn("model_name=args.model_path", source)

    def test_evaluate_does_not_tokenise_from_the_default_checkpoint(self):
        """create_tokenized_datasets defaults to the base checkpoint. Relying on
        that default is the bug: it would agree with the saved tokeniser often
        enough never to raise."""
        source = (Path(__file__).resolve().parents[1] / "src" / "evaluate.py"
                  ).read_text(encoding="utf-8")
        self.assertNotIn("create_tokenized_datasets(train_df, test_df)", source)


class ArtifactDirectories(unittest.TestCase):
    """A model file with no preprocessing beside it is an orphan."""

    def test_a_complete_baseline_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            joblib.dump({"a": 1}, directory / "model.joblib")
            joblib.dump({"b": 2}, directory / "vectorizer.joblib")
            artifacts.write_metadata(directory, {"kind": "baseline"})
            described = artifacts.describe("baseline", directory)
            self.assertEqual(described["kind"], "baseline")
            self.assertEqual(len(described["files"]), 3)
            for entry in described["files"]:
                self.assertEqual(len(entry["sha256_prefix"]), 16)

    def test_a_directory_without_its_vectoriser_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            joblib.dump({"a": 1}, directory / "model.joblib")
            artifacts.write_metadata(directory, {"kind": "baseline"})
            with self.assertRaises(SystemExit) as caught:
                artifacts.require("baseline", directory)
            self.assertIn("vectorizer.joblib", str(caught.exception))

    def test_a_transformer_directory_without_its_tokeniser_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "model.safetensors").write_bytes(b"x")
            (directory / "config.json").write_text("{}", encoding="utf-8")
            artifacts.write_metadata(directory, {"kind": "transformer"})
            with self.assertRaises(SystemExit) as caught:
                artifacts.require("transformer", directory)
            self.assertIn("tokenizer", str(caught.exception))

    def test_the_hash_changes_when_the_bytes_change(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "file.bin"
            path.write_bytes(b"one")
            first = artifacts.digest(path)
            path.write_bytes(b"two")
            self.assertNotEqual(first, artifacts.digest(path))


class RecordedArtifactsMatchTheResults(unittest.TestCase):
    """The results file names the artifacts. Those names must be real."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "results" / "benchmark.json"
        if not path.is_file():
            raise unittest.SkipTest("no recorded results yet")
        cls.data = json.loads(path.read_text(encoding="utf-8"))

    def test_each_recorded_artifact_names_its_preprocessing(self):
        for section, required in (("baselines", "vectorizer.joblib"),
                                  ("transformer", "tokenizer.json")):
            if section not in self.data:
                continue
            with self.subTest(section=section):
                names = {entry["name"]
                         for entry in self.data[section]["artifact"]["files"]}
                self.assertIn(required, names)
                self.assertIn("metadata.json", names)

    def test_recorded_predictions_line_up_with_the_test_split(self):
        if "baselines" not in self.data:
            self.skipTest("no baseline results")
        recorded = self.data["baselines"]["predictions"]
        self.assertEqual(len(recorded["test_true"]), self.data["data"]["test"])
        for name, predicted in recorded["test_predicted"].items():
            with self.subTest(model=name):
                self.assertEqual(len(predicted), self.data["data"]["test"])
                self.assertEqual(set(np.unique(predicted)) - {0, 1, 2, 3}, set())


if __name__ == "__main__":
    unittest.main()


class DatasetIdentifierIsCanonical(unittest.TestCase):
    """A bare Hub alias resolves from a warm cache but not from a clean machine.

    `load_dataset("ag_news")` worked on the machine the results were produced on,
    because the dataset was already downloaded, and failed in continuous
    integration with HfUriError: current huggingface_hub requires namespace/name.
    The failure is invisible to anyone whose cache is already populated, so it is
    pinned here rather than left to the next clean checkout.
    """

    def test_every_load_call_names_a_namespace(self):
        source_directory = Path(__file__).resolve().parents[1] / "src"
        calls = []
        for path in sorted(source_directory.glob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if "load_dataset(" in line and "def " not in line:
                    calls.append((path.name, line.strip()))
        self.assertTrue(calls, "no load_dataset call found to check")
        for name, line in calls:
            with self.subTest(file=name, line=line):
                # Either a literal with a namespace, or the DATASET constant.
                literal = re.search(r'load_dataset\(\s*"([^"]+)"', line)
                if literal:
                    self.assertIn("/", literal.group(1),
                                  "a bare dataset alias will not resolve on a "
                                  "machine without a warm cache")
                else:
                    self.assertIn("DATASET", line)

    def test_the_benchmark_constant_carries_a_namespace(self):
        from src.benchmark import DATASET
        self.assertIn("/", DATASET)
        self.assertTrue(DATASET.endswith("ag_news"))
