"""Tests for the ways a reported score here could be measuring the wrong thing.

Most of these run on small in-memory frames rather than AG News, so the suite
needs no network and finishes in seconds. The one test that does need the real
dataset skips itself if it cannot be reached, rather than failing for a reason
that has nothing to do with the code.

Each check that something is clean is paired with one that the same probe fires
when it is not. A suite of only the first kind would pass on a probe that always
answered "no problem", and the contamination check in particular returns a clean
result on AG News, so its ability to return a dirty one has to be demonstrated
separately.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.benchmark import LABEL_NAMES, contamination, scores


def frame(texts, labels):
    return (np.array(texts), np.array(labels))


class ContaminationProbe(unittest.TestCase):
    """The probe has to be able to find contamination before its silence means anything."""

    def test_it_reports_none_when_the_splits_are_disjoint(self):
        splits = {
            "train": frame([f"story number {i} about markets" for i in range(50)],
                           [0] * 50),
            "test": frame([f"different story {i} about sport" for i in range(20)],
                          [1] * 20),
        }
        found = contamination(splits)
        self.assertEqual(found["exact_matches_in_train"], 0)
        self.assertEqual(found["normalised_matches_in_train"], 0)

    def test_it_finds_a_row_copied_from_train_into_test(self):
        shared = "the central bank raised rates again this morning"
        splits = {
            "train": frame([shared] + [f"other {i}" for i in range(49)], [0] * 50),
            "test": frame([shared] + [f"unseen {i}" for i in range(19)], [0] * 20),
        }
        found = contamination(splits)
        self.assertEqual(found["exact_matches_in_train"], 1)
        self.assertEqual(found["normalised_matches_in_train"], 1)

    def test_it_finds_a_row_that_only_matches_after_normalising(self):
        """Reformatting is what an exact-match check misses."""
        splits = {
            "train": frame(["The Central Bank RAISED rates, again!"]
                           + [f"other {i}" for i in range(49)], [0] * 50),
            "test": frame(["the central bank raised rates again"]
                          + [f"unseen {i}" for i in range(19)], [0] * 20),
        }
        found = contamination(splits)
        self.assertEqual(found["exact_matches_in_train"], 0)
        self.assertEqual(found["normalised_matches_in_train"], 1)

    def test_it_notices_when_a_shared_row_carries_a_different_label(self):
        shared = "shares fell after the announcement"
        splits = {
            "train": frame([shared] + [f"other {i}" for i in range(49)],
                           [2] + [0] * 49),
            "test": frame([shared] + [f"unseen {i}" for i in range(19)],
                          [3] + [1] * 19),
        }
        self.assertEqual(
            contamination(splits)["of_those_the_train_label_disagrees"], 1)


class Splits(unittest.TestCase):
    def setUp(self):
        from src.data_loader import create_tfidf_features

        self.create_tfidf_features = create_tfidf_features
        self.train = pd.DataFrame({
            "text": [f"markets moved on {word} news" for word in
                     ("rates", "trade", "oil", "jobs", "growth")] * 20,
            "label": [0, 1, 2, 3, 0] * 20,
        })
        self.test = pd.DataFrame({
            "text": [f"a report on {word} appeared" for word in
                     ("sport", "science")] * 10,
            "label": [1, 3] * 10,
        })

    def test_the_vectoriser_is_fitted_on_training_text_only(self):
        """A word that appears only in test must not enter the vocabulary."""
        test = self.test.copy()
        test.loc[0, "text"] = "quokka quokka quokka quokka"
        built = self.create_tfidf_features(self.train, test)
        self.assertNotIn("quokka", built["vectorizer"].vocabulary_)

    def test_a_vectoriser_fitted_elsewhere_produces_a_different_width(self):
        """Why src/evaluate.py must load the saved one rather than refit.

        The saved model's coefficients are indexed by the vocabulary it was
        trained with. Refitting on any other frame gives a different vocabulary,
        and the model is then handed columns that mean something else.
        """
        other = pd.DataFrame({
            "text": [f"the team won the {word} final" for word in
                     ("cup", "league", "derby", "open", "series")] * 20,
            "label": [1] * 100,
        })
        first = self.create_tfidf_features(self.train, self.test)
        second = self.create_tfidf_features(other, self.test)
        self.assertNotEqual(first["X_train"].shape[1], second["X_train"].shape[1])

        from sklearn.svm import LinearSVC
        model = LinearSVC().fit(first["X_train"], self.train["label"])
        with self.assertRaises(ValueError):
            model.predict(second["X_test"])


class Scores(unittest.TestCase):
    def test_a_perfect_prediction_scores_one(self):
        actual = np.array([0, 1, 2, 3] * 10)
        result = scores(actual, actual)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)

    def test_always_predicting_one_class_scores_the_base_rate(self):
        actual = np.array([0, 1, 2, 3] * 10)
        result = scores(actual, np.zeros_like(actual))
        self.assertAlmostEqual(result["accuracy"], 0.25)
        self.assertLess(result["macro_f1"], 0.25)

    def test_every_class_is_reported(self):
        actual = np.array([0, 1, 2, 3] * 10)
        self.assertEqual(tuple(scores(actual, actual)["per_class_f1"]), LABEL_NAMES)


class AgNews(unittest.TestCase):
    """The one test that needs the dataset. Skipped if it cannot be reached."""

    @classmethod
    def setUpClass(cls):
        try:
            from src.benchmark import load_splits
            cls.splits = load_splits()
        except Exception as error:
            raise unittest.SkipTest(f"AG News unavailable: {error}") from error

    def test_the_published_test_split_is_the_expected_size(self):
        self.assertEqual(len(self.splits["test"][1]), 7_600)

    def test_the_validation_split_comes_out_of_training(self):
        train, validation = self.splits["train"][1], self.splits["validation"][1]
        self.assertEqual(len(train) + len(validation), 120_000)

    def test_all_four_classes_appear_in_every_split(self):
        for name in ("train", "validation", "test"):
            with self.subTest(split=name):
                self.assertEqual(set(np.unique(self.splits[name][1])), {0, 1, 2, 3})

    def test_the_test_split_barely_overlaps_training(self):
        """Recorded as a fact. If a future dataset revision changes it, the
        reported accuracy stops meaning what the README says it means."""
        found = contamination(self.splits)
        self.assertEqual(found["exact_matches_in_train"], 0)
        self.assertLess(found["normalised_match_fraction"], 0.01)


if __name__ == "__main__":
    unittest.main()
