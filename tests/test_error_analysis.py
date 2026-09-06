"""Tests for how representative mistakes are chosen.

The point of selecting errors by rule is that the examples shown are the ones the
confusion matrix says matter, rather than the ones that read well. These tests
pin down that the categories come from the data: a confusion pair with no
mistakes behind it must not appear, and the counts must match the predictions.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.benchmark import LABEL_NAMES, confusion_pairs, scores
from src.error_analysis import SHORT_WORDS, categorise, excerpt


def make(actual, predicted, texts=None):
    actual = np.array(actual)
    predicted = np.array(predicted)
    if texts is None:
        texts = [f"a story about something number {index} with several words here"
                 for index in range(len(actual))]
    matrix = scores(actual, predicted)["confusion_matrix"]
    return texts, actual, predicted, matrix


class Categories(unittest.TestCase):
    def test_a_confusion_pair_with_no_mistakes_is_not_reported(self):
        """Everything correct means no categories at all, not empty ones."""
        actual = [0, 1, 2, 3] * 10
        texts, actual, predicted, matrix = make(actual, actual)
        self.assertEqual(categorise(texts, actual, predicted, matrix), {})

    def test_the_largest_confusion_pair_is_reported_first(self):
        actual = [2] * 20 + [0] * 10
        predicted = [3] * 12 + [2] * 8 + [1] * 3 + [0] * 7
        texts, actual, predicted, matrix = make(actual, predicted)
        categories = categorise(texts, actual, predicted, matrix)
        first = next(iter(categories))
        self.assertEqual(first, "Business called Sci/Tech")
        self.assertEqual(categories[first]["count"], 12)

    def test_counts_match_the_predictions(self):
        actual = [0] * 10 + [1] * 10
        predicted = [1] * 4 + [0] * 6 + [0] * 3 + [1] * 7
        texts, actual, predicted, matrix = make(actual, predicted)
        categories = categorise(texts, actual, predicted, matrix)
        total = sum(entry["count"] for entry in categories.values()
                    if entry["kind"] == "confusion pair")
        self.assertEqual(total, int((actual != predicted).sum()))

    def test_short_items_are_only_counted_when_some_are_wrong(self):
        actual = [0, 1] * 10
        predicted = [0, 1] * 10
        texts = ["tiny"] * 20
        matrix = scores(np.array(actual), np.array(predicted))["confusion_matrix"]
        categories = categorise(texts, np.array(actual), np.array(predicted), matrix)
        self.assertNotIn("short items", categories)

    def test_short_items_are_found_when_they_are_wrong(self):
        actual = [0] * 10
        predicted = [1] * 10
        texts = ["very short headline"] * 10
        self.assertLess(len(texts[0].split()), SHORT_WORDS)
        matrix = scores(np.array(actual), np.array(predicted))["confusion_matrix"]
        categories = categorise(texts, np.array(actual), np.array(predicted), matrix)
        self.assertIn("short items", categories)
        self.assertEqual(categories["short items"]["count"], 10)

    def test_entity_heavy_items_need_several_capitalised_tokens(self):
        actual = [0] * 4
        predicted = [1] * 4
        plain = "the market moved a little lower over the course of the day"
        heavy = "Reuters Smith Jones Lopez Bank Of England Governor Speaks"
        texts = [plain, plain, heavy, heavy]
        matrix = scores(np.array(actual), np.array(predicted))["confusion_matrix"]
        categories = categorise(texts, np.array(actual), np.array(predicted), matrix)
        self.assertIn("entity-heavy items", categories)
        self.assertEqual(categories["entity-heavy items"]["count"], 2)


class ConfusionPairs(unittest.TestCase):
    def test_pairs_are_ordered_by_size_and_exclude_the_diagonal(self):
        actual = [0] * 20 + [1] * 20
        predicted = [1] * 5 + [0] * 15 + [0] * 9 + [1] * 11
        matrix = scores(np.array(actual), np.array(predicted))["confusion_matrix"]
        pairs = confusion_pairs(matrix)
        self.assertEqual(pairs[0]["count"], 9)
        self.assertEqual(pairs[1]["count"], 5)
        for entry in pairs:
            self.assertNotEqual(entry["true"], entry["predicted"])

    def test_the_shares_add_up_over_all_pairs(self):
        actual = [0] * 20 + [1] * 20
        predicted = [1] * 5 + [0] * 15 + [0] * 9 + [1] * 11
        matrix = scores(np.array(actual), np.array(predicted))["confusion_matrix"]
        pairs = confusion_pairs(matrix, top=len(LABEL_NAMES) ** 2)
        self.assertAlmostEqual(sum(entry["share_of_errors"] for entry in pairs), 1.0,
                               places=3)


class Excerpts(unittest.TestCase):
    def test_a_long_excerpt_is_cut_and_marked(self):
        text = "word " * 200
        cut = excerpt(text)
        self.assertLess(len(cut), 220)
        self.assertTrue(cut.endswith("..."))

    def test_whitespace_and_escapes_are_collapsed(self):
        self.assertEqual(excerpt("a\\b   c\n d"), "a b c d")


if __name__ == "__main__":
    unittest.main()
