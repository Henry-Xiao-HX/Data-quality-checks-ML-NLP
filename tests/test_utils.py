"""
Unit tests for utility functions.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import (
    preprocess_text,
    tokenize_text,
    get_ngrams,
    calculate_precision_recall_f1,
    calculate_length_similarity,
    detect_data_quality_issues
)


class TestTextProcessing(unittest.TestCase):
    """Test text preprocessing functions."""

    def test_preprocess_lowercase(self):
        """Test lowercase conversion."""
        result = preprocess_text("HELLO WORLD", lowercase=True)
        self.assertEqual(result, "hello world")

    def test_preprocess_keep_case(self):
        """Test keeping original case."""
        text = "HELLO world"
        result = preprocess_text(text, lowercase=False)
        self.assertEqual(result, text)

    def test_preprocess_remove_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, World! How are you?"
        result = preprocess_text(text, lowercase=False, remove_punctuation=True)
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)

    def test_preprocess_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello   World    this    has    extra    spaces"
        result = preprocess_text(text)
        self.assertNotIn("   ", result)
        self.assertEqual(result, "hello world this has extra spaces")

    def test_preprocess_strips_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is stripped."""
        result = preprocess_text("  hello world  ")
        self.assertEqual(result, "hello world")

    def test_preprocess_lowercase_and_remove_punctuation(self):
        """Test combining lowercase and punctuation removal."""
        result = preprocess_text("Hello, World!", lowercase=True, remove_punctuation=True)
        self.assertEqual(result, "hello world")

    def test_preprocess_empty_string(self):
        """Test preprocessing an empty string."""
        result = preprocess_text("")
        self.assertEqual(result, "")


class TestTokenization(unittest.TestCase):
    """Test tokenization functions."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello world this is a test"
        tokens = tokenize_text(text)

        self.assertEqual(len(tokens), 6)
        self.assertEqual(tokens[0], "Hello")
        self.assertEqual(tokens[-1], "test")

    def test_tokenize_empty(self):
        """Test tokenization of empty string returns empty list."""
        tokens = tokenize_text("")
        self.assertEqual(len(tokens), 0)
        self.assertEqual(tokens, [])

    def test_tokenize_single_word(self):
        """Test tokenization of single word."""
        tokens = tokenize_text("hello")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], "hello")

    def test_tokenize_preserves_case(self):
        """Test that tokenization preserves case."""
        tokens = tokenize_text("Hello WORLD")
        self.assertEqual(tokens[0], "Hello")
        self.assertEqual(tokens[1], "WORLD")

    def test_tokenize_multiple_spaces(self):
        """Test tokenization with multiple spaces between words."""
        tokens = tokenize_text("hello   world")
        # str.split() handles multiple spaces correctly
        self.assertEqual(tokens, ["hello", "world"])


class TestNgrams(unittest.TestCase):
    """Test n-gram extraction."""

    def test_unigrams(self):
        """Test unigram extraction."""
        tokens = ["the", "cat", "sat"]
        ngrams = get_ngrams(tokens, 1)

        self.assertEqual(len(ngrams), 3)
        self.assertEqual(ngrams[0], ("the",))
        self.assertEqual(ngrams[1], ("cat",))

    def test_bigrams(self):
        """Test bigram extraction."""
        tokens = ["the", "cat", "sat", "down"]
        ngrams = get_ngrams(tokens, 2)

        self.assertEqual(len(ngrams), 3)
        self.assertEqual(ngrams[0], ("the", "cat"))
        self.assertEqual(ngrams[1], ("cat", "sat"))

    def test_trigrams(self):
        """Test trigram extraction."""
        tokens = ["the", "cat", "sat", "down"]
        ngrams = get_ngrams(tokens, 3)

        self.assertEqual(len(ngrams), 2)
        self.assertEqual(ngrams[0], ("the", "cat", "sat"))

    def test_ngrams_n_equals_length(self):
        """Test n-gram where n equals token list length."""
        tokens = ["the", "cat", "sat"]
        ngrams = get_ngrams(tokens, 3)

        self.assertEqual(len(ngrams), 1)
        self.assertEqual(ngrams[0], ("the", "cat", "sat"))

    def test_ngrams_n_exceeds_length(self):
        """Test n-gram where n exceeds token list length returns empty list."""
        tokens = ["the", "cat"]
        ngrams = get_ngrams(tokens, 5)

        self.assertEqual(ngrams, [])

    def test_ngrams_empty_tokens(self):
        """Test n-gram extraction from empty token list."""
        ngrams = get_ngrams([], 2)
        self.assertEqual(ngrams, [])

    def test_ngrams_returns_tuples(self):
        """Test that n-grams are returned as tuples."""
        tokens = ["a", "b", "c"]
        ngrams = get_ngrams(tokens, 2)

        for ngram in ngrams:
            self.assertIsInstance(ngram, tuple)


class TestPrecisionRecallF1(unittest.TestCase):
    """Test precision, recall, and F1 calculation."""

    def test_exact_match(self):
        """Test with exact matching texts."""
        predictions = ["the cat sat"]
        references = ["the cat sat"]

        result = calculate_precision_recall_f1(predictions, references)

        self.assertAlmostEqual(result['precision'], 1.0)
        self.assertAlmostEqual(result['recall'], 1.0)
        self.assertAlmostEqual(result['f1'], 1.0)

    def test_partial_match(self):
        """Test with partial matches."""
        predictions = ["the cat"]
        references = ["the cat sat"]

        result = calculate_precision_recall_f1(predictions, references)

        self.assertGreater(result['precision'], 0)
        self.assertGreater(result['recall'], 0)
        self.assertGreater(result['f1'], 0)

    def test_no_match(self):
        """Test with no matching tokens."""
        predictions = ["apple banana"]
        references = ["xyz qwerty"]

        result = calculate_precision_recall_f1(predictions, references)

        self.assertEqual(result['precision'], 0.0)
        self.assertEqual(result['recall'], 0.0)
        self.assertEqual(result['f1'], 0.0)

    def test_empty_predictions(self):
        """Test with empty prediction string."""
        predictions = [""]
        references = ["the cat sat"]

        result = calculate_precision_recall_f1(predictions, references)

        self.assertEqual(result['precision'], 0.0)
        self.assertEqual(result['recall'], 0.0)
        self.assertEqual(result['f1'], 0.0)

    def test_empty_references(self):
        """Test with empty reference string."""
        predictions = ["the cat sat"]
        references = [""]

        result = calculate_precision_recall_f1(predictions, references)

        self.assertEqual(result['recall'], 0.0)

    def test_result_keys(self):
        """Test that result contains all expected keys."""
        result = calculate_precision_recall_f1(["hello"], ["hello"])

        self.assertIn('precision', result)
        self.assertIn('recall', result)
        self.assertIn('f1', result)

    def test_f1_is_harmonic_mean(self):
        """Test that F1 is the harmonic mean of precision and recall."""
        predictions = ["the cat"]
        references = ["the cat sat"]

        result = calculate_precision_recall_f1(predictions, references)

        p, r = result['precision'], result['recall']
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            self.assertAlmostEqual(result['f1'], expected_f1, places=5)


class TestLengthSimilarity(unittest.TestCase):
    """Test length similarity calculation."""

    def test_equal_lengths(self):
        """Test with equal length texts."""
        predictions = ["one two three"]
        references = ["four five six"]

        result = calculate_length_similarity(predictions, references)

        self.assertAlmostEqual(result['length_ratio'], 1.0)

    def test_different_lengths(self):
        """Test with different length texts."""
        predictions = ["one two three four five"]
        references = ["one two"]

        result = calculate_length_similarity(predictions, references)

        self.assertGreater(result['length_ratio'], 1.0)

    def test_prediction_longer(self):
        """Test when prediction is longer."""
        predictions = ["one two three four"]
        references = ["one two"]

        result = calculate_length_similarity(predictions, references)

        self.assertEqual(result['avg_pred_length'], 4)
        self.assertEqual(result['avg_ref_length'], 2)
        self.assertAlmostEqual(result['length_ratio'], 2.0)

    def test_result_keys(self):
        """Test that result contains all expected keys."""
        result = calculate_length_similarity(["hello world"], ["hi there"])

        self.assertIn('avg_pred_length', result)
        self.assertIn('avg_ref_length', result)
        self.assertIn('length_ratio', result)

    def test_multiple_texts_averages(self):
        """Test that lengths are averaged across multiple texts."""
        predictions = ["one two", "one two three four"]  # avg = 3
        references = ["a b c", "a b c d e f"]           # avg = 4.5

        result = calculate_length_similarity(predictions, references)

        self.assertAlmostEqual(result['avg_pred_length'], 3.0)
        self.assertAlmostEqual(result['avg_ref_length'], 4.5)

    def test_zero_reference_length(self):
        """Test length ratio when reference is empty (zero length)."""
        predictions = ["one two three"]
        references = [""]

        result = calculate_length_similarity(predictions, references)

        # avg_ref_length is 0, so length_ratio should be 0.0
        self.assertEqual(result['length_ratio'], 0.0)


class TestQualityIssueDetection(unittest.TestCase):
    """Test quality issue detection."""

    def test_detect_empty_predictions(self):
        """Test detection of empty predictions."""
        predictions = ["", "valid text"]
        references = ["ref1", "ref2"]

        issues = detect_data_quality_issues(predictions, references)

        self.assertIn('empty_predictions', issues)
        self.assertEqual(len(issues['empty_predictions']), 1)

    def test_detect_duplicates(self):
        """Test detection of duplicate predictions."""
        predictions = ["text one", "text one", "text two"]
        references = ["ref1", "ref2", "ref3"]

        issues = detect_data_quality_issues(predictions, references)

        self.assertIn('duplicate_predictions', issues)
        self.assertEqual(len(issues['duplicate_predictions']), 1)

    def test_detect_short_predictions(self):
        """Test detection of very short predictions (< 5 tokens)."""
        predictions = ["a b", "this is a longer text with more words"]
        references = ["reference one", "reference two"]

        issues = detect_data_quality_issues(predictions, references)

        self.assertIn('very_short_predictions', issues)

    def test_detect_length_deviation(self):
        """Test detection of predictions with high length deviation."""
        # Prediction is much longer than reference (ratio > 2.0)
        predictions = ["one two three four five six seven eight nine ten"]
        references = ["one two"]

        issues = detect_data_quality_issues(
            predictions, references,
            max_length_ratio_threshold=(0.5, 2.0)
        )

        self.assertIn('high_length_deviation', issues)

    def test_detect_short_length_deviation(self):
        """Test detection of predictions much shorter than reference (ratio < 0.5)."""
        predictions = ["one"]
        references = ["one two three four five six seven eight nine ten"]

        issues = detect_data_quality_issues(
            predictions, references,
            max_length_ratio_threshold=(0.5, 2.0)
        )

        self.assertIn('high_length_deviation', issues)

    def test_no_issues(self):
        """Test when no issues are detected."""
        predictions = ["This is a good prediction"]
        references = ["This is a good reference"]

        issues = detect_data_quality_issues(
            predictions, references,
            max_length_ratio_threshold=(0.5, 2.0)
        )

        # Should be empty or minimal issues
        total_issues = sum(len(v) for v in issues.values())
        self.assertEqual(total_issues, 0)

    def test_returns_only_non_empty_issue_lists(self):
        """Test that issues dict only contains keys with actual issues."""
        predictions = ["This is a perfectly fine prediction"]
        references = ["This is a perfectly fine reference"]

        issues = detect_data_quality_issues(
            predictions, references,
            max_length_ratio_threshold=(0.5, 2.0)
        )

        # All values in the returned dict should be non-empty lists
        for key, value in issues.items():
            self.assertGreater(len(value), 0, f"Key '{key}' has empty list")

    def test_duplicate_detection_case_insensitive(self):
        """Test that duplicate detection is case-insensitive (via preprocess_text)."""
        predictions = ["Hello World", "hello world", "different text"]
        references = ["ref1", "ref2", "ref3"]

        issues = detect_data_quality_issues(predictions, references)

        self.assertIn('duplicate_predictions', issues)

    def test_multiple_duplicates(self):
        """Test detection of multiple duplicate predictions."""
        predictions = ["dup text", "dup text", "dup text", "unique"]
        references = ["r1", "r2", "r3", "r4"]

        issues = detect_data_quality_issues(predictions, references)

        self.assertIn('duplicate_predictions', issues)
        # Two duplicates of the first occurrence
        self.assertEqual(len(issues['duplicate_predictions']), 2)


if __name__ == '__main__':
    unittest.main()

# Made with Bob
