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
        """Test tokenization of empty string."""
        tokens = tokenize_text("")
        self.assertEqual(len(tokens), 1)  # Empty string tokenizes to ['']
    
    def test_tokenize_single_word(self):
        """Test tokenization of single word."""
        tokens = tokenize_text("hello")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0], "hello")


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


class TestQualityIssueDetection(unittest.TestCase):
    """Test quality issue detection."""
    
    def test_detect_empty_predictions(self):
        """Test detection of empty predictions."""
        predictions = ["", "valid text"]
        references = ["ref1", "ref2"]
        
        issues = detect_data_quality_issues(predictions, references)
        
        self.assertIn('empty_predictions', issues)
    
    def test_detect_duplicates(self):
        """Test detection of duplicate predictions."""
        predictions = ["text one", "text one", "text two"]
        references = ["ref1", "ref2", "ref3"]
        
        issues = detect_data_quality_issues(predictions, references)
        
        self.assertIn('duplicate_predictions', issues)
    
    def test_detect_short_predictions(self):
        """Test detection of very short predictions."""
        predictions = ["a b", "this is a longer text"]
        references = ["reference one", "reference two"]
        
        issues = detect_data_quality_issues(predictions, references)
        
        self.assertIn('very_short_predictions', issues)
    
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


if __name__ == '__main__':
    unittest.main()
