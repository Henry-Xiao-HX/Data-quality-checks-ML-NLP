"""
Unit tests for data quality checker module.
"""

import unittest
import sys
sys.path.insert(0, '../src')

from data_quality_checker import DataQualityChecker, QualityMetricsAggregator


class TestDataQualityChecker(unittest.TestCase):
    """Test cases for DataQualityChecker."""
    
    def setUp(self):
        """Initialize checker for each test."""
        self.checker = DataQualityChecker()
    
    def test_rouge_single_text(self):
        """Test ROUGE computation with single texts."""
        pred = "The cat sat on the mat"
        ref = "A cat was on the mat"
        
        result = self.checker.compute_rouge(pred, ref)
        
        self.assertIn('rouge1', result)
        self.assertIn('rouge2', result)
        self.assertIn('rougeL', result)
        
        # Verify scores are between 0 and 1
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 1)
    
    def test_rouge_multiple_texts(self):
        """Test ROUGE computation with multiple texts."""
        preds = ["The cat sat", "Dogs run fast"]
        refs = ["A cat sat", "The dog runs"]
        
        result = self.checker.compute_rouge(preds, refs)
        
        self.assertIn('rouge1', result)
        self.assertIsInstance(result['rouge1'], (int, float))
    
    def test_bleu_single_text(self):
        """Test BLEU computation with single texts."""
        pred = ["the cat sat on the mat"]
        ref = ["a cat was on the mat"]
        
        result = self.checker.compute_bleu(pred, ref)
        
        self.assertIn('bleu', result)
        self.assertGreaterEqual(result['bleu'], 0)
        self.assertLessEqual(result['bleu'], 1)
    
    def test_bleu_multiple_references(self):
        """Test BLEU with multiple references."""
        pred = ["the cat sat"]
        refs = [["a cat sat", "the cat was sitting"]]
        
        result = self.checker.compute_bleu(pred, refs)
        
        self.assertIn('bleu', result)
        self.assertIsInstance(result['bleu'], (int, float))
    
    def test_all_metrics(self):
        """Test computing all metrics at once."""
        pred = "Machine learning is powerful"
        ref = "ML is very powerful"
        
        result = self.checker.compute_all_metrics(pred, ref)
        
        self.assertIn('rouge', result)
        self.assertIn('bleu', result)
    
    def test_batch_compute(self):
        """Test batch computation."""
        preds = [
            ["The cat sat", "Dogs run"],
            ["Birds fly", "Fish swim"]
        ]
        refs = [
            ["A cat sat", "The dog runs"],
            ["Birds are flying", "Fish are swimming"]
        ]
        
        results = self.checker.batch_compute_metrics(preds, refs)
        
        self.assertEqual(len(results), 2)
        self.assertIn('rouge', results[0])
        self.assertIn('bleu', results[1])
    
    def test_rouge_types_filter(self):
        """Test selective ROUGE types."""
        pred = "The quick brown fox"
        ref = "A quick brown fox"
        
        result = self.checker.compute_rouge(
            pred, ref,
            rouge_types=['rouge1', 'rouge2']
        )
        
        self.assertIn('rouge1', result)
        self.assertIn('rouge2', result)
    
    def test_format_results(self):
        """Test result formatting."""
        results = {'metric1': 0.5, 'metric2': 0.75}
        
        formatted = self.checker.format_results({'test': results})
        
        self.assertIn('0.5', formatted)
        self.assertIn('TEST', formatted)


class TestQualityMetricsAggregator(unittest.TestCase):
    """Test cases for QualityMetricsAggregator."""
    
    def setUp(self):
        """Initialize aggregator for each test."""
        self.aggregator = QualityMetricsAggregator()
    
    def test_aggregate_rouge_mean(self):
        """Test ROUGE aggregation with mean."""
        scores = [
            {'rouge1': 0.5, 'rouge2': 0.3},
            {'rouge1': 0.7, 'rouge2': 0.5}
        ]
        
        result = self.aggregator.aggregate_rouge_scores(scores, 'mean')
        
        self.assertAlmostEqual(result['rouge1'], 0.6)
        self.assertAlmostEqual(result['rouge2'], 0.4)
    
    def test_aggregate_rouge_median(self):
        """Test ROUGE aggregation with median."""
        scores = [
            {'rouge1': 0.2},
            {'rouge1': 0.5},
            {'rouge1': 0.8}
        ]
        
        result = self.aggregator.aggregate_rouge_scores(scores, 'median')
        
        self.assertAlmostEqual(result['rouge1'], 0.5)
    
    def test_aggregate_rouge_min_max(self):
        """Test ROUGE aggregation with min and max."""
        scores = [
            {'rouge1': 0.3},
            {'rouge1': 0.7}
        ]
        
        result_min = self.aggregator.aggregate_rouge_scores(scores, 'min')
        result_max = self.aggregator.aggregate_rouge_scores(scores, 'max')
        
        self.assertAlmostEqual(result_min['rouge1'], 0.3)
        self.assertAlmostEqual(result_max['rouge1'], 0.7)
    
    def test_aggregate_bleu_mean(self):
        """Test BLEU aggregation with mean."""
        scores = [
            {'bleu': 0.5},
            {'bleu': 0.7}
        ]
        
        result = self.aggregator.aggregate_bleu_scores(scores, 'mean')
        
        self.assertAlmostEqual(result['bleu'], 0.6)
    
    def test_aggregate_empty_list(self):
        """Test aggregation with empty list."""
        result = self.aggregator.aggregate_rouge_scores([], 'mean')
        
        self.assertEqual(result, {})


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Initialize checker for each test."""
        self.checker = DataQualityChecker()
    
    def test_empty_strings(self):
        """Test with empty strings."""
        result = self.checker.compute_rouge("", "")
        self.assertIn('rouge1', result)
    
    def test_identical_texts(self):
        """Test with identical prediction and reference."""
        text = "This is identical text"
        result = self.checker.compute_rouge(text, text)
        
        # All ROUGE scores should be 1.0
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertAlmostEqual(result[key], 1.0, places=2)
    
    def test_completely_different_texts(self):
        """Test with completely different texts."""
        pred = "apple banana cherry"
        ref = "xyz qwerty asdf"
        
        result = self.checker.compute_rouge(pred, ref)
        
        # Scores should be 0 or very low
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertLess(result[key], 0.1)


if __name__ == '__main__':
    unittest.main()
