"""
Unit tests for data quality checker module.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data_quality_checker import DataQualityChecker, QualityMetricsAggregator
from binary_classification import BinaryClassifierChecker
from regression import RegressionChecker


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


class TestBinaryClassificationChecker(unittest.TestCase):
    """Test cases for BinaryClassifierChecker."""
    
    def setUp(self):
        """Initialize binary classification checker for each test."""
        self.checker = BinaryClassifierChecker()
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.randint(0, 2, self.n_samples)
        self.y_pred = np.random.randint(0, 2, self.n_samples)
        self.y_pred_proba = np.random.rand(self.n_samples)
    
    def test_check_quality_basic(self):
        """Test basic quality check with binary predictions."""
        metrics = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred
        )
        
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_measure', metrics)
        self.assertIn('true_positive_rate', metrics)
        self.assertIn('false_positive_rate', metrics)
    
    def test_check_quality_with_probabilities(self):
        """Test quality check with probability predictions."""
        metrics = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_pred_proba=self.y_pred_proba
        )
        
        self.assertIn('roc_auc', metrics)
        self.assertIn('log_loss', metrics)
        self.assertIn('auc_pr', metrics)
    
    def test_compute_precision(self):
        """Test precision computation."""
        precision = self.checker.compute_precision(self.y_true, self.y_pred)
        
        self.assertIsInstance(precision, float)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
    
    def test_compute_recall(self):
        """Test recall computation."""
        recall = self.checker.compute_recall(self.y_true, self.y_pred)
        
        self.assertIsInstance(recall, float)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
    
    def test_compute_f1_measure(self):
        """Test F1-measure computation."""
        f1 = self.checker.compute_f1_measure(self.y_true, self.y_pred)
        
        self.assertIsInstance(f1, float)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
    
    def test_compute_roc_auc(self):
        """Test ROC AUC computation."""
        roc_auc = self.checker.compute_roc_auc(self.y_true, self.y_pred_proba)
        
        self.assertIsInstance(roc_auc, float)
        self.assertGreaterEqual(roc_auc, 0.0)
        self.assertLessEqual(roc_auc, 1.0)
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        precision = self.checker.compute_precision(y_true, y_pred)
        recall = self.checker.compute_recall(y_true, y_pred)
        f1 = self.checker.compute_f1_measure(y_true, y_pred)
        
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)
    
    def test_mismatched_lengths(self):
        """Test with mismatched input lengths."""
        with self.assertRaises(ValueError):
            self.checker.check_quality(
                y_true=np.array([0, 1, 0]),
                y_pred=np.array([1, 0])
            )


class TestRegressionChecker(unittest.TestCase):
    """Test cases for RegressionChecker."""
    
    def setUp(self):
        """Initialize regression checker for each test."""
        self.checker = RegressionChecker()
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.rand(self.n_samples) * 100
        self.noise = np.random.normal(0, 5, self.n_samples)
        self.y_pred = self.y_true + self.noise
    
    def test_check_quality(self):
        """Test quality check for regression."""
        metrics = self.checker.check_quality(
            y_true=self.y_true,
            y_pred=self.y_pred
        )
        
        self.assertIn('r_squared', metrics)
        self.assertIn('explained_variance', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
    
    def test_compute_r_squared(self):
        """Test R-squared computation."""
        r_squared = self.checker.compute_r_squared(self.y_true, self.y_pred)
        
        self.assertIsInstance(r_squared, float)
        # R-squared can be negative for very poor models
        self.assertLess(r_squared, 1.0)
    
    def test_compute_rmse(self):
        """Test RMSE computation."""
        rmse = self.checker.compute_rmse(self.y_true, self.y_pred)
        
        self.assertIsInstance(rmse, float)
        self.assertGreaterEqual(rmse, 0.0)
    
    def test_compute_mae(self):
        """Test MAE computation."""
        mae = self.checker.compute_mae(self.y_true, self.y_pred)
        
        self.assertIsInstance(mae, float)
        self.assertGreaterEqual(mae, 0.0)
    
    def test_compute_mse(self):
        """Test MSE computation."""
        mse = self.checker.compute_mse(self.y_true, self.y_pred)
        
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0.0)
    
    def test_compute_explained_variance(self):
        """Test explained variance computation."""
        explained_var = self.checker.compute_explained_variance(self.y_true, self.y_pred)
        
        self.assertIsInstance(explained_var, float)
        self.assertLess(explained_var, 1.0)
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        r_squared = self.checker.compute_r_squared(y_true, y_pred)
        rmse = self.checker.compute_rmse(y_true, y_pred)
        mae = self.checker.compute_mae(y_true, y_pred)
        
        self.assertAlmostEqual(r_squared, 1.0, places=5)
        self.assertAlmostEqual(rmse, 0.0, places=5)
        self.assertAlmostEqual(mae, 0.0, places=5)
    
    def test_get_residuals(self):
        """Test residuals computation."""
        residuals = self.checker.get_residuals(self.y_true, self.y_pred)
        
        self.assertEqual(len(residuals), self.n_samples)
        self.assertIsInstance(residuals, np.ndarray)
    
    def test_get_residual_stats(self):
        """Test residual statistics."""
        residual_stats = self.checker.get_residual_stats(self.y_true, self.y_pred)
        
        self.assertIn('mean', residual_stats)
        self.assertIn('std', residual_stats)
        self.assertIn('min', residual_stats)
        self.assertIn('max', residual_stats)
    
    def test_mismatched_lengths(self):
        """Test with mismatched input lengths."""
        with self.assertRaises(ValueError):
            self.checker.check_quality(
                y_true=np.array([1.0, 2.0, 3.0]),
                y_pred=np.array([1.0, 2.0])
            )


if __name__ == '__main__':
    unittest.main()
