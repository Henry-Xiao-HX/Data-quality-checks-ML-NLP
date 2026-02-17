"""
Unified Data Quality Checker - Master Orchestrator.

Brings together quality checks for:
- Binary Classification Models
- Regression Models  
- Generative AI Text Models

Provides a unified interface for comprehensive model evaluation across all model types.
"""

from typing import Dict, Union, Optional, Any
import numpy as np

try:
    from .binary_classification import BinaryClassifierChecker
    from .regression import RegressionChecker
    from .generative_ai_text_model import BLEUCalculator, ROUGECalculator, BLEUAggregator, ROUGEAggregator
except ImportError:
    # Fallback for direct imports
    from binary_classification import BinaryClassifierChecker
    from regression import RegressionChecker
    from generative_ai_text_model import BLEUCalculator, ROUGECalculator, BLEUAggregator, ROUGEAggregator


class UnifiedDataQualityChecker:
    """
    Master orchestrator for comprehensive data quality checking.
    
    Coordinates quality assessment across three model categories:
    1. Binary Classification: AUROC, precision, recall, F1, log loss, etc.
    2. Regression: R², RMSE, MAE, MSE, explained variance
    3. Generative AI Text: BLEU, ROUGE-1/2/L/S scores
    """
    
    def __init__(self):
        """Initialize all quality checkers."""
        self.binary_classifier_checker = BinaryClassifierChecker()
        self.regression_checker = RegressionChecker()
        self.bleu_calculator = BLEUCalculator()
        self.rouge_calculator = ROUGECalculator()
        self.bleu_aggregator = BLEUAggregator()
        self.rouge_aggregator = ROUGEAggregator()
    
    # Binary Classification Methods
    def check_binary_classification_quality(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        y_pred_proba: Optional[Union[np.ndarray, list]] = None,
        sample_weight: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Check quality of binary classification predictions.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_pred_proba: Predicted probabilities (optional).
            sample_weight: Optional sample weights.
            cache_key: Optional cache key.
        
        Returns:
            Dictionary of binary classification metrics.
        """
        return self.binary_classifier_checker.check_quality(
            y_true, y_pred, y_pred_proba, sample_weight, cache_key
        )
    
    # Regression Methods
    def check_regression_quality(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Check quality of regression predictions.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            cache_key: Optional cache key.
        
        Returns:
            Dictionary of regression metrics.
        """
        return self.regression_checker.check_quality(
            y_true, y_pred, sample_weight, cache_key
        )
    
    # Generative AI Text Methods
    def compute_bleu(
        self,
        predictions: Union[list, str],
        references: Union[list, str],
        max_order: int = 4,
        smooth: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute BLEU score for text generation.
        
        Args:
            predictions: Model predictions.
            references: Reference texts.
            max_order: Maximum n-gram order.
            smooth: Whether to apply smoothing.
        
        Returns:
            BLEU scores.
        """
        return self.bleu_calculator.compute_bleu(
            predictions, references, max_order, smooth
        )
    
    def compute_rouge(
        self,
        predictions: Union[list, str],
        references: Union[list, str],
        rouge_types: Optional[list] = None,
        use_stemmer: bool = False,
        use_aggregator: bool = True,
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for text generation.
        
        Args:
            predictions: Model predictions.
            references: Reference texts.
            rouge_types: Types of ROUGE to compute.
            use_stemmer: Whether to use stemmer.
            use_aggregator: Whether to aggregate scores.
        
        Returns:
            ROUGE scores.
        """
        return self.rouge_calculator.compute_rouge(
            predictions, references, rouge_types, use_stemmer, use_aggregator
        )
    
    def aggregate_bleu_scores(
        self,
        scores_list: list,
        aggregation_type: str = 'mean',
    ) -> Dict[str, float]:
        """
        Aggregate BLEU scores across samples.
        
        Args:
            scores_list: List of BLEU score dictionaries.
            aggregation_type: Aggregation method.
        
        Returns:
            Aggregated BLEU scores.
        """
        return self.bleu_aggregator.aggregate_bleu_scores(scores_list, aggregation_type)
    
    def aggregate_rouge_scores(
        self,
        scores_list: list,
        aggregation_type: str = 'mean',
    ) -> Dict[str, float]:
        """
        Aggregate ROUGE scores across samples.
        
        Args:
            scores_list: List of ROUGE score dictionaries.
            aggregation_type: Aggregation method.
        
        Returns:
            Aggregated ROUGE scores.
        """
        return self.rouge_aggregator.aggregate_rouge_scores(scores_list, aggregation_type)
    
    def get_available_modules(self) -> Dict[str, str]:
        """
        Get summary of available quality check modules.
        
        Returns:
            Dictionary describing available modules.
        """
        return {
            'binary_classification': 'AUROC, precision, recall, F1, log loss, FPR, TPR, AUPR',
            'regression': 'R², RMSE, MAE, MSE, explained variance',
            'generative_ai_text': 'BLEU, ROUGE-1/2/L/S',
        }
    
    def clear_all_caches(self) -> None:
        """Clear all metric caches."""
        self.binary_classifier_checker.clear_cache()
        self.regression_checker.clear_cache()
