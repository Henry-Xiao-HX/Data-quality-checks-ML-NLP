"""
Binary Classifier Quality Checker - Orchestrator Module.

Coordinates computation of binary classification metrics and provides
a unified interface for quality assessment.
"""

from typing import Dict, Union, Optional, Tuple
import numpy as np

try:
    from .metrics import BinaryClassificationMetrics
except ImportError:
    # Fallback for direct imports
    from metrics import BinaryClassificationMetrics


class BinaryClassifierChecker:
    """
    Comprehensive binary classification quality checker.
    
    This is an orchestrator class that coordinates metric computation by delegating to:
    - BinaryClassificationMetrics: Handles all binary classification metrics
    
    Supported Metrics:
    - AUROC (Area Under ROC Curve)
    - True Positive Rate (Sensitivity/Recall)
    - Precision
    - F1-Measure
    - Log Loss (Cross-Entropy)
    - False Positive Rate
    - AUPR (Area Under Precision-Recall Curve)
    - Recall
    """
    
    def __init__(self):
        """Initialize the binary classifier checker with metrics calculator."""
        self.metrics_calculator = BinaryClassificationMetrics()
        self.metrics_cache = {}
    
    def check_quality(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        y_pred_proba: Optional[Union[np.ndarray, list]] = None,
        sample_weight: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Perform comprehensive quality check on binary classification predictions.
        
        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred: Predicted binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class (optional).
                         If not provided for probability-based metrics, they will be skipped.
            sample_weight: Optional sample weights for weighted metrics.
            cache_key: Optional key for caching results.
        
        Returns:
            dict: Dictionary containing all computed quality metrics.
        
        Raises:
            ValueError: If y_true and y_pred have different lengths.
            TypeError: If probability-dependent metrics are requested but y_pred_proba is None.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length. "
                f"Got {len(y_true)} and {len(y_pred)}"
            )
        
        # Check cache
        if cache_key and cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        # Compute metrics requiring only binary predictions
        base_metrics = {
            'true_positive_rate': self.metrics_calculator.compute_true_positive_rate(
                y_true, y_pred
            ),
            'precision': self.metrics_calculator.compute_precision(y_true, y_pred),
            'f1_measure': self.metrics_calculator.compute_f1_measure(y_true, y_pred),
            'false_positive_rate': self.metrics_calculator.compute_false_positive_rate(
                y_true, y_pred
            ),
            'recall': self.metrics_calculator.compute_recall(y_true, y_pred),
        }
        
        # Compute metrics requiring probability predictions
        if y_pred_proba is not None:
            y_pred_proba = np.array(y_pred_proba)
            if len(y_true) != len(y_pred_proba):
                raise ValueError(
                    f"y_true and y_pred_proba must have the same length. "
                    f"Got {len(y_true)} and {len(y_pred_proba)}"
                )
            
            probability_metrics = {
                'roc_auc': self.metrics_calculator.compute_roc_auc(
                    y_true, y_pred_proba, sample_weight
                ),
                'log_loss': self.metrics_calculator.compute_log_loss(
                    y_true, y_pred_proba, sample_weight
                ),
                'auc_pr': self.metrics_calculator.compute_auc_pr(y_true, y_pred_proba),
            }
            base_metrics.update(probability_metrics)
        
        # Cache results
        if cache_key:
            self.metrics_cache[cache_key] = base_metrics
        
        return base_metrics
    
    def compute_roc_auc(
        self,
        y_true: Union[np.ndarray, list],
        y_pred_proba: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Area Under the ROC Curve.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            sample_weight: Optional sample weights.
        
        Returns:
            float: AUROC score.
        """
        return self.metrics_calculator.compute_roc_auc(y_true, y_pred_proba, sample_weight)
    
    def compute_precision(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ) -> float:
        """
        Compute Precision.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
        
        Returns:
            float: Precision score.
        """
        return self.metrics_calculator.compute_precision(y_true, y_pred)
    
    def compute_recall(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ) -> float:
        """
        Compute Recall (True Positive Rate).
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
        
        Returns:
            float: Recall score.
        """
        return self.metrics_calculator.compute_recall(y_true, y_pred)
    
    def compute_f1_measure(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ) -> float:
        """
        Compute F1-Measure.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
        
        Returns:
            float: F1-score.
        """
        return self.metrics_calculator.compute_f1_measure(y_true, y_pred)
    
    def compute_log_loss(
        self,
        y_true: Union[np.ndarray, list],
        y_pred_proba: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Logarithmic Loss.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            sample_weight: Optional sample weights.
        
        Returns:
            float: Log loss.
        """
        return self.metrics_calculator.compute_log_loss(y_true, y_pred_proba, sample_weight)
    
    def get_roc_curve_data(
        self,
        y_true: Union[np.ndarray, list],
        y_pred_proba: Union[np.ndarray, list],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data for visualization.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
        
        Returns:
            tuple: (fpr, tpr, thresholds) arrays.
        """
        return self.metrics_calculator.get_roc_curve_data(y_true, y_pred_proba)
    
    def get_pr_curve_data(
        self,
        y_true: Union[np.ndarray, list],
        y_pred_proba: Union[np.ndarray, list],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data for visualization.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
        
        Returns:
            tuple: (precision, recall, thresholds) arrays.
        """
        return self.metrics_calculator.get_pr_curve_data(y_true, y_pred_proba)
    
    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        self.metrics_cache.clear()
    
    def get_metrics_summary(self) -> str:
        """
        Get a human-readable summary of available metrics.
        
        Returns:
            str: Summary of all available metrics.
        """
        summary = """
Binary Classification Quality Metrics Available:
================================================

Threshold-Independent Metrics (require y_pred_proba):
  - roc_auc: Area Under ROC Curve (0-1, higher is better)
  - auc_pr: Area Under Precision-Recall Curve (0-1, higher is better)
  - log_loss: Logarithmic Loss (lower is better)

Threshold-Dependent Metrics (require y_pred):
  - precision: TP / (TP + FP)
  - recall: TP / (TP + FN)
  - true_positive_rate: Same as recall
  - false_positive_rate: FP / (FP + TN)
  - f1_measure: 2 * (Precision * Recall) / (Precision + Recall)

All metrics return values between 0 and 1 (except log_loss).
        """
        return summary
