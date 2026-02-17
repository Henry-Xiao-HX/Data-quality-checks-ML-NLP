"""
Regression Quality Checker - Orchestrator Module.

Coordinates computation of regression metrics and provides
a unified interface for quality assessment.
"""

from typing import Dict, Union, Optional
import numpy as np

try:
    from .metrics import RegressionMetrics
except ImportError:
    # Fallback for direct imports
    from metrics import RegressionMetrics


class RegressionChecker:
    """
    Comprehensive regression quality checker.
    
    This is an orchestrator class that coordinates metric computation by delegating to:
    - RegressionMetrics: Handles all regression metrics
    
    Supported Metrics:
    - R-Squared (Coefficient of Determination)
    - Explained Variance Score
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    """
    
    def __init__(self):
        """Initialize the regression checker with metrics calculator."""
        self.metrics_calculator = RegressionMetrics()
        self.metrics_cache = {}
    
    def check_quality(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Perform comprehensive quality check on regression predictions.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            cache_key: Optional key for caching results.
        
        Returns:
            dict: Dictionary containing all computed quality metrics.
        
        Raises:
            ValueError: If y_true and y_pred have different lengths.
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
        
        # Compute all metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            y_true, y_pred, sample_weight
        )
        
        # Cache results
        if cache_key:
            self.metrics_cache[cache_key] = metrics
        
        return metrics
    
    def compute_r_squared(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute R-Squared (Coefficient of Determination).
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        
        Returns:
            float: R² score.
        """
        return self.metrics_calculator.compute_r_squared(y_true, y_pred, sample_weight)
    
    def compute_explained_variance(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Explained Variance Score.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        
        Returns:
            float: Explained variance score.
        """
        return self.metrics_calculator.compute_explained_variance(
            y_true, y_pred, sample_weight
        )
    
    def compute_rmse(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Root Mean Squared Error (RMSE).
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        
        Returns:
            float: RMSE score.
        """
        return self.metrics_calculator.compute_rmse(y_true, y_pred, sample_weight)
    
    def compute_mae(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        
        Returns:
            float: MAE score.
        """
        return self.metrics_calculator.compute_mae(y_true, y_pred, sample_weight)
    
    def compute_mse(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Mean Squared Error (MSE).
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        
        Returns:
            float: MSE score.
        """
        return self.metrics_calculator.compute_mse(y_true, y_pred, sample_weight)
    
    def get_residuals(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ) -> np.ndarray:
        """
        Get prediction residuals.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            ndarray: Array of residuals.
        """
        return self.metrics_calculator.compute_residuals(y_true, y_pred)
    
    def get_residual_stats(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
    ) -> Dict[str, float]:
        """
        Get statistics on prediction residuals.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            dict: Dictionary containing residual statistics.
        """
        return self.metrics_calculator.compute_residual_stats(y_true, y_pred)
    
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
Regression Quality Metrics Available:
=====================================

Model Performance Metrics:
  - r_squared: Coefficient of Determination (-∞ to 1, higher is better)
  - explained_variance: Explained Variance Score (-∞ to 1, higher is better)
  - rmse: Root Mean Squared Error (0 to ∞, lower is better)
  - mae: Mean Absolute Error (0 to ∞, lower is better)
  - mse: Mean Squared Error (0 to ∞, lower is better)

Residual Analysis:
  - residuals: Individual prediction errors
  - residual_stats: Mean, std, min, max, median of residuals

Notes:
  - All error metrics (RMSE, MAE, MSE) are in the same units as y_true
  - RMSE is more sensitive to outliers than MAE
  - Explained Variance can be larger than R² due to sensitivity to outliers
  - R² can be negative if the model performs worse than a horizontal line
        """
        return summary
