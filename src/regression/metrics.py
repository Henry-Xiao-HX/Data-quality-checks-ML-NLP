"""
Regression Metrics Calculator.

Computes various quality metrics for regression models using scikit-learn.
"""

from typing import Union, Dict, Optional
import numpy as np
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
)


class RegressionMetrics:
    """
    Comprehensive regression metrics calculator.
    
    Computes metrics for regression models:
    - R-Squared (Coefficient of Determination)
    - Explained Variance Score
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics_cache = {}
    
    def compute_r_squared(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average',
    ) -> float:
        """
        Compute R-Squared (Coefficient of Determination).
        
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - y_mean)²
        
        Best score is 1.0, and it can be negative if the model is arbitrarily worse.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            multioutput: Type of averaging for multioutput targets.
        
        Returns:
            float: R² score (typically between 0 and 1, can be negative).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(r2_score(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        ))
    
    def compute_explained_variance(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average',
    ) -> float:
        """
        Compute Explained Variance Score.
        
        Explained Variance = 1 - (Var(y_true - y_pred) / Var(y_true))
        
        Best score is 1.0. The score can be negative if the model is arbitrarily worse.
        Explained variance is sensitive to outliers (unlike R²), and can be larger than R².
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            multioutput: Type of averaging for multioutput targets.
        
        Returns:
            float: Explained variance score (typically between 0 and 1, can be negative).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(explained_variance_score(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        ))
    
    def compute_rmse(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'raw_values',
    ) -> Union[float, np.ndarray]:
        """
        Compute Root Mean Squared Error (RMSE).
        
        RMSE = sqrt(1/n * Σ(y_true - y_pred)²)
        
        Lower values are better. RMSE is more sensitive to outliers than MAE.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            multioutput: Type of averaging for multioutput targets.
        
        Returns:
            float or array: RMSE score(s) (always non-negative).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )
        rmse = np.sqrt(mse)
        return float(rmse) if isinstance(rmse, np.ndarray) and rmse.shape == () else rmse
    
    def compute_mae(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average',
    ) -> Union[float, np.ndarray]:
        """
        Compute Mean Absolute Error (MAE).
        
        MAE = 1/n * Σ|y_true - y_pred|
        
        Lower values are better. MAE is less sensitive to outliers than RMSE.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            multioutput: Type of averaging for multioutput targets.
        
        Returns:
            float or array: MAE score(s) (always non-negative).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(mean_absolute_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        ))
    
    def compute_mse(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'raw_values',
    ) -> Union[float, np.ndarray]:
        """
        Compute Mean Squared Error (MSE).
        
        MSE = 1/n * Σ(y_true - y_pred)²
        
        Lower values are better. MSE is more sensitive to outliers than MAE.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
            multioutput: Type of averaging for multioutput targets.
        
        Returns:
            float or array: MSE score(s) (always non-negative).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )
        return float(mse) if isinstance(mse, np.ndarray) and mse.shape == () else mse
    
    def compute_residuals(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
    ) -> np.ndarray:
        """
        Compute residuals (errors) from predictions.
        
        Residuals = y_true - y_pred
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            ndarray: Array of residuals.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return y_true - y_pred
    
    def compute_residual_stats(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
    ) -> Dict[str, float]:
        """
        Compute statistics on prediction residuals.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
        
        Returns:
            dict: Dictionary containing residual statistics.
        """
        residuals = self.compute_residuals(y_true, y_pred)
        
        stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'median': float(np.median(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q75': float(np.percentile(residuals, 75)),
        }
        
        return stats
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all quality metrics at once.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        
        Returns:
            dict: Dictionary containing all computed metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'r_squared': self.compute_r_squared(y_true, y_pred, sample_weight),
            'explained_variance': self.compute_explained_variance(
                y_true, y_pred, sample_weight
            ),
            'rmse': self.compute_rmse(y_true, y_pred, sample_weight),
            'mae': self.compute_mae(y_true, y_pred, sample_weight),
            'mse': self.compute_mse(y_true, y_pred, sample_weight),
        }
        
        return metrics
