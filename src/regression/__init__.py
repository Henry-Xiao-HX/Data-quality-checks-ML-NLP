"""
Regression Quality Checks Module.

This module provides comprehensive quality metrics for regression models.
Includes metrics such as R-Squared, RMSE, MAE, MSE, and more.

Metrics:
- R-Squared
- Proportion Explained Variance
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
"""

from .metrics import RegressionMetrics
from .regression_checker import RegressionChecker

__all__ = [
    'RegressionMetrics',
    'RegressionChecker',
]
