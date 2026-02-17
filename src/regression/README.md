"""
Regression Module Documentation

README for src/regression/
"""

# Regression Quality Checks

A comprehensive module for computing quality metrics for regression models using scikit-learn.

## Overview

The `regression` module provides tools for evaluating regression model performance through a collection of standard machine learning metrics.

## Module Structure

```
src/regression/
├── __init__.py                  # Package initialization
├── metrics.py                   # Core metrics calculator
├── regression_checker.py        # Orchestrator class
└── README.py                    # This file
```

## Features

### Supported Metrics

#### Model Performance Metrics
- **R-Squared (R²)**: Coefficient of determination; proportion of variance explained
  - Range: -∞ to 1 (higher is better)
  - 1.0 = perfect predictions
  - 0.0 = model performs as well as the mean
  - Negative = model performs worse than horizontal line

- **Explained Variance Score**: Similar to R² but more sensitive to outliers
  - Range: -∞ to 1 (higher is better)
  - Can be larger than R² due to outlier sensitivity

- **Root Mean Squared Error (RMSE)**: Square root of average squared errors
  - Range: 0 to ∞ (lower is better)
  - More sensitive to outliers than MAE
  - Same units as y_true

- **Mean Absolute Error (MAE)**: Average absolute errors
  - Range: 0 to ∞ (lower is better)
  - Less sensitive to outliers than RMSE
  - Same units as y_true

- **Mean Squared Error (MSE)**: Average squared errors
  - Range: 0 to ∞ (lower is better)
  - More sensitive to outliers than MAE
  - Same units as y_true², often used for model training

### Residual Analysis
- **Residuals**: Individual prediction errors (y_true - y_pred)
- **Residual Statistics**: Mean, std, min, max, median, and quartiles of residuals

## Usage

### Basic Usage

```python
from src.regression import RegressionChecker
import numpy as np

# Initialize checker
checker = RegressionChecker()

# Prepare your data
y_true = np.array([10.5, 20.3, 15.8, 25.4, 30.1])  # Ground truth values
y_pred = np.array([10.2, 21.1, 14.9, 26.5, 29.8])  # Predicted values

# Compute all metrics
metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
)

print(metrics)
# Output:
# {
#     'r_squared': 0.9956,
#     'explained_variance': 0.9956,
#     'rmse': 0.7368,
#     'mae': 0.58,
#     'mse': 0.5428
# }
```

### Individual Metric Computation

```python
# Compute individual metrics
r_squared = checker.compute_r_squared(y_true, y_pred)
explained_var = checker.compute_explained_variance(y_true, y_pred)
rmse = checker.compute_rmse(y_true, y_pred)
mae = checker.compute_mae(y_true, y_pred)
mse = checker.compute_mse(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
```

### Residual Analysis

```python
# Get residuals
residuals = checker.get_residuals(y_true, y_pred)

# Get residual statistics
residual_stats = checker.get_residual_stats(y_true, y_pred)
print(residual_stats)
# Output:
# {
#     'mean': -0.12,
#     'std': 0.68,
#     'min': -1.45,
#     'max': 1.32,
#     'median': -0.05,
#     'q25': -0.45,
#     'q75': 0.38
# }
```

### Using Sample Weights

For imbalanced datasets or when certain samples should be weighted more heavily:

```python
sample_weights = np.array([1.0, 2.0, 1.5, 1.0, 2.0])

metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    sample_weight=sample_weights
)
```

### Using Metrics Cache

Cache results for repeated access:

```python
# Cache results
metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    cache_key='model_v1_test_set'
)

# Later access uses cached values:
cached_metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    cache_key='model_v1_test_set'  # Returns cached result
)

# Clear cache when done
checker.clear_cache()
```

## API Reference

### RegressionChecker

Main orchestrator class for regression quality checks.

#### Methods

- `check_quality(y_true, y_pred, sample_weight=None, cache_key=None)` → Dict
  - Compute all available metrics
  
- `compute_r_squared(y_true, y_pred, sample_weight=None)` → float
  - Coefficient of determination
  
- `compute_explained_variance(y_true, y_pred, sample_weight=None)` → float
  - Explained variance score
  
- `compute_rmse(y_true, y_pred, sample_weight=None)` → float
  - Root mean squared error
  
- `compute_mae(y_true, y_pred, sample_weight=None)` → float
  - Mean absolute error
  
- `compute_mse(y_true, y_pred, sample_weight=None)` → float
  - Mean squared error
  
- `get_residuals(y_true, y_pred)` → ndarray
  - Individual prediction errors
  
- `get_residual_stats(y_true, y_pred)` → Dict
  - Statistics on residuals (mean, std, min, max, median, quartiles)
  
- `clear_cache()` → None
  - Clear metrics cache
  
- `get_metrics_summary()` → str
  - Display available metrics

### RegressionMetrics

Low-level metrics calculator with individual metric computation methods.

## Requirements

- numpy
- scikit-learn >= 0.24

## Examples

See [example_5_regression.py](../examples/example_5_regression.py) for a complete working example.

## Error Handling

The module provides validation for:
- Mismatched array lengths

Example:
```python
try:
    metrics = checker.check_quality(
        y_true=np.array([10, 20, 30]),
        y_pred=np.array([10, 20])  # Wrong length!
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: y_true and y_pred must have the same length. Got 3 and 2
```

## Interpretation Guide

### When to use which metric?

- **R²**: Use as your primary metric; directly interpretable as proportion of variance explained
- **Explained Variance**: Use when you want to understand variance in predictions; more sensitive to outliers than R²
- **RMSE**: Use when large errors are particularly undesirable (outlier-sensitive)
- **MAE**: Use when all errors are equally important (robust to outliers)
- **MSE**: Use as intermediate metric; used in training most regression models

### Typical Ranges

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| R²     | 0.90-1.00 | 0.70-0.90 | 0.50-0.70 | < 0.50 |
| RMSE   | Low (< 10% of y_true range) | Moderate (10-20%) | High (20-30%) | Very High (> 30%) |
| MAE    | Low (< 10% of y_true range) | Moderate (10-20%) | High (20-30%) | Very High (> 30%) |

(Actual thresholds depend on your specific problem domain)

## Residual Diagnostics

Good regression models typically have residuals that:
- Have a mean close to 0
- Are normally distributed
- Have constant variance (homoscedastic)
- Show no correlation with predicted values
- Contain no systematic patterns

Use `get_residual_stats()` and plot residuals against predictions to diagnose issues.

## References

- scikit-learn metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- Regression Metrics: https://en.wikipedia.org/wiki/Regression_analysis
