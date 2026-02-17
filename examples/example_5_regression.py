"""
Example: Regression Quality Checks

This example demonstrates how to use the RegressionChecker module
to compute quality metrics for regression models.
"""

import numpy as np
from src.regression import RegressionChecker

# Generate sample data
np.random.seed(42)
n_samples = 100

# Ground truth values
y_true = np.random.rand(n_samples) * 100

# Predicted values (with some noise)
noise = np.random.normal(0, 5, n_samples)
y_pred = y_true + noise

# Initialize the checker
checker = RegressionChecker()

# Check 1: Compute all metrics
print("=" * 60)
print("REGRESSION QUALITY METRICS")
print("=" * 60)

metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
)

print("\nAll Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"  {metric_name:.<30} {metric_value:.4f}")

# Check 2: Compute individual metrics
print("\n" + "=" * 60)
print("INDIVIDUAL METRIC COMPUTATION")
print("=" * 60)

r_squared = checker.compute_r_squared(y_true, y_pred)
print(f"\nR-Squared: {r_squared:.4f}")

explained_var = checker.compute_explained_variance(y_true, y_pred)
print(f"Explained Variance: {explained_var:.4f}")

rmse = checker.compute_rmse(y_true, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

mae = checker.compute_mae(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

mse = checker.compute_mse(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Check 3: Residual analysis
print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)

residuals = checker.get_residuals(y_true, y_pred)
print(f"\nFirst 10 residuals: {residuals[:10]}")

residual_stats = checker.get_residual_stats(y_true, y_pred)
print("\nResidual Statistics:")
for stat_name, stat_value in residual_stats.items():
    print(f"  {stat_name:.<20} {stat_value:.4f}")

# Check 4: Using sample weights
print("\n" + "=" * 60)
print("WEIGHTED METRICS (Example)")
print("=" * 60)

sample_weights = np.ones(n_samples)
# Give higher weight to first 20 samples
sample_weights[:20] = 2.0

weighted_metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    sample_weight=sample_weights,
)

print("\nWeighted Metrics:")
for metric_name, metric_value in weighted_metrics.items():
    print(f"  {metric_name:.<30} {metric_value:.4f}")

# Check 5: Caching example
print("\n" + "=" * 60)
print("METRICS CACHING")
print("=" * 60)

metrics_cached = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    cache_key='model_v1_test_set'
)
print("\nMetrics cached with key 'model_v1_test_set'")

# Retrieve from cache (same key will use cached results)
metrics_from_cache = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    cache_key='model_v1_test_set'
)
print("Retrieved metrics from cache (uses stored values)")

checker.clear_cache()
print("Cache cleared")

# Check 6: Display metrics summary
print("\n" + "=" * 60)
metrics_summary = checker.get_metrics_summary()
print(metrics_summary)
