"""
Regression Example 2: Advanced Metrics, Residual Analysis, and Caching
Demonstrates residual analysis, sample weights, and metrics caching
for regression model evaluation.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from regression import RegressionChecker

# Generate sample data
np.random.seed(42)
n_samples = 100

y_true = np.random.rand(n_samples) * 100
noise = np.random.normal(0, 5, n_samples)
y_pred = y_true + noise

checker = RegressionChecker()

# Check 1: All metrics
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

# Check 2: Individual metrics
print("\n" + "=" * 60)
print("INDIVIDUAL METRIC COMPUTATION")
print("=" * 60)

print(f"\nR-Squared:          {checker.compute_r_squared(y_true, y_pred):.4f}")
print(f"Explained Variance: {checker.compute_explained_variance(y_true, y_pred):.4f}")
print(f"RMSE:               {checker.compute_rmse(y_true, y_pred):.4f}")
print(f"MAE:                {checker.compute_mae(y_true, y_pred):.4f}")
print(f"MSE:                {checker.compute_mse(y_true, y_pred):.4f}")

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

# Check 4: Sample weights
print("\n" + "=" * 60)
print("WEIGHTED METRICS")
print("=" * 60)

sample_weights = np.ones(n_samples)
sample_weights[:20] = 2.0  # Higher weight for first 20 samples

weighted_metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    sample_weight=sample_weights,
)

print("\nWeighted Metrics (first 20 samples weighted 2x):")
for metric_name, metric_value in weighted_metrics.items():
    print(f"  {metric_name:.<30} {metric_value:.4f}")

# Check 5: Metrics caching
print("\n" + "=" * 60)
print("METRICS CACHING")
print("=" * 60)

metrics_cached = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    cache_key='model_v1_test_set'
)
print("\nMetrics cached with key 'model_v1_test_set'")

metrics_from_cache = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    cache_key='model_v1_test_set'
)
print("Retrieved metrics from cache (uses stored values)")

checker.clear_cache()
print("Cache cleared")

# Check 6: Metrics summary
print("\n" + "=" * 60)
metrics_summary = checker.get_metrics_summary()
print(metrics_summary)

# Made with Bob
