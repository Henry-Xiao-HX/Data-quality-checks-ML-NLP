"""
Regression Example 1: Basic Quality Metrics
Demonstrates how to use RegressionChecker to compute
quality metrics for a regression model.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from regression import RegressionChecker

# Generate sample data
np.random.seed(42)
n_samples = 50

y_true = np.random.rand(n_samples) * 100
noise = np.random.normal(0, 5, n_samples)
y_pred = y_true + noise

checker = RegressionChecker()

# All metrics at once
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

# Individual metrics
print("\n" + "=" * 60)
print("INDIVIDUAL METRICS")
print("=" * 60)

print(f"\nR-Squared:          {checker.compute_r_squared(y_true, y_pred):.4f}")
print(f"Explained Variance: {checker.compute_explained_variance(y_true, y_pred):.4f}")
print(f"RMSE:               {checker.compute_rmse(y_true, y_pred):.4f}")
print(f"MAE:                {checker.compute_mae(y_true, y_pred):.4f}")
print(f"MSE:                {checker.compute_mse(y_true, y_pred):.4f}")

print("\n" + "=" * 60)

# Made with Bob
