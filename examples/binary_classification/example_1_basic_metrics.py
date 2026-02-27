"""
Binary Classification Example 1: Basic Quality Metrics
Demonstrates how to use BinaryClassifierChecker to compute
quality metrics for a binary classification model.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from binary_classification import BinaryClassifierChecker

# Generate sample data
np.random.seed(42)
n_samples = 50

y_true = np.random.randint(0, 2, n_samples)
y_pred = np.random.randint(0, 2, n_samples)
y_pred_proba = np.random.rand(n_samples)

checker = BinaryClassifierChecker()

# All metrics at once
print("=" * 60)
print("BINARY CLASSIFICATION QUALITY METRICS")
print("=" * 60)

metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
)

print("\nAll Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"  {metric_name:.<30} {metric_value:.4f}")

# Individual metrics
print("\n" + "=" * 60)
print("INDIVIDUAL METRICS")
print("=" * 60)

print(f"\nROC AUC:    {checker.compute_roc_auc(y_true, y_pred_proba):.4f}")
print(f"Precision:  {checker.compute_precision(y_true, y_pred):.4f}")
print(f"Recall:     {checker.compute_recall(y_true, y_pred):.4f}")
print(f"F1-Measure: {checker.compute_f1_measure(y_true, y_pred):.4f}")
print(f"Log Loss:   {checker.compute_log_loss(y_true, y_pred_proba):.4f}")

print("\n" + "=" * 60)

# Made with Bob
