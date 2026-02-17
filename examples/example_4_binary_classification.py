"""
Example: Binary Classification Quality Checks

This example demonstrates how to use the BinaryClassifierChecker module
to compute quality metrics for binary classification models.
"""

import numpy as np
from src.binary_classification import BinaryClassifierChecker

# Generate sample data
np.random.seed(42)
n_samples = 100

# Ground truth labels
y_true = np.random.randint(0, 2, n_samples)

# Predicted binary labels
y_pred = np.random.randint(0, 2, n_samples)

# Predicted probabilities (confidence scores for positive class)
y_pred_proba = np.random.rand(n_samples)

# Initialize the checker
checker = BinaryClassifierChecker()

# Check 1: Compute all metrics
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

# Check 2: Compute individual metrics
print("\n" + "=" * 60)
print("INDIVIDUAL METRIC COMPUTATION")
print("=" * 60)

roc_auc = checker.compute_roc_auc(y_true, y_pred_proba)
print(f"\nArea Under ROC Curve: {roc_auc:.4f}")

precision = checker.compute_precision(y_true, y_pred)
print(f"Precision: {precision:.4f}")

recall = checker.compute_recall(y_true, y_pred)
print(f"Recall: {recall:.4f}")

f1 = checker.compute_f1_measure(y_true, y_pred)
print(f"F1-Measure: {f1:.4f}")

log_loss = checker.compute_log_loss(y_true, y_pred_proba)
print(f"Log Loss: {log_loss:.4f}")

# Check 3: Get curve data for plotting
print("\n" + "=" * 60)
print("CURVE DATA FOR VISUALIZATION")
print("=" * 60)

fpr, tpr, thresholds = checker.get_roc_curve_data(y_true, y_pred_proba)
print(f"\nROC Curve Data:")
print(f"  FPR shape: {fpr.shape}")
print(f"  TPR shape: {tpr.shape}")
print(f"  Thresholds shape: {thresholds.shape}")

precision_curve, recall_curve, pr_thresholds = checker.get_pr_curve_data(y_true, y_pred_proba)
print(f"\nPrecision-Recall Curve Data:")
print(f"  Precision shape: {precision_curve.shape}")
print(f"  Recall shape: {recall_curve.shape}")
print(f"  Thresholds shape: {pr_thresholds.shape}")

# Check 4: Display metrics summary
print("\n" + "=" * 60)
metrics_summary = checker.get_metrics_summary()
print(metrics_summary)
