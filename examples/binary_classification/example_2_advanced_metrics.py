"""
Binary Classification Example 2: Advanced Metrics and Curve Data
Demonstrates individual metric computation, ROC/PR curve data,
and metrics caching for binary classification models.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from binary_classification import BinaryClassifierChecker

# Generate sample data
np.random.seed(42)
n_samples = 100

y_true = np.random.randint(0, 2, n_samples)
y_pred = np.random.randint(0, 2, n_samples)
y_pred_proba = np.random.rand(n_samples)

checker = BinaryClassifierChecker()

# Check 1: All metrics
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

# Check 2: Individual metrics
print("\n" + "=" * 60)
print("INDIVIDUAL METRIC COMPUTATION")
print("=" * 60)

print(f"\nArea Under ROC Curve: {checker.compute_roc_auc(y_true, y_pred_proba):.4f}")
print(f"Precision:            {checker.compute_precision(y_true, y_pred):.4f}")
print(f"Recall:               {checker.compute_recall(y_true, y_pred):.4f}")
print(f"F1-Measure:           {checker.compute_f1_measure(y_true, y_pred):.4f}")
print(f"Log Loss:             {checker.compute_log_loss(y_true, y_pred_proba):.4f}")

# Check 3: Curve data for plotting
print("\n" + "=" * 60)
print("CURVE DATA FOR VISUALIZATION")
print("=" * 60)

fpr, tpr, thresholds = checker.get_roc_curve_data(y_true, y_pred_proba)
print(f"\nROC Curve Data:")
print(f"  FPR shape:        {fpr.shape}")
print(f"  TPR shape:        {tpr.shape}")
print(f"  Thresholds shape: {thresholds.shape}")

precision_curve, recall_curve, pr_thresholds = checker.get_pr_curve_data(y_true, y_pred_proba)
print(f"\nPrecision-Recall Curve Data:")
print(f"  Precision shape:  {precision_curve.shape}")
print(f"  Recall shape:     {recall_curve.shape}")
print(f"  Thresholds shape: {pr_thresholds.shape}")

# Check 4: Metrics summary
print("\n" + "=" * 60)
metrics_summary = checker.get_metrics_summary()
print(metrics_summary)

# Made with Bob
