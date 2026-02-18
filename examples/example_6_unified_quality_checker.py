"""
Example: Unified Data Quality Checker

Demonstrates comprehensive quality assessment across all model types:
- Binary classification
- Regression
- Generative AI text models
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from unified_quality_checker import UnifiedDataQualityChecker

# Initialize unified checker
checker = UnifiedDataQualityChecker()

print("=" * 70)
print("UNIFIED DATA QUALITY CHECKER - COMPREHENSIVE EXAMPLE")
print("=" * 70)

# ============================================================================
# 1. BINARY CLASSIFICATION QUALITY
# ============================================================================
print("\n1. BINARY CLASSIFICATION QUALITY CHECKS")
print("-" * 70)

y_true_clf = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
y_pred_clf = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
y_pred_proba_clf = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.15, 0.85, 0.92, 0.1, 0.6])

clf_metrics = checker.check_binary_classification_quality(
    y_true=y_true_clf,
    y_pred=y_pred_clf,
    y_pred_proba=y_pred_proba_clf,
)

print("Binary Classification Metrics:")
for metric, value in clf_metrics.items():
    print(f"  {metric:.<30} {value:.4f}")

# ============================================================================
# 2. REGRESSION QUALITY
# ============================================================================
print("\n2. REGRESSION QUALITY CHECKS")
print("-" * 70)

np.random.seed(42)
y_true_reg = np.random.rand(10) * 100
y_pred_reg = y_true_reg + np.random.normal(0, 5, 10)

reg_metrics = checker.check_regression_quality(
    y_true=y_true_reg,
    y_pred=y_pred_reg,
)

print("Regression Metrics:")
for metric, value in reg_metrics.items():
    print(f"  {metric:.<30} {value:.4f}")

# ============================================================================
# 3. GENERATIVE AI TEXT MODEL QUALITY
# ============================================================================
print("\n3. GENERATIVE AI TEXT MODEL QUALITY CHECKS")
print("-" * 70)

predictions = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Neural networks are inspired by biological neurons"
]

references = [
    "A quick brown fox jumped over the lazy dog",
    "Machine learning is part of artificial intelligence",
    "Neural networks take inspiration from biological neurons"
]

print("\nBLEU Scores:")
bleu_scores = checker.compute_bleu(predictions, references)
for metric, value in bleu_scores.items():
    print(f"  {metric:.<30} {value}")

print("\nROUGE Scores:")
rouge_scores = checker.compute_rouge(predictions, references)
for metric, value in rouge_scores.items():
    print(f"  {metric:.<30} {value:.4f}")

# ============================================================================
# 4. AVAILABLE MODULES SUMMARY
# ============================================================================
print("\n4. AVAILABLE QUALITY CHECK MODULES")
print("-" * 70)

modules = checker.get_available_modules()
for module, metrics in modules.items():
    print(f"\n{module}:")
    print(f"  Metrics: {metrics}")

print("\n" + "=" * 70)
print("QUALITY ASSESSMENT COMPLETE")
print("=" * 70)
