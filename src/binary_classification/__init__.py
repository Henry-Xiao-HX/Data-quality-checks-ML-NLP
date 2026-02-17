"""
Binary Classification Quality Checks Module.

This module provides comprehensive quality metrics for binary classification models.
Includes metrics such as ROC-AUC, Precision, Recall, F1-Score, and more.

Metrics:
- Area Under ROC (AUROC)
- True Positive Rate (TPR/Recall)
- Precision
- F1-Measure
- Logarithmic Loss (Log Loss)
- False Positive Rate (FPR)
- Area Under PR (AUPR)
- Recall
"""

from .metrics import BinaryClassificationMetrics
from .binary_classifier_checker import BinaryClassifierChecker

__all__ = [
    'BinaryClassificationMetrics',
    'BinaryClassifierChecker',
]
