# Binary Classification Quality Checks

A comprehensive module for computing quality metrics for binary classification models using scikit-learn.

## Overview

The `binary_classification` module provides tools for evaluating binary classification model performance through a collection of standard machine learning metrics.

## Module Structure

```
src/binary_classification/
├── __init__.py                      # Package initialization
├── metrics.py                       # Core metrics calculator
├── binary_classifier_checker.py     # Orchestrator class
└── README.md                        # This file
```

## Features

### Supported Metrics

#### Threshold-Independent Metrics (require probability predictions)
- **Area Under ROC (AUROC)**: Measures the model's ability to distinguish between classes across all thresholds
- **Area Under PR (AUPR)**: Area under the Precision-Recall curve
- **Logarithmic Loss (Log Loss)**: Cross-entropy loss; measures prediction confidence

#### Threshold-Dependent Metrics (require binary predictions)
- **Precision**: TP / (TP + FP) - Accuracy of positive predictions
- **Recall (Sensitivity/TPR)**: TP / (TP + FN) - Coverage of positive instances
- **True Positive Rate (TPR)**: Same as Recall
- **False Positive Rate (FPR)**: FP / (FP + TN) - Error on negative instances
- **F1-Measure**: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean of precision and recall

### Visualization Support
- ROC Curve data (FPR, TPR, thresholds)
- Precision-Recall Curve data (Precision, Recall, thresholds)

## Usage

### Basic Usage

```python
from src.binary_classification import BinaryClassifierChecker
import numpy as np

# Initialize checker
checker = BinaryClassifierChecker()

# Prepare your data
y_true = np.array([0, 1, 1, 0, 1])                    # Ground truth labels
y_pred = np.array([0, 1, 0, 0, 1])                    # Predicted labels
y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])   # Predicted probabilities

# Compute all metrics
metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba
)

print(metrics)
# Output:
# {
#     'roc_auc': 0.95,
#     'true_positive_rate': 0.67,
#     'precision': 1.0,
#     'f1_measure': 0.8,
#     'log_loss': 0.52,
#     'false_positive_rate': 0.0,
#     'auc_pr': 0.96,
#     'recall': 0.67
# }
```

### Individual Metric Computation

```python
# Compute individual metrics
roc_auc = checker.compute_roc_auc(y_true, y_pred_proba)
precision = checker.compute_precision(y_true, y_pred)
recall = checker.compute_recall(y_true, y_pred)
f1 = checker.compute_f1_measure(y_true, y_pred)
log_loss = checker.compute_log_loss(y_true, y_pred_proba)
```

### Getting Curve Data for Visualization

```python
# Get ROC curve data
fpr, tpr, thresholds = checker.get_roc_curve_data(y_true, y_pred_proba)

# Get Precision-Recall curve data
precision_curve, recall_curve, pr_thresholds = checker.get_pr_curve_data(y_true, y_pred_proba)

# Use with matplotlib or other visualization libraries
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

### Using Sample Weights

```python
sample_weights = np.array([1.0, 2.0, 1.5, 1.0, 2.0])

metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    sample_weight=sample_weights
)
```

### Using Metrics Cache

```python
# Cache results for repeated access
metrics = checker.check_quality(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    cache_key='model_v1_test_set'
)

# Clear cache when done
checker.clear_cache()
```

## API Reference

### BinaryClassifierChecker

Main orchestrator class for binary classification quality checks.

#### Methods

- `check_quality(y_true, y_pred, y_pred_proba=None, sample_weight=None, cache_key=None)` → Dict
  - Compute all available metrics
  
- `compute_roc_auc(y_true, y_pred_proba, sample_weight=None)` → float
  - Area Under ROC Curve
  
- `compute_precision(y_true, y_pred)` → float
  - Precision score
  
- `compute_recall(y_true, y_pred)` → float
  - Recall score
  
- `compute_f1_measure(y_true, y_pred)` → float
  - F1-Measure
  
- `compute_log_loss(y_true, y_pred_proba, sample_weight=None)` → float
  - Logarithmic Loss
  
- `get_roc_curve_data(y_true, y_pred_proba)` → Tuple[ndarray, ndarray, ndarray]
  - Returns (fpr, tpr, thresholds)
  
- `get_pr_curve_data(y_true, y_pred_proba)` → Tuple[ndarray, ndarray, ndarray]
  - Returns (precision, recall, thresholds)
  
- `clear_cache()` → None
  - Clear metrics cache
  
- `get_metrics_summary()` → str
  - Display available metrics

### BinaryClassificationMetrics

Low-level metrics calculator with individual metric computation methods.

## Requirements

- numpy
- scikit-learn >= 0.24

## Examples

See [example_4_binary_classification.py](../examples/example_4_binary_classification.py) for a complete working example.

## Error Handling

The module provides validation for:
- Mismatched array lengths
- Missing probability predictions when required
- Invalid label values

Example:
```python
try:
    metrics = checker.check_quality(
        y_true=np.array([0, 1, 1]),
        y_pred=np.array([0, 1])  # Wrong length!
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: y_true and y_pred must have the same length. Got 3 and 2
```

## Notes

- All metrics return values between 0 and 1, except `log_loss` which is unbounded (lower is better)
- Probability-based metrics (AUROC, AUPR, Log Loss) require `y_pred_proba`
- For imbalanced datasets, consider using weighted metrics with `sample_weight`
- Metrics are computed using scikit-learn's robust implementations

## References

- scikit-learn metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Binary Classification Metrics: https://en.wikipedia.org/wiki/Binary_classification
