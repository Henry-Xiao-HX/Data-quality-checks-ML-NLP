# Data Quality Checks for ML/NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Data quality checker for binary classification, regression, and generative AI models. Supports ROUGE/BLEU metrics (text), classification metrics (precision/recall/F1), and regression metrics (MAE/MSE/RMSE/R²).

## Project Structure

```
src/
├── data_quality_checker.py           # Text metrics (ROUGE, BLEU)
├── unified_quality_checker.py        # Multi-task orchestrator
├── utils.py                          # Utility functions
├── binary_classification/
│   ├── binary_classifier_checker.py  # Classification metrics
│   └── metrics.py                    # Helper functions
└── regression/
    ├── regression_checker.py         # Regression metrics
    └── metrics.py                    # Helper functions

tests/
├── conftest.py                       # Pytest fixtures
├── test_data_quality_checker.py
└── test_utils.py

examples/
├── example_1_basic_usage.py
├── example_2_batch_aggregation.py
├── example_3_quality_detection.py
├── example_4_binary_classification.py
├── example_5_regression.py
└── example_6_unified_quality_checker.py
```

## Installation

```bash
git clone https://github.com/Henry-Xiao-HX/Data-quality-checks-ML-NLP.git
cd Data-quality-checks-ML-NLP
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, evaluate, numpy, scikit-learn

## Quick Start

### Text Generation (ROUGE/BLEU)
```python
from src.data_quality_checker import DataQualityChecker

checker = DataQualityChecker()
rouge = checker.compute_rouge("The quick brown fox", "A quick brown fox")
bleu = checker.compute_bleu(["the cat sat"], [["a cat sat"]])
```

### Binary Classification
```python
from src.binary_classification.binary_classifier_checker import BinaryClassifierChecker

checker = BinaryClassifierChecker()
metrics = checker.compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])
# Returns: precision, recall, f1, accuracy
```

### Regression
```python
from src.regression.regression_checker import RegressionChecker

checker = RegressionChecker()
metrics = checker.compute_metrics([2.1, 3.9, 5.8], [2.0, 4.0, 6.0])
# Returns: mae, mse, rmse, r2, mape
```

### Unified Checker
```python
from src.unified_quality_checker import UnifiedQualityChecker

checker = UnifiedQualityChecker()
report = checker.check_quality(predictions, references, task_type='classification')
```

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v

# Using manage.sh
bash manage.sh test-quick        # Without coverage
```

### Run Specific Tests
```bash
python -m pytest tests/test_data_quality_checker.py -v
python -m pytest tests/test_data_quality_checker.py::TestDataQualityChecker::test_compute_rouge -v
python -m unittest tests.test_data_quality_checker
```

### Run Examples
```bash
python examples/example_1_basic_usage.py
python examples/example_2_batch_aggregation.py
python examples/example_3_quality_detection.py
python examples/example_4_binary_classification.py
python examples/example_5_regression.py
python examples/example_6_unified_quality_checker.py

# Or all at once
python run_examples.py
```

## API Reference

### `src.data_quality_checker.DataQualityChecker`

**Methods:**

| Method | Parameters | Returns |
|--------|-----------|---------|
| `compute_rouge(predictions, references, rouge_types=None, use_stemmer=False, use_aggregator=True)` | predictions: str/List[str], references: str/List[str], rouge_types: List[str] | Dict with rouge1, rouge2, rougeL, rougeS scores |
| `compute_bleu(predictions, references, max_order=4, smooth=False)` | predictions: str/List[str], references: List[List[str]]/List[str], max_order: int, smooth: bool | Dict with bleu, precisions, brevity_penalty |
| `compute_all_metrics(predictions, references, compute_rouge=True, compute_bleu=True)` | Same as above | Dict combining all metrics |
| `batch_compute_metrics(predictions_list, references_list, aggregation_type='mean')` | predictions_list: List, references_list: List, aggregation_type: str | Aggregated metrics |

### `src.binary_classification.binary_classifier_checker.BinaryClassifierChecker`

**Methods:**

| Method | Parameters | Returns |
|--------|-----------|---------|
| `compute_metrics(predictions, references, threshold=0.5)` | predictions: List[int/float], references: List[int], threshold: float | Dict with precision, recall, f1, accuracy, specificity, sensitivity |
| `threshold_analysis(prediction_probs, references)` | prediction_probs: List[float], references: List[int] | Dict with optimal threshold and metrics for different thresholds |
| `detect_data_imbalance(references)` | references: List[int] | Dict with class distribution and imbalance ratio |
| `detect_quality_issues(predictions, references)` | predictions: List, references: List | Dict with issue locations and types |

### `src.regression.regression_checker.RegressionChecker`

**Methods:**

| Method | Parameters | Returns |
|--------|-----------|---------|
| `compute_metrics(predictions, references)` | predictions: List[float], references: List[float] | Dict with mae, mse, rmse, r2, mape |
| `residual_analysis(predictions, references)` | predictions: List[float], references: List[float] | Dict with residuals, mean_residual, std_residual |
| `outlier_detection(predictions, references, threshold=2.0)` | predictions: List[float], references: List[float], threshold: float | Dict with outlier indices and values |
| `correlation_analysis(inputs, predictions)` | inputs: List[List[float]], predictions: List[float] | Correlation coefficients |

### `src.unified_quality_checker.UnifiedQualityChecker`

**Methods:**

| Method | Parameters | Returns |
|--------|-----------|---------|
| `check_quality(predictions, references, task_type)` | predictions: List, references: List, task_type: str ('classification'/'regression'/'text_generation') | Comprehensive quality report |

### `src.utils`

**Functions:**

| Function | Parameters | Returns |
|----------|-----------|---------|
| `preprocess_text(text, lowercase=True, remove_punctuation=False)` | text: str, lowercase: bool, remove_punctuation: bool | Preprocessed str |
| `detect_data_quality_issues(predictions, references)` | predictions: List, references: List | Dict with empty, short, duplicate, mismatch issues |
| `calculate_precision_recall_f1(predictions, references)` | predictions: List[str], references: List[str] | Dict with precision, recall, f1 |
| `calculate_length_similarity(predictions, references)` | predictions: List[str], references: List[str] | Dict with avg lengths and ratio |


## Development Commands

```bash
# Setup
bash manage.sh install               # Install dependencies

# Testing
bash manage.sh test-quick            # Run tests without coverage

# Examples
bash manage.sh examples              # Run all examples
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: evaluate` | `pip install evaluate` |
| ROUGE scores are 0 | Check case sensitivity, punctuation, tokenization, language match |
| BLEU score too low | Try ROUGE-L (order-independent), provide multiple references |
| Memory errors | Use batch processing with `batch_compute_metrics()` |

## References

- [ROUGE Metric](https://aclanthology.org/W04-1013/)
- [BLEU Score](https://aclanthology.org/P02-1040/)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/)

## License

MIT - See [LICENSE](LICENSE)

---

**Version:** 0.1.0 | **Author:** Henry Xiao
