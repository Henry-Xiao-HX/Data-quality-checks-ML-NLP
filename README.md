# Data Quality Checks for ML/NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive data quality checker for ML/NLP models with support for ROUGE and BLEU metrics using the Hugging Face evaluate library.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [API Documentation](#api-documentation)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **ROUGE Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-S for text summarization evaluation
- **BLEU Score**: Bilingual Evaluation Understudy metric for machine translation and text generation
- **Batch Processing**: Efficiently compute metrics across multiple samples
- **Metrics Aggregation**: Support for mean, median, min, and max aggregation
- **Quality Issue Detection**: Identify common data quality problems
- **Text Preprocessing**: Built-in utilities for text normalization and tokenization
- **Flexible Evaluation**: Multi-reference support, customizable preprocessing, and stemming options
- **Comprehensive Testing**: Unit tests with high code coverage

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Data-quality-checks-ML-NLP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `evaluate`: Hugging Face's evaluation library for metrics
- `numpy`: For numerical operations
- `scikit-learn`: For additional utilities

## Quick Start

### Basic Usage

```python
from src.data_quality_checker import DataQualityChecker

# Initialize the checker
checker = DataQualityChecker()

# Compute ROUGE metrics
rouge_scores = checker.compute_rouge(
    predictions="The quick brown fox jumps",
    references="A quick brown fox jumps"
)

# Compute BLEU score
bleu_scores = checker.compute_bleu(
    predictions=["the cat sat"],
    references=["a cat sat"]
)

# Compute all metrics
all_metrics = checker.compute_all_metrics(
    predictions="Machine learning is powerful",
    references="ML is very powerful"
)
```

### Supported Metrics

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **ROUGE-1**: Unigram overlap between prediction and reference
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-S**: Skip-bigram overlap (non-consecutive bigrams)

#### BLEU (Bilingual Evaluation Understudy)

- Measures n-gram precision between prediction and reference
- Includes brevity penalty for shorter predictions
- Scores range from 0 to 1

### Examples

Run the provided examples:

```bash
python examples/example_1_basic_usage.py
python examples/example_2_batch_aggregation.py
python examples/example_3_quality_detection.py
```

## API Documentation

### DataQualityChecker

Main class for computing quality metrics.

#### Methods

##### `compute_rouge(predictions, references, rouge_types=None, use_stemmer=False, use_aggregator=True)`

Compute ROUGE metrics.

**Parameters:**
- `predictions` (str or List[str]): Model predictions
- `references` (str or List[str]): Reference texts
- `rouge_types` (List[str], optional): Types of ROUGE to compute. Defaults to all
- `use_stemmer` (bool): Whether to use stemmer (default: False)
- `use_aggregator` (bool): Whether to aggregate scores (default: True)

**Returns:** Dictionary with ROUGE scores

**Example:**
```python
scores = checker.compute_rouge(
    predictions="The cat sat on the mat",
    references="A cat was sitting on the mat"
)
print(scores)  # {'rouge1': 0.67, 'rouge2': 0.33, 'rougeL': 0.67, 'rougeS': 0.33}
```

##### `compute_bleu(predictions, references, max_order=4, smooth=False)`

Compute BLEU score.

**Parameters:**
- `predictions` (str or List[str]): Model predictions
- `references` (List[List[str]] or List[str]): Reference texts
- `max_order` (int): Maximum n-gram order (default: 4)
- `smooth` (bool): Whether to apply smoothing (default: False)

**Returns:** Dictionary with BLEU score

**Example:**
```python
scores = checker.compute_bleu(
    predictions=["the cat sat"],
    references=["a cat sat"]
)
print(scores)  # {'bleu': 0.75, 'precisions': [...], ...}
```

##### `compute_all_metrics(predictions, references, compute_rouge=True, compute_bleu=True, ...)`

Compute all supported metrics at once.

**Returns:** Dictionary with all metrics

### QualityMetricsAggregator

Aggregate metrics across multiple samples.

#### Methods

##### `aggregate_rouge_scores(scores_list, aggregation_type='mean')`

Aggregate ROUGE scores.

**Parameters:**
- `scores_list` (List[Dict]): List of ROUGE score dictionaries
- `aggregation_type` (str): 'mean', 'median', 'min', or 'max'

**Returns:** Aggregated scores

##### `aggregate_bleu_scores(scores_list, aggregation_type='mean')`

Aggregate BLEU scores.

Same parameters and returns as `aggregate_rouge_scores`.

### Utility Functions

#### `preprocess_text(text, lowercase=True, remove_punctuation=False)`

Preprocess text for quality checks.

```python
from src.utils import preprocess_text

cleaned = preprocess_text("HELLO, World!", lowercase=True, remove_punctuation=True)
# Output: "hello world"
```

#### `detect_data_quality_issues(predictions, references, ...)`

Detect common data quality problems.

```python
from src.utils import detect_data_quality_issues

issues = detect_data_quality_issues(predictions, references)
# Returns: {
#     'empty_predictions': [...],
#     'very_short_predictions': [...],
#     'duplicate_predictions': [...],
#     ...
# }
```

#### `calculate_precision_recall_f1(predictions, references)`

Calculate token-level overlap metrics.

```python
from src.utils import calculate_precision_recall_f1

metrics = calculate_precision_recall_f1(predictions, references)
# Returns: {'precision': 0.8, 'recall': 0.75, 'f1': 0.77}
```

#### `calculate_length_similarity(predictions, references)`

Calculate text length similarity metrics.

```python
from src.utils import calculate_length_similarity

metrics = calculate_length_similarity(predictions, references)
# Returns: {
#     'avg_pred_length': 12.5,
#     'avg_ref_length': 15.0,
#     'length_ratio': 0.833
# }
```

## Running Tests

Run all unit tests:

```bash
cd tests
python -m unittest discover -s . -p "test_*.py"
```

Run specific test file:

```bash
python -m unittest test_data_quality_checker.py
```

Run specific test class:

```bash
python -m unittest test_data_quality_checker.TestDataQualityChecker
```



## Advanced Usage

### Multi-Reference Evaluation

When you have multiple valid references per prediction:

```python
checker = DataQualityChecker()

# One prediction, multiple references
prediction = "the cat is on the mat"
references = ["a cat is on the mat", "the feline is on the rug"]

# BLEU handles multiple references well
scores = checker.compute_bleu([prediction], [[r for r in references]])
```

### Domain-Specific Evaluation

Customize for specific domains:

```python
from src.utils import preprocess_text
from src.data_quality_checker import DataQualityChecker

checker = DataQualityChecker()

def preprocess_medical(text):
    return preprocess_text(text, lowercase=True, remove_punctuation=False)

medical_pred = "The patient's symptoms include HIGH FEVER and COUGH"
medical_ref = "patient symptoms: high fever, cough"

pred_clean = preprocess_medical(medical_pred)
ref_clean = preprocess_medical(medical_ref)

scores = checker.compute_rouge(pred_clean, ref_clean, use_stemmer=True)
```

See README for more advanced examples: quality filtering pipelines, confidence scoring, statistical analysis, tracking trends over time, and comparative model evaluation.

## Project Structure

```
Data-quality-checks-ML-NLP/
├── src/
│   ├── __init__.py                    # Package exports
│   ├── data_quality_checker.py        # Main driver/orchestrator
│   ├── rouge.py                       # ROUGE metrics
│   ├── bleu.py                        # BLEU metrics
│   └── utils.py                       # Utility functions
│
├── scripts/                           # Bash orchestration scripts
│   ├── install.sh                     # Install dependencies
│   ├── install-dev.sh                 # Install dev dependencies
│   ├── test.sh                        # Run tests with coverage
│   ├── lint.sh                        # Lint code
│   ├── format.sh                      # Format code
│   ├── clean.sh                       # Clean build files
│   ├── run-examples.sh                # Run all examples
│   └── help.sh                        # Show help
│
├── examples/
│   ├── example_1_basic_usage.py
│   ├── example_2_batch_aggregation.py
│   └── example_3_quality_detection.py
│
├── tests/
│   ├── test_data_quality_checker.py
│   └── test_utils.py
│
├── orchestrate.sh                     # Master orchestration script
├── README.md                          # This file
├── LICENSE                            # MIT License
└── requirements.txt                   # Dependencies
```

## Development

All commands available through bash scripts or orchestration:

```bash
# Installation and setup
bash orchestrate.sh install              # Install dependencies
bash orchestrate.sh install-dev          # Install dev tools
bash orchestrate.sh setup-dev            # Full dev environment

# Testing
bash orchestrate.sh test                 # Run tests with coverage
bash orchestrate.sh quick-test           # Run tests without coverage
bash orchestrate.sh test-specific FILE   # Run specific test

# Code quality
bash orchestrate.sh lint                 # Lint code
bash orchestrate.sh format               # Format code
bash orchestrate.sh clean                # Clean build files

# Examples
bash orchestrate.sh run-examples         # Run all examples
bash orchestrate.sh run-example FILE     # Run specific example
```

## Troubleshooting

### Import Error: "evaluate not found"

**Solution:** Install the evaluate library
```bash
pip install evaluate
```

### ROUGE scores are all 0

This means there's no overlap between predictions and references. Check:
1. **Case sensitivity**: Use `lowercase=True` when creating checker
2. **Punctuation**: Remove punctuation with preprocessing utilities
3. **Tokenization differences**: Verify that text is properly tokenized
4. **Language mismatch**: Ensure predictions and references are in same language

### BLEU score is very low

BLEU is sensitive to word order. Common reasons:
1. **Different word ordering**: Consider using ROUGE-L for order-independent match
2. **Paraphrasing**: Different wording of same content reduces BLEU
3. **Multiple valid references**: Provide multiple reference variations

### Memory issues with large datasets

**Solution:** Use batch processing with `batch_compute_metrics()`

## Contributing

Contributions are welcome! Areas for improvement:

- Additional metrics (METEOR, CIDEr, BERTScore)
- GPU acceleration
- Multi-language support
- Visualization utilities
- MLOps platform integration

**Contribution Process:**
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `bash orchestrate.sh format` and `bash orchestrate.sh lint`
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) file for details

## References

- [ROUGE: A Package for Automatic Evaluation of Summarization](https://aclanthology.org/W04-1013/)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- [Hugging Face Evaluate Library](https://huggingface.co/docs/evaluate/)

## Citation

If you use this library, please cite:

```bibtex
@software{dq_checks_ml_nlp,
  title={Data Quality Checks for ML/NLP},
  author={Xiao, Henry},
  year={2026},
  url={https://github.com/yourusername/Data-quality-checks-ML-NLP}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the `examples/` directory
- Review the advanced usage patterns in this README

---

**Version:** 0.1.0  
**Last Updated:** February 2026  
**Maintained by:** Henry Xiao
