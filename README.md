# Data Quality Checks for ML/NLP

Comprehensive data quality checker for ML/NLP models with support for ROUGE and BLEU metrics using the Hugging Face evaluate library.

## Features

- **ROUGE Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-S
- **BLEU Score**: Bilingual Evaluation Understudy metric
- **Batch Processing**: Compute metrics across multiple samples efficiently
- **Metrics Aggregation**: Support for mean, median, min, and max aggregation
- **Quality Issue Detection**: Identify common data quality problems
- **Text Preprocessing**: Built-in utilities for text normalization
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

## Use Cases

1. **Model Evaluation**: Evaluate NLP model outputs against reference texts
2. **Data Quality Monitoring**: Detect issues in generated text data
3. **Summarization Quality**: Assess abstractive summarization quality
4. **Machine Translation**: Evaluate translation quality
5. **Batch Processing**: Efficiently compute metrics across large datasets
6. **Quality Metrics Dashboards**: Create monitoring dashboards with aggregated metrics

## Architecture

```
Data-quality-checks-ML-NLP/
├── src/
│   ├── __init__.py
│   ├── data_quality_checker.py    # Main checker classes
│   └── utils.py                    # Utility functions
├── examples/
│   ├── example_1_basic_usage.py
│   ├── example_2_batch_aggregation.py
│   └── example_3_quality_detection.py
├── tests/
│   ├── test_data_quality_checker.py
│   └── test_utils.py
├── requirements.txt
└── README.md
```

## Performance Considerations

- **Batch Processing**: Use `batch_compute_metrics()` for large datasets
- **Caching**: Results are cached internally by the evaluate library
- **Stemming**: Use `use_stemmer=True` for more lenient matching but slower computation
- **Memory**: ROUGE and BLEU computations are memory-efficient for typical datasets

## Troubleshooting

### Import Error: "evaluate not found"
```bash
pip install evaluate
```

### ROUGE scores are all 0
This usually means there's no overlap between predictions and references. Check:
1. Text preprocessing (case sensitivity, punctuation)
2. Tokenization differences
3. Language or domain differences

### BLEU score is very low
BLEU is sensitive to word order. Common reasons:
1. Different word ordering (use n-gram metrics for position-independent scores)
2. Paraphrasing (consider using ROUGE-L)
3. Multiple valid references help (provide multiple references if available)

## Contributing

Contributions are welcome! Areas for improvement:

- Additional metrics (e.g., METEOR, CIDEr)
- GPU support for faster computation
- Integration with other evaluation frameworks
- Performance optimizations

## License

See LICENSE file for details.

## References

- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/
- ROUGE: https://aclanthology.org/W04-1013/
- BLEU: https://aclanthology.org/P02-1040/
- ROUGE Implementation: https://github.com/google-research/rouge

## Citation

If you use this library, please cite:

```bibtex
@software{data_quality_checks_ml_nlp,
  title={Data Quality Checks for ML/NLP},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Data-quality-checks-ML-NLP}
}
```

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.
