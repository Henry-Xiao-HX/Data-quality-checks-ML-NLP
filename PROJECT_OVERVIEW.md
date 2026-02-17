# PROJECT OVERVIEW

## Data Quality Checks for ML/NLP

A comprehensive Python library for evaluating the quality of machine learning and NLP model outputs using ROUGE and BLEU metrics with the Hugging Face evaluate library.

## Project Structure

```
Data-quality-checks-ML-NLP/
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── data_quality_checker.py    # Main quality checker classes
│   │   ├── DataQualityChecker         # Primary evaluation class
│   │   └── QualityMetricsAggregator  # Metrics aggregation
│   └── utils.py                   # Utility functions
│       ├── preprocess_text()
│       ├── tokenize_text()
│       ├── get_ngrams()
│       ├── calculate_precision_recall_f1()
│       ├── calculate_length_similarity()
│       └── detect_data_quality_issues()
│
├── examples/                      # Example usage
│   ├── example_1_basic_usage.py
│   ├── example_2_batch_aggregation.py
│   └── example_3_quality_detection.py
│
├── tests/                         # Unit tests
│   ├── test_data_quality_checker.py
│   └── test_utils.py
│
├── docs/                          # Documentation
│   ├── README.md                  # Main documentation
│   ├── QUICKSTART.md              # Quick start guide
│   └── ADVANCED_USAGE.md          # Advanced usage patterns
│
├── requirements.txt               # Project dependencies
├── setup.py                       # Setup configuration
├── pyproject.toml                 # Modern Python project config
├── pytest.ini                     # Pytest configuration
├── Makefile                       # Development tasks
├── LICENSE                        # License file
└── .gitignore                     # Git ignore rules
```

## Key Features

### 1. Metrics Supported

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Unigram overlap (single word matches)
- **ROUGE-2**: Bigram overlap (consecutive word pair matches)
- **ROUGE-L**: Longest common subsequence (considers word order)
- **ROUGE-S**: Skip-bigram overlap (non-consecutive word pairs)

#### BLEU (Bilingual Evaluation Understudy)
- N-gram precision metric
- Brevity penalty for shorter predictions
- Primarily used for translation and generation tasks

### 2. Core Classes

#### `DataQualityChecker`
Main class for computing quality metrics. Provides:
- ROUGE metric computation
- BLEU score computation
- Batch processing capabilities
- Result formatting and display
- Support for text preprocessing options (stemming)

#### `QualityMetricsAggregator`
Aggregates metrics across multiple samples using:
- Mean aggregation
- Median aggregation
- Min/Max aggregation
- Support for both ROUGE and BLEU scores

### 3. Utility Functions
- **Text preprocessing**: Lowercase conversion, punctuation removal, whitespace normalization
- **Tokenization**: Simple whitespace tokenization
- **N-gram extraction**: Support for any n-gram size
- **Quality issue detection**: Identifies empty predictions, duplicates, extreme length variations
- **Basic metrics**: Token-level precision, recall, F1, length similarity

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation Steps

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd Data-quality-checks-ML-NLP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install development tools**
   ```bash
   make install-dev
   # or
   pip install -e ".[dev]"
   ```

## Quick Example

```python
from src.data_quality_checker import DataQualityChecker

# Initialize
checker = DataQualityChecker()

# Evaluate predictions
prediction = "The model performed well on the test set"
reference = "Model achieved good performance"

# Compute metrics
scores = checker.compute_all_metrics(prediction, reference)

# Display results
print(checker.format_results(scores))
```

## Usage Patterns

### Pattern 1: Single Prediction Evaluation
```python
rouge_scores = checker.compute_rouge(prediction, reference)
bleu_scores = checker.compute_bleu([prediction], [[reference]])
```

### Pattern 2: Batch Processing
```python
batch_results = checker.batch_compute_metrics(
    predictions_list, 
    references_list
)
```

### Pattern 3: Metrics Aggregation
```python
all_rouge_scores = aggregator.aggregate_rouge_scores(
    individual_scores, 
    aggregation_type='mean'
)
```

### Pattern 4: Quality Issue Detection
```python
issues = detect_data_quality_issues(predictions, references)
```

## Testing

### Run all tests
```bash
make test
```

### Run specific test
```bash
python -m pytest tests/test_data_quality_checker.py -v
```

### Generate coverage report
```bash
make test
```

## Code Quality

### Linting
```bash
make lint
```

### Code formatting
```bash
make format
```

### Type checking
```bash
mypy src/
```

## Development Workflow

1. **Setup development environment**
   ```bash
   make setup-dev
   ```

2. **Make changes to code**

3. **Run tests**
   ```bash
   make quick-test
   ```

4. **Format and lint**
   ```bash
   make format
   make lint
   ```

5. **Commit changes**

## Key Implementation Details

### Dependencies
- **evaluate**: Hugging Face's evaluation library (ROUGE, BLEU computation)
- **numpy**: Numerical operations for aggregation
- **scikit-learn**: Optional utilities for statistical operations

### Design Principles
1. **Simplicity**: Easy-to-use API for common tasks
2. **Flexibility**: Support for custom preprocessing and aggregation
3. **Performance**: Efficient batch processing
4. **Extensibility**: Clear structure for adding new metrics
5. **Testing**: Comprehensive unit test coverage

### Thread Safety
Current implementation is not thread-safe for concurrent metric computations. For parallel processing, create separate instances per thread.

## Performance Characteristics

- **ROUGE computation**: O(n*m) where n=prediction length, m=reference length
- **BLEU computation**: O(n*m) for n-gram matching
- **Batch processing**: Linear with number of samples
- **Aggregation**: O(n) where n=number of scores

## API Versioning

Current version: **0.1.0** (Alpha)

Semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes

## Future Enhancements

### Planned Features
- [ ] Additional metrics (METEOR, CIDEr, BERTScore)
- [ ] GPU acceleration for faster computation
- [ ] Multi-language support enhancements
- [ ] Visualization utilities
- [ ] Export results to various formats (JSON, CSV, HTML)
- [ ] Integration with MLOps platforms

### Potential Extensions
- Custom metric definitions
- Interactive dashboard for metric visualization
- Real-time monitoring capabilities
- Integration with model serving frameworks

## Common Use Cases

1. **Summarization Evaluation**: Assess quality of abstractive summaries
2. **Machine Translation**: Evaluate translation quality
3. **Text Generation**: Check quality of generated text
4. **Chatbot Responses**: Evaluate response quality
5. **Question Answering**: Assess answer relevance
6. **Data Pipeline Monitoring**: Detect data quality degradation

## Troubleshooting Guide

### Issue: "ModuleNotFoundError: No module named 'evaluate'"
**Solution**: `pip install evaluate`

### Issue: Low ROUGE scores despite seeming similarity
**Solution**: 
- Check case sensitivity (use `lowercase=True`)
- Remove punctuation if needed
- Use stemmer: `use_stemmer=True`

### Issue: BLEU score format issues
**Solution**: Ensure references are in format `List[List[str]]` or `List[str]`

### Issue: Memory issues with large datasets
**Solution**: Use batch processing instead of loading all data at once

## Contributing Guidelines

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Run `make format` and `make lint`
5. Submit pull request

## License

MIT License - See LICENSE file for details

## Citation

If you use this library in research, please cite:

```bibtex
@software{dq_checks_ml_nlp,
  title={Data Quality Checks for ML/NLP},
  author={Xiao, Henry},
  year={2026},
  url={https://github.com/yourusername/Data-quality-checks-ML-NLP}
}
```

## Contact & Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: See README.md and QUICKSTART.md

## Changelog

### Version 0.1.0 (Initial Release)
- Initial implementation of ROUGE metrics
- BLEU score support
- Batch processing capabilities
- Quality issue detection
- Comprehensive testing
- Documentation and examples

## References

1. [ROUGE: A Package for Automatic Evaluation of Summarization](https://aclanthology.org/W04-1013/)
2. [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
3. [Hugging Face Evaluate Library](https://huggingface.co/docs/evaluate/)
4. [NLG Evaluation Metrics](https://github.com/google-research/rouge)

---

**Last Updated**: February 2026
**Maintained by**: Henry Xiao
**Repository**: https://github.com/yourusername/Data-quality-checks-ML-NLP
