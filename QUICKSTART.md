# Quick Start Guide

## Installation (5 minutes)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Data-quality-checks-ML-NLP
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or with development tools:
```bash
make install-dev
```

### Step 3: Verify Installation
```bash
python -c "from src.data_quality_checker import DataQualityChecker; print('Installation successful!')"
```

## Your First Quality Check (2 minutes)

Create a file called `first_check.py`:

```python
from src.data_quality_checker import DataQualityChecker

# Initialize
checker = DataQualityChecker()

# Your data
predictions = "The model performed well on the test set"
references = "Model achieved good performance"

# Compute metrics
scores = checker.compute_all_metrics(predictions, references)

# Display results
print(checker.format_results(scores))
```

Run it:
```bash
python first_check.py
```

## Common Tasks

### Task 1: Evaluate a Summary

```python
from src.data_quality_checker import DataQualityChecker

checker = DataQualityChecker()

# Article summary task
summary = "COVID-19 pandemic has affected millions worldwide"
reference = "The COVID-19 pandemic has had a significant impact on global populations"

scores = checker.compute_rouge(summary, reference)
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-2: {scores['rouge2']:.4f}")
```

### Task 2: Evaluate a Translation

```python
checker = DataQualityChecker()

translations = [
    "The cat sat on the mat",
    "A dog runs in the park"
]

references = [
    "Un gato estaba sentado en la alfombra",
    "Un perro corre en el parque"
]

# Compute BLEU
bleu_scores = checker.compute_bleu(translations, references)
print(f"BLEU Score: {bleu_scores['bleu']:.4f}")
```

### Task 3: Batch Evaluate Multiple Outputs

```python
from src.data_quality_checker import DataQualityChecker, QualityMetricsAggregator

checker = DataQualityChecker()
aggregator = QualityMetricsAggregator()

predictions = ["text1", "text2", "text3"]
references = ["ref1", "ref2", "ref3"]

# Compute individual scores
scores = []
for pred, ref in zip(predictions, references):
    score = checker.compute_rouge(pred, ref)
    scores.append(score)

# Aggregate
avg_scores = aggregator.aggregate_rouge_scores(scores, aggregation_type='mean')
print(f"Average ROUGE-1: {avg_scores['rouge1']:.4f}")
```

### Task 4: Detect Quality Issues

```python
from src.utils import detect_data_quality_issues

predictions = [
    "",  # Empty
    "short",  # Too short
    "valid prediction text here"
]

references = [
    "reference",
    "another reference",
    "another valid reference here"
]

issues = detect_data_quality_issues(predictions, references)
for issue_type, issues_list in issues.items():
    print(f"{issue_type}:")
    for issue in issues_list:
        print(f"  - {issue}")
```

## Using Different ROUGE Types

```python
checker = DataQualityChecker()

text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A quick brown fox jumps over a lazy dog"

# Individual ROUGE types
rouge1 = checker.compute_rouge(text1, text2, rouge_types=['rouge1'])
rouge2 = checker.compute_rouge(text1, text2, rouge_types=['rouge2'])
rougeL = checker.compute_rouge(text1, text2, rouge_types=['rougeL'])

print(f"ROUGE-1 (unigrams): {rouge1['rouge1']:.4f}")
print(f"ROUGE-2 (bigrams): {rouge2['rouge2']:.4f}")
print(f"ROUGE-L (LCS): {rougeL['rougeL']:.4f}")
```

## Using Text Preprocessing

```python
from src.utils import preprocess_text, detect_data_quality_issues

# Preprocess before checking
predictions = ["The CAT sat, on the MAT!"]
references = ["a cat sat on the mat"]

# Clean text
cleaned_pred = [preprocess_text(p, lowercase=True, remove_punctuation=True) for p in predictions]
cleaned_ref = [preprocess_text(r, lowercase=True, remove_punctuation=True) for r in references]

# Now check quality
checker = DataQualityChecker()
scores = checker.compute_rouge(cleaned_pred, cleaned_ref)
print(scores)
```

## Stemming for More Lenient Matching

```python
checker = DataQualityChecker()

pred = "The models are training"
ref = "The model is trained"

# Without stemming
scores1 = checker.compute_rouge(pred, ref, use_stemmer=False)
print(f"Without stemmer: {scores1['rouge1']:.4f}")

# With stemming (matches 'train', 'training', 'trained')
scores2 = checker.compute_rouge(pred, ref, use_stemmer=True)
print(f"With stemmer: {scores2['rouge1']:.4f}")
```

## Running Examples

### Run all examples at once:
```bash
make run-examples
```

### Run a specific example:
```bash
python examples/example_1_basic_usage.py
python examples/example_2_batch_aggregation.py
python examples/example_3_quality_detection.py
```

## Running Tests

### All tests:
```bash
make test
```

### Specific test:
```bash
python -m pytest tests/test_data_quality_checker.py -v
```

### With coverage:
```bash
make test
```

## Metrics Reference

| Metric | Purpose | Use Case |
|--------|---------|----------|
| ROUGE-1 | Unigram overlap | General text similarity |
| ROUGE-2 | Bigram overlap | Phrase-level similarity |
| ROUGE-L | Longest common subsequence | Word order preservation |
| ROUGE-S | Skip-bigram | Non-consecutive word pairs |
| BLEU | N-gram precision | Translation and MT evaluation |

## Performance Tips

1. **Batch Processing**: For large datasets, use batch operations
   ```python
   results = checker.batch_compute_metrics(pred_list, ref_list)
   ```

2. **Selective Metrics**: Only compute needed metrics
   ```python
   scores = checker.compute_all_metrics(pred, ref, compute_bleu=False)
   ```

3. **Aggregation**: Use aggregation for dataset-level statistics
   ```python
   agg = aggregator.aggregate_rouge_scores(scores, 'mean')
   ```

## Troubleshooting

**Q: I'm getting "ModuleNotFoundError: No module named 'evaluate'"**
```bash
pip install evaluate
```

**Q: ROUGE scores are 0**
- Check for case sensitivity differences
- Verify token matching
- Use stemmer: `use_stemmer=True`

**Q: BLEU score is very low**
- Ensure references format is correct (List[List[str]])
- Check for word order differences
- Consider using ROUGE-L for better results

## Next Steps

1. Read the [full documentation](README.md)
2. Check out [example files](examples/)
3. Review [API documentation](README.md#api-documentation)
4. Explore [advanced use cases](README.md#use-cases)

## Need Help?

- Check the examples in `examples/` directory
- Read the docstrings in source code
- Review test cases for usage patterns
- Open an issue for bugs or questions

Happy evaluating! 🚀
