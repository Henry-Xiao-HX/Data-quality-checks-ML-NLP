# Advanced Usage Guide

## Advanced Evaluation Scenarios

### Scenario 1: Multi-Reference Evaluation

When you have multiple valid references for each prediction:

```python
from src.data_quality_checker import DataQualityChecker

checker = DataQualityChecker()

# One prediction, multiple valid references
prediction = ["the cat is on the mat"]

# Multiple reference formats for the same prediction
references = [
    ["a cat is on the mat"],
    ["the feline is on the rug"],
    ["cat sitting on mat"]
]

# BLEU handles multiple references well
bleu_scores = checker.compute_bleu(prediction, references)
print(f"BLEU with multiple refs: {bleu_scores['bleu']:.4f}")

# For ROUGE, can average across references
rouge_avg = {}
for ref in references:
    score = checker.compute_rouge(prediction[0], ref[0])
    for key, val in score.items():
        rouge_avg[key] = rouge_avg.get(key, 0) + val / len(references)

print(f"ROUGE-1 (averaged): {rouge_avg['rouge1']:.4f}")
```

### Scenario 2: Domain-Specific Evaluation

Customize evaluation for specific domains:

```python
from src.data_quality_checker import DataQualityChecker
from src.utils import preprocess_text

checker = DataQualityChecker()

# Medical domain - case and punctuation matters less
def preprocess_medical(text):
    return preprocess_text(text, lowercase=True, remove_punctuation=False)

medical_pred = "The patient's symptoms include HIGH FEVER and COUGH"
medical_ref = "patient symptoms: high fever, cough"

# Clean before comparison
pred_clean = preprocess_medical(medical_pred)
ref_clean = preprocess_medical(medical_ref)

scores = checker.compute_rouge(pred_clean, ref_clean, use_stemmer=True)
print(scores)
```

### Scenario 3: Tracking Metric Trends Over Time

Monitor model performance over iterations:

```python
import json
from datetime import datetime
from src.data_quality_checker import DataQualityChecker, QualityMetricsAggregator

checker = DataQualityChecker()
aggregator = QualityMetricsAggregator()

# Simulate model checkpoints
checkpoints = {
    'checkpoint_1': {
        'predictions': ["text one", "text two", "text three"],
        'references': ["ref one", "ref two", "ref three"]
    },
    'checkpoint_2': {
        'predictions': ["better text one", "better text two", "better text three"],
        'references': ["ref one", "ref two", "ref three"]
    }
}

metrics_history = []

for checkpoint_name, data in checkpoints.items():
    scores = []
    for pred, ref in zip(data['predictions'], data['references']):
        score = checker.compute_rouge(pred, ref)
        scores.append(score)
    
    avg_scores = aggregator.aggregate_rouge_scores(scores)
    
    metrics_history.append({
        'checkpoint': checkpoint_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': avg_scores
    })

# Export history
with open('metrics_history.json', 'w') as f:
    json.dump(metrics_history, f, indent=2)

# Print trend
for entry in metrics_history:
    print(f"Checkpoint: {entry['checkpoint']}")
    print(f"  ROUGE-1: {entry['metrics'].get('rouge1', 0):.4f}")
    print(f"  ROUGE-2: {entry['metrics'].get('rouge2', 0):.4f}")
```

### Scenario 4: Comparative Evaluation (Multiple Models)

Compare performance across different models:

```python
from src.data_quality_checker import DataQualityChecker, QualityMetricsAggregator

checker = DataQualityChecker()
aggregator = QualityMetricsAggregator()

# Test set
test_predictions = {
    'model_A': ["output one", "output two", "output three"],
    'model_B': ["different one", "different two", "different three"],
    'baseline': ["simple one", "simple two", "simple three"]
}

test_references = ["ref one", "ref two", "ref three"]

# Evaluate each model
results = {}
for model_name, predictions in test_predictions.items():
    model_scores = []
    for pred, ref in zip(predictions, test_references):
        score = checker.compute_rouge(pred, ref)
        model_scores.append(score)
    
    avg = aggregator.aggregate_rouge_scores(model_scores)
    results[model_name] = avg

# Compare
print("Model Comparison:")
print("-" * 50)
print(f"{'Model':<15} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
print("-" * 50)
for model, metrics in sorted(results.items()):
    print(f"{model:<15} {metrics['rouge1']:<12.4f} {metrics['rouge2']:<12.4f} {metrics['rougeL']:<12.4f}")
```

### Scenario 5: Statistical Analysis of Quality Metrics

Perform statistical analysis on evaluation results:

```python
import numpy as np
from src.data_quality_checker import DataQualityChecker
from src.utils import detect_data_quality_issues

checker = DataQualityChecker()

# Generate scores
predictions = [f"prediction {i}" for i in range(100)]
references = [f"reference {i}" for i in range(100)]

scores = []
for pred, ref in zip(predictions, references):
    score = checker.compute_rouge(pred, ref)
    scores.append(score['rouge1'])

# Statistical analysis
print("Statistical Summary of ROUGE-1 Scores:")
print(f"  Mean: {np.mean(scores):.4f}")
print(f"  Median: {np.median(scores):.4f}")
print(f"  Std Dev: {np.std(scores):.4f}")
print(f"  Min: {np.min(scores):.4f}")
print(f"  Max: {np.max(scores):.4f}")
print(f"  Quartiles: Q1={np.percentile(scores, 25):.4f}, Q3={np.percentile(scores, 75):.4f}")

# Identify outliers (beyond 1.5 * IQR)
Q1, Q3 = np.percentile(scores, 25), np.percentile(scores, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = [(i, s) for i, s in enumerate(scores) 
            if s < lower_bound or s > upper_bound]
print(f"\nOutliers detected: {len(outliers)}")
for idx, score in outliers[:5]:
    print(f"  Sample {idx}: {score:.4f}")
```

### Scenario 6: Quality Filtering Pipeline

Build a quality filtering pipeline:

```python
from src.data_quality_checker import DataQualityChecker
from src.utils import detect_data_quality_issues, calculate_length_similarity

checker = DataQualityChecker()

class QualityFilter:
    def __init__(self, min_rouge_threshold=0.2, max_length_ratio=2.0):
        self.min_rouge_threshold = min_rouge_threshold
        self.max_length_ratio = max_length_ratio
    
    def filter_predictions(self, predictions, references):
        """Filter low-quality predictions."""
        filtered = []
        rejected = []
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Check for quality issues
            issues = detect_data_quality_issues([pred], [ref])
            if issues:
                rejected.append((i, pred, "Quality issues detected"))
                continue
            
            # Check ROUGE score
            scores = checker.compute_rouge(pred, ref)
            if scores.get('rouge1', 0) < self.min_rouge_threshold:
                rejected.append((i, pred, f"Low ROUGE: {scores['rouge1']:.4f}"))
                continue
            
            # Check length ratio
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            ratio = pred_len / ref_len if ref_len > 0 else 1
            
            if ratio > self.max_length_ratio:
                rejected.append((i, pred, f"Length ratio too high: {ratio:.2f}"))
                continue
            
            filtered.append((i, pred, scores))
        
        return filtered, rejected

# Use the filter
filter = QualityFilter(min_rouge_threshold=0.3)

predictions = [
    "good prediction here",
    "",  # Will be rejected
    "another good one",
    "x" * 1000  # Will be rejected for length
]

references = [
    "good reference",
    "empty ref",
    "another reference",
    "short"
]

accepted, rejected = filter.filter_predictions(predictions, references)

print(f"Accepted: {len(accepted)} / {len(predictions)}")
print(f"Rejected: {len(rejected)} / {len(predictions)}")

for i, pred, reason in rejected:
    print(f"  Sample {i}: {reason}")
```

### Scenario 7: Metric Correlation Analysis

Analyze correlation between different metrics:

```python
import numpy as np
from src.data_quality_checker import DataQualityChecker
from src.utils import calculate_precision_recall_f1

checker = DataQualityChecker()

# Generate paired metrics
predictions = ["pred " * i for i in range(1, 11)]
references = ["ref " * i for i in range(1, 11)]

rouge1_scores = []
rouge2_scores = []
bleu_scores = []
f1_scores = []

for pred, ref in zip(predictions, references):
    rouge = checker.compute_rouge(pred, ref)
    bleu = checker.compute_bleu([pred], [[ref]])
    prf = calculate_precision_recall_f1([pred], [ref])
    
    rouge1_scores.append(rouge.get('rouge1', 0))
    rouge2_scores.append(rouge.get('rouge2', 0))
    bleu_scores.append(bleu.get('bleu', 0))
    f1_scores.append(prf.get('f1', 0))

# Calculate correlations
print("Metric Correlations:")
print(f"ROUGE-1 vs ROUGE-2: {np.corrcoef(rouge1_scores, rouge2_scores)[0,1]:.4f}")
print(f"ROUGE-1 vs BLEU: {np.corrcoef(rouge1_scores, bleu_scores)[0,1]:.4f}")
print(f"ROUGE-1 vs F1: {np.corrcoef(rouge1_scores, f1_scores)[0,1]:.4f}")
print(f"BLEU vs F1: {np.corrcoef(bleu_scores, f1_scores)[0,1]:.4f}")
```

### Scenario 8: Confidence-Based Scoring

Combine multiple metrics for confidence scoring:

```python
from src.data_quality_checker import DataQualityChecker

checker = DataQualityChecker()

class ConfidenceScorer:
    """Combine multiple metrics into confidence score."""
    
    def __init__(self, weights=None):
        if weights is None:
            weights = {
                'rouge1': 0.3,
                'rouge2': 0.2,
                'rougeL': 0.3,
                'bleu': 0.2
            }
        self.weights = weights
    
    def score_prediction(self, prediction, reference):
        """Calculate confidence score."""
        rouge = checker.compute_rouge(prediction, reference)
        bleu = checker.compute_bleu([prediction], [[reference]])
        
        weighted_score = (
            self.weights['rouge1'] * rouge.get('rouge1', 0) +
            self.weights['rouge2'] * rouge.get('rouge2', 0) +
            self.weights['rougeL'] * rouge.get('rougeL', 0) +
            self.weights['bleu'] * bleu.get('bleu', 0)
        )
        
        return weighted_score, {
            'rouge1': rouge.get('rouge1', 0),
            'rouge2': rouge.get('rouge2', 0),
            'rougeL': rouge.get('rougeL', 0),
            'bleu': bleu.get('bleu', 0)
        }

# Use confidence scorer
scorer = ConfidenceScorer()

predictions = [
    "The model works well",
    "Good prediction",
    "Perfect match"
]

references = [
    "The model is good",
    "Great prediction",
    "Perfect match"
]

print("Confidence Scores:")
print("-" * 60)
for pred, ref in zip(predictions, references):
    score, details = scorer.score_prediction(pred, ref)
    print(f"Prediction: {pred}")
    print(f"Reference:  {ref}")
    print(f"Confidence: {score:.4f}")
    print(f"  ROUGE-1: {details['rouge1']:.4f}")
    print(f"  ROUGE-2: {details['rouge2']:.4f}")
    print(f"  ROUGE-L: {details['rougeL']:.4f}")
    print(f"  BLEU: {details['bleu']:.4f}")
    print()
```

## Best Practices

1. **Always preprocess consistently**: Use same preprocessing for predictions and references
2. **Use multiple metrics**: Don't rely on single metric; combine ROUGE, BLEU, and custom metrics
3. **Aggregate strategically**: Use appropriate aggregation method for your use case
4. **Handle edge cases**: Check for empty strings, very short/long texts
5. **Monitor trends**: Track metrics over time to catch regressions
6. **Validate on test set**: Don't optimize for single metric

## Performance Optimization

For large-scale evaluation:

```python
# Use batch processing
results = checker.batch_compute_metrics(
    predictions_list,
    references_list,
    metric_types=['rouge']  # Only compute needed metrics
)

# Pre-process once
from src.utils import preprocess_text
clean_preds = [preprocess_text(p) for p in predictions]
clean_refs = [preprocess_text(r) for r in references]

# Then evaluate
scores = checker.compute_all_metrics(clean_preds, clean_refs)
```
