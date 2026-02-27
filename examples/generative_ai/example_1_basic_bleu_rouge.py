"""
Generative AI Example 1: Basic BLEU and ROUGE Computation
Demonstrates simple and batch usage of BLEU and ROUGE metrics
for evaluating generative AI / LLM text outputs.
"""

import sys
sys.path.insert(0, 'src')

from data_quality_checker import DataQualityChecker, QualityMetricsAggregator


def single_prediction_example(checker):
    print("=" * 60)
    print("Example 1: Single Prediction")
    print("=" * 60)

    prediction = "The quick brown fox jumps over the lazy dog"
    reference = "A quick brown fox jumps over a lazy dog"

    rouge_scores = checker.compute_rouge(prediction, reference)
    print("\nROUGE Scores:")
    print(checker.format_results({'rouge': rouge_scores}))

    bleu_scores = checker.compute_bleu([prediction], [[reference]])
    print("\nBLEU Scores:")
    print(checker.format_results({'bleu': bleu_scores}))


def multiple_predictions_example(checker):
    print("\n" + "=" * 60)
    print("Example 2: Multiple Predictions")
    print("=" * 60)

    predictions = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text data"
    ]

    references = [
        "Machine learning is part of artificial intelligence",
        "Deep neural networks have many layers",
        "NLP focuses on processing natural language"
    ]

    rouge_scores = checker.compute_rouge(predictions, references)
    print("\nROUGE Scores (Aggregated):")
    print(checker.format_results({'rouge': rouge_scores}))

    bleu_scores = checker.compute_bleu(predictions, references)
    print("\nBLEU Scores:")
    print(checker.format_results({'bleu': bleu_scores}))


def all_metrics_example(checker):
    print("\n" + "=" * 60)
    print("Example 3: All Metrics Combined")
    print("=" * 60)

    predictions = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text data"
    ]

    references = [
        "Machine learning is part of artificial intelligence",
        "Deep neural networks have many layers",
        "NLP focuses on processing natural language"
    ]

    all_metrics = checker.compute_all_metrics(
        predictions=predictions,
        references=references,
        compute_rouge=True,
        compute_bleu=True
    )

    print(checker.format_results(all_metrics))


def rouge_with_stemmer_example(checker):
    print("\n" + "=" * 60)
    print("Example 4: ROUGE with Stemmer")
    print("=" * 60)

    pred = "The computing systems are running"
    ref = "Computing system runs"

    rouge_with_stem = checker.compute_rouge(
        pred, ref,
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    print("\nROUGE Scores (with stemming):")
    for key, value in rouge_with_stem.items():
        print(f"  {key}: {value:.4f}")


def batch_aggregation_example(checker):
    print("\n" + "=" * 60)
    print("Example 5: Batch Aggregation Across Samples")
    print("=" * 60)

    aggregator = QualityMetricsAggregator()

    batch_predictions = [
        "The cat sat on the mat",
        "Machine learning is powerful",
        "Python is great for data science",
        "Natural language understanding is complex",
        "Deep neural networks learn hierarchical features"
    ]

    batch_references = [
        "A cat sitting on the mat",
        "ML is very powerful",
        "Python works well for data science",
        "NLU involves complex computations",
        "Neural networks learn hierarchical information"
    ]

    all_rouge_scores = []
    all_bleu_scores = []

    print("\nPer-sample scores:")
    for i, (pred, ref) in enumerate(zip(batch_predictions, batch_references)):
        rouge_score = checker.compute_rouge(pred, ref)
        bleu_score = checker.compute_bleu([pred], [[ref]])

        all_rouge_scores.append(rouge_score)
        all_bleu_scores.append(bleu_score)

        print(f"\n  Sample {i + 1}: \"{pred[:40]}...\"" if len(pred) > 40 else f"\n  Sample {i + 1}: \"{pred}\"")
        print(f"    ROUGE-1: {rouge_score.get('rouge1', 0):.4f}")
        print(f"    ROUGE-2: {rouge_score.get('rouge2', 0):.4f}")
        print(f"    ROUGE-L: {rouge_score.get('rougeL', 0):.4f}")
        print(f"    BLEU:    {bleu_score.get('bleu', 0):.4f}")

    # Aggregate across all samples
    print("\n" + "-" * 40)
    print("Aggregated Scores (Mean across all samples):")

    aggregated_rouge = aggregator.aggregate_rouge_scores(all_rouge_scores, aggregation_type='mean')
    aggregated_bleu = aggregator.aggregate_bleu_scores(all_bleu_scores, aggregation_type='mean')

    print("\n  ROUGE (Mean):")
    for key, value in aggregated_rouge.items():
        if isinstance(value, (int, float)):
            print(f"    {key}: {value:.4f}")

    print("\n  BLEU (Mean):")
    for key, value in aggregated_bleu.items():
        if isinstance(value, (int, float)):
            print(f"    {key}: {value:.4f}")

    # Compare aggregation methods
    print("\n" + "-" * 40)
    print("ROUGE-1 across aggregation methods:")
    for method in ['mean', 'median', 'min', 'max']:
        agg = aggregator.aggregate_rouge_scores(all_rouge_scores, aggregation_type=method)
        print(f"  {method.upper():.<10} {agg.get('rouge1', 0):.4f}")


def main():
    checker = DataQualityChecker()

    single_prediction_example(checker)
    multiple_predictions_example(checker)
    all_metrics_example(checker)
    rouge_with_stemmer_example(checker)
    batch_aggregation_example(checker)


if __name__ == "__main__":
    main()

# Made with Bob
