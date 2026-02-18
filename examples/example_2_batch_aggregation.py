"""
Example 2: Batch Processing and Metrics Aggregation
Demonstrates batch computation and aggregation of metrics across samples.
"""

import sys
sys.path.insert(0, 'src')

from data_quality_checker import DataQualityChecker, QualityMetricsAggregator


def main():
    # Initialize the checker and aggregator
    checker = DataQualityChecker()
    aggregator = QualityMetricsAggregator()
    
    print("=" * 60)
    print("Example: Batch Processing and Aggregation")
    print("=" * 60)
    
    # Prepare batch data
    batch_1_predictions = [
        "The cat sat on the mat",
        "Machine learning is powerful",
        "Python is great for data science"
    ]
    
    batch_1_references = [
        "A cat sitting on the mat",
        "ML is very powerful",
        "Python works well for data science"
    ]
    
    batch_2_predictions = [
        "Natural language understanding is complex",
        "Deep neural networks learn hierarchical features"
    ]
    
    batch_2_references = [
        "NLU involves complex computations",
        "Neural networks learn hierarchical information"
    ]
    
    # Example 1: Batch compute metrics
    print("\nBatch 1 - Computing all metrics:")
    print("-" * 40)
    
    batch_1_results = checker.compute_all_metrics(
        predictions=batch_1_predictions,
        references=batch_1_references
    )
    
    print(checker.format_results(batch_1_results, precision=3))
    
    print("\nBatch 2 - Computing all metrics:")
    print("-" * 40)
    
    batch_2_results = checker.compute_all_metrics(
        predictions=batch_2_predictions,
        references=batch_2_references
    )
    
    print(checker.format_results(batch_2_results, precision=3))
    
    # Example 2: Individual sample metrics
    print("\n" + "=" * 60)
    print("Individual Sample Metrics")
    print("=" * 60)
    
    for i, (pred, ref) in enumerate(zip(batch_1_predictions, batch_1_references)):
        print(f"\nSample {i+1}:")
        rouge_score = checker.compute_rouge(pred, ref)
        bleu_score = checker.compute_bleu([pred], [[ref]])
        
        print(f"  ROUGE-1: {rouge_score.get('rouge1', 0):.4f}")
        print(f"  ROUGE-2: {rouge_score.get('rouge2', 0):.4f}")
        print(f"  ROUGE-L: {rouge_score.get('rougeL', 0):.4f}")
        print(f"  BLEU: {bleu_score.get('bleu', 0):.4f}")
    
    # Example 3: Collect individual scores for aggregation
    print("\n" + "=" * 60)
    print("Aggregated Scores Across Dataset")
    print("=" * 60)
    
    all_rouge_scores = []
    all_bleu_scores = []
    
    all_predictions = batch_1_predictions + batch_2_predictions
    all_references = batch_1_references + batch_2_references
    
    for pred, ref in zip(all_predictions, all_references):
        rouge_score = checker.compute_rouge(pred, ref)
        bleu_score = checker.compute_bleu([pred], [[ref]])
        
        all_rouge_scores.append(rouge_score)
        all_bleu_scores.append(bleu_score)
    
    # Aggregate scores
    aggregated_rouge = aggregator.aggregate_rouge_scores(all_rouge_scores, aggregation_type='mean')
    aggregated_bleu = aggregator.aggregate_bleu_scores(all_bleu_scores, aggregation_type='mean')
    
    print("\nAggregated ROUGE Scores (Mean):")
    for key, value in aggregated_rouge.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print("\nAggregated BLEU Scores (Mean):")
    for key, value in aggregated_bleu.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Example 4: Compare aggregation methods
    print("\n" + "=" * 60)
    print("Comparison of Aggregation Methods")
    print("=" * 60)
    
    methods = ['mean', 'median', 'min', 'max']
    
    for method in methods:
        agg_scores = aggregator.aggregate_rouge_scores(all_rouge_scores, aggregation_type=method)
        print(f"\n{method.upper()} aggregation:")
        print(f"  ROUGE-1: {agg_scores.get('rouge1', 0):.4f}")
        print(f"  ROUGE-2: {agg_scores.get('rouge2', 0):.4f}")
        print(f"  ROUGE-L: {agg_scores.get('rougeL', 0):.4f}")


if __name__ == "__main__":
    main()
