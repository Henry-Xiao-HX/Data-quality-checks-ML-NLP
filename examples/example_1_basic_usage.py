"""
Example 1: Basic ROUGE and BLEU Computation
Demonstrates simple usage of ROUGE and BLEU metrics.
"""

import sys
sys.path.insert(0, '../src')

from data_quality_checker import DataQualityChecker


def main():
    # Initialize the checker
    checker = DataQualityChecker()
    
    # Example 1: Single prediction and reference
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
    
    # Example 2: Multiple predictions and references
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
    
    # Example 3: All metrics at once
    print("\n" + "=" * 60)
    print("Example 3: All Metrics Combined")
    print("=" * 60)
    
    all_metrics = checker.compute_all_metrics(
        predictions=predictions,
        references=references,
        compute_rouge=True,
        compute_bleu=True
    )
    
    formatted_output = checker.format_results(all_metrics)
    print(formatted_output)
    
    # Example 4: Different ROUGE types with stemmer
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


if __name__ == "__main__":
    main()
