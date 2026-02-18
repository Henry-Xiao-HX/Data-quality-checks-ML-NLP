"""
Example 3: Data Quality Issue Detection
Demonstrates detection of common data quality problems.
"""

import sys
sys.path.insert(0, 'src')

from data_quality_checker import DataQualityChecker
from utils import (
    detect_data_quality_issues,
    calculate_precision_recall_f1,
    calculate_length_similarity,
    preprocess_text,
    tokenize_text
)


def main():
    checker = DataQualityChecker()
    
    print("=" * 60)
    print("Example: Data Quality Issue Detection")
    print("=" * 60)
    
    # Example 1: Dataset with quality issues
    print("\nDataset with Intentional Issues:")
    print("-" * 40)
    
    predictions = [
        "The quick brown fox",  # Too short
        "",  # Empty
        "Machine learning models are trained on large datasets",  # Normal
        "Machine learning models are trained on large datasets",  # Duplicate
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. " +
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog",  # Too long
        "data science analytics",  # Normal
    ]
    
    references = [
        "A quick brown fox",
        "Empty reference",
        "Machine learning uses large datasets",
        "Training requires big data",
        "Short",
        "Data science and analytics"
    ]
    
    # Detect issues
    issues = detect_data_quality_issues(predictions, references)
    
    print("\nDetected Issues:")
    for issue_type, issue_list in issues.items():
        print(f"\n{issue_type}:")
        for issue in issue_list:
            print(f"  - {issue}")
    
    # Example 2: Token-level analysis
    print("\n" + "=" * 60)
    print("Token-Level Analysis")
    print("=" * 60)
    
    test_predictions = [
        "The model performed well",
        "Good performance observed",
        "Model achieved high accuracy"
    ]
    
    test_references = [
        "The model worked well",
        "Performance was good",
        "High accuracy achieved"
    ]
    
    prf = calculate_precision_recall_f1(test_predictions, test_references)
    print("\nToken Overlap Metrics:")
    print(f"  Precision: {prf['precision']:.4f}")
    print(f"  Recall: {prf['recall']:.4f}")
    print(f"  F1 Score: {prf['f1']:.4f}")
    
    # Example 3: Length similarity analysis
    print("\n" + "=" * 60)
    print("Length Analysis")
    print("=" * 60)
    
    length_sim = calculate_length_similarity(test_predictions, test_references)
    print("\nLength Similarity Metrics:")
    print(f"  Average Prediction Length: {length_sim['avg_pred_length']:.2f} tokens")
    print(f"  Average Reference Length: {length_sim['avg_ref_length']:.2f} tokens")
    print(f"  Length Ratio (Pred/Ref): {length_sim['length_ratio']:.4f}")
    
    # Example 4: Comprehensive quality report
    print("\n" + "=" * 60)
    print("Comprehensive Quality Report")
    print("=" * 60)
    
    clean_predictions = [
        "The machine learning model was trained successfully",
        "Deep learning networks require significant computational power",
        "Natural language processing is essential for chatbots"
    ]
    
    clean_references = [
        "ML models were successfully trained",
        "Deep neural networks need lots of computing resources",
        "NLP is important for conversational AI"
    ]
    
    # Compute all metrics
    all_metrics = checker.compute_all_metrics(
        predictions=clean_predictions,
        references=clean_references
    )
    
    # Check for issues
    quality_issues = detect_data_quality_issues(
        clean_predictions,
        clean_references,
        min_rouge_threshold=0.2,
        max_length_ratio_threshold=(0.3, 3.0)
    )
    
    # Token analysis
    token_metrics = calculate_precision_recall_f1(clean_predictions, clean_references)
    
    # Length analysis
    length_metrics = calculate_length_similarity(clean_predictions, clean_references)
    
    print("\nQuality Report Summary:")
    print(checker.format_results(all_metrics, precision=3))
    
    print("\nToken-Level Metrics:")
    for key, value in token_metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    print("\nLength Metrics:")
    print(f"  Average Prediction Length: {length_metrics['avg_pred_length']:.2f} tokens")
    print(f"  Average Reference Length: {length_metrics['avg_ref_length']:.2f} tokens")
    print(f"  Length Ratio: {length_metrics['length_ratio']:.4f}")
    
    if quality_issues:
        print("\nQuality Issues Found:")
        for issue_type, issues_list in quality_issues.items():
            for issue in issues_list:
                print(f"  - {issue}")
    else:
        print("\nNo quality issues detected ✓")
    
    # Example 5: Text preprocessing demonstration
    print("\n" + "=" * 60)
    print("Text Preprocessing Examples")
    print("=" * 60)
    
    raw_texts = [
        "The QUICK Brown FOX!!!",
        "Machine   Learning   is    POWERFUL",
        "Natural Language Processing (NLP) works great!!!"
    ]
    
    print("\nOriginal vs Preprocessed:")
    for text in raw_texts:
        cleaned = preprocess_text(text, lowercase=True, remove_punctuation=True)
        tokens = tokenize_text(cleaned)
        print(f"\n  Original: {text}")
        print(f"  Cleaned:  {cleaned}")
        print(f"  Tokens:   {tokens}")


if __name__ == "__main__":
    main()
