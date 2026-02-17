"""
Utility functions for data quality checks.
"""

from typing import List, Dict, Tuple
import re
from collections import Counter


def preprocess_text(text: str, lowercase: bool = True, remove_punctuation: bool = False) -> str:
    """
    Preprocess text for quality checks.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase (default: True)
        remove_punctuation: Remove punctuation (default: False)
    
    Returns:
        Preprocessed text
    """
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Simple whitespace tokenization.
    
    Args:
        text: Input text
    
    Returns:
        List of tokens
    """
    return text.split()


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from token list.
    
    Args:
        tokens: List of tokens
        n: Size of n-grams
    
    Returns:
        List of n-grams as tuples
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def calculate_precision_recall_f1(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 for token overlap.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    pred_tokens = set()
    ref_tokens = set()
    
    for pred in predictions:
        pred_tokens.update(tokenize_text(preprocess_text(pred)))
    
    for ref in references:
        ref_tokens.update(tokenize_text(preprocess_text(ref)))
    
    overlap = pred_tokens & ref_tokens
    
    if len(pred_tokens) == 0:
        precision = 0.0
    else:
        precision = len(overlap) / len(pred_tokens)
    
    if len(ref_tokens) == 0:
        recall = 0.0
    else:
        recall = len(overlap) / len(ref_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_length_similarity(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate text length similarity.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with length statistics
    """
    pred_lengths = [len(preprocess_text(p).split()) for p in predictions]
    ref_lengths = [len(preprocess_text(r).split()) for r in references]
    
    avg_pred_length = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
    avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
    
    if avg_ref_length == 0:
        length_ratio = 0.0
    else:
        length_ratio = avg_pred_length / avg_ref_length
    
    return {
        'avg_pred_length': avg_pred_length,
        'avg_ref_length': avg_ref_length,
        'length_ratio': length_ratio
    }


def detect_data_quality_issues(
    predictions: List[str],
    references: List[str],
    min_rouge_threshold: float = 0.3,
    max_length_ratio_threshold: Tuple[float, float] = (0.5, 2.0)
) -> Dict[str, List[str]]:
    """
    Detect quality issues in predictions.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        min_rouge_threshold: Minimum acceptable ROUGE score
        max_length_ratio_threshold: Acceptable range for length ratio
    
    Returns:
        Dictionary with lists of detected issues
    """
    issues = {
        'empty_predictions': [],
        'very_short_predictions': [],
        'very_long_predictions': [],
        'high_length_deviation': [],
        'duplicate_predictions': []
    }
    
    # Check for empty predictions
    for i, pred in enumerate(predictions):
        if not pred.strip():
            issues['empty_predictions'].append(f"Prediction {i}: empty")
    
    # Check for very short predictions
    for i, pred in enumerate(predictions):
        tokens = tokenize_text(preprocess_text(pred))
        if len(tokens) < 5:
            issues['very_short_predictions'].append(f"Prediction {i}: only {len(tokens)} tokens")
    
    # Check for length deviations
    pred_lengths = [len(tokenize_text(preprocess_text(p))) for p in predictions]
    ref_lengths = [len(tokenize_text(preprocess_text(r))) for r in references]
    
    avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 1
    
    for i, pred_len in enumerate(pred_lengths):
        ratio = pred_len / avg_ref_length if avg_ref_length > 0 else 1
        if ratio < max_length_ratio_threshold[0] or ratio > max_length_ratio_threshold[1]:
            issues['high_length_deviation'].append(
                f"Prediction {i}: length ratio {ratio:.2f} outside acceptable range"
            )
    
    # Check for duplicates
    seen = {}
    for i, pred in enumerate(predictions):
        normalized = preprocess_text(pred)
        if normalized in seen:
            issues['duplicate_predictions'].append(
                f"Prediction {i}: duplicate of prediction {seen[normalized]}"
            )
        else:
            seen[normalized] = i
    
    # Filter out empty issue lists
    return {k: v for k, v in issues.items() if v}
