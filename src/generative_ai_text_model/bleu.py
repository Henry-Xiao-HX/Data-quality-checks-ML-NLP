"""
BLEU (Bilingual Evaluation Understudy) metric module.
Computes BLEU score for machine translation and text generation evaluation.
Uses Hugging Face evaluate library for metric computation.
"""

from typing import List, Dict, Union, Optional
import warnings
import numpy as np

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    warnings.warn("Hugging Face evaluate library not installed. Install with: pip install evaluate")


class BLEUCalculator:
    """
    BLEU (Bilingual Evaluation Understudy) metrics calculator.
    
    BLEU measures n-gram precision between predictions and references,
    with a brevity penalty for shorter texts.
    
    Scores range from 0 to 1:
    - 0: No n-gram matches
    - 1: Perfect match with references
    """
    
    def __init__(self):
        """Initialize the BLEU calculator."""
        if not EVALUATE_AVAILABLE:
            raise ImportError(
                "This module requires the 'evaluate' library. "
                "Install it with: pip install evaluate"
            )
        self.bleu = evaluate.load('bleu')
    
    def compute_bleu(
        self,
        predictions: Union[List[str], str],
        references: Union[List[List[str]], List[str]],
        max_order: int = 4,
        smooth: bool = False
    ) -> Dict[str, float]:
        """
        Compute BLEU score.
        
        Args:
            predictions: Model predictions (list of strings or single string)
            references: Reference texts
                       For single reference per prediction, use List[str]
                       For multiple references, use List[List[str]]
            max_order: Maximum n-gram order to compute (default: 4)
                       Computes 1-grams, 2-grams, ..., n-grams up to max_order
            smooth: Whether to apply smoothing for zero counts (default: False)
        
        Returns:
            Dictionary with 'bleu' score and 'precisions' list
            Example: {'bleu': 0.45, 'precisions': [0.8, 0.6, 0.4, 0.2]}
        
        Example:
            >>> calculator = BLEUCalculator()
            >>> scores = calculator.compute_bleu(
            ...     predictions=["the cat sat on the mat"],
            ...     references=["a cat was sitting on the mat"]
            ... )
        """
        # Normalize inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        
        # Normalize references structure
        if isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                # Convert single references to list of lists
                references = [[ref] for ref in references]
        
        results = self.bleu.compute(
            predictions=predictions,
            references=references,
            max_order=max_order,
            smooth=smooth
        )
        
        return results
    
    def compute_bleu_detailed(
        self,
        predictions: Union[List[str], str],
        references: Union[List[List[str]], List[str]],
        max_order: int = 4,
        smooth: bool = False
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute BLEU score with detailed n-gram breakdown.
        
        Args:
            predictions: Model predictions
            references: Reference texts
            max_order: Maximum n-gram order
            smooth: Whether to apply smoothing
        
        Returns:
            Dictionary with overall BLEU score and individual n-gram precisions
        """
        results = self.compute_bleu(predictions, references, max_order, smooth)
        
        # Add more detailed breakdown
        detailed_results = {
            'bleu': results.get('bleu', 0.0),
            'precisions': results.get('precisions', []),
            'n_gram_details': {}
        }
        
        precisions = results.get('precisions', [])
        for i, precision in enumerate(precisions, 1):
            detailed_results['n_gram_details'][f'{i}-gram'] = precision
        
        return detailed_results


class BLEUAggregator:
    """Aggregate BLEU scores across multiple samples."""
    
    @staticmethod
    def aggregate_bleu_scores(
        scores_list: List[Dict[str, float]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate BLEU scores across samples.
        
        Args:
            scores_list: List of BLEU score dictionaries
            aggregation_type: Aggregation method ('mean', 'median', 'min', 'max')
        
        Returns:
            Aggregated scores
        """
        if not scores_list:
            return {}
        
        aggregated = {}
        
        # Get all unique keys
        all_keys = set()
        for scores in scores_list:
            all_keys.update(scores.keys())
        
        for key in all_keys:
            # Handle both float and list values
            values = []
            for s in scores_list:
                if key in s:
                    val = s[key]
                    if isinstance(val, (int, float)):
                        values.append(float(val))
            
            if values:
                if aggregation_type == 'mean':
                    aggregated[key] = np.mean(values)
                elif aggregation_type == 'median':
                    aggregated[key] = np.median(values)
                elif aggregation_type == 'min':
                    aggregated[key] = np.min(values)
                elif aggregation_type == 'max':
                    aggregated[key] = np.max(values)
        
        return aggregated
