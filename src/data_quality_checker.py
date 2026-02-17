"""
Data Quality Checker module with ROUGE and BLEU metrics.
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


class DataQualityChecker:
    """
    Comprehensive data quality checker supporting ROUGE and BLEU metrics.
    
    Metrics:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - ROUGE-S: Skip-bigram overlap
    - BLEU: Bilingual Evaluation Understudy
    """
    
    def __init__(self):
        """Initialize the data quality checker with metrics."""
        if not EVALUATE_AVAILABLE:
            raise ImportError(
                "This module requires the 'evaluate' library. "
                "Install it with: pip install evaluate"
            )
        
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.metrics_cache = {}
    
    def compute_rouge(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        rouge_types: Optional[List[str]] = None,
        use_stemmer: bool = False,
        use_aggregator: bool = True
    ) -> Dict[str, float]:
        """
        Compute ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S).
        
        Args:
            predictions: Model predictions (list of strings or single string)
            references: Reference/ground truth texts (list of strings or single string)
            rouge_types: Types of ROUGE to compute. Defaults to all.
            use_stemmer: Whether to use stemmer (default: False)
            use_aggregator: Whether to aggregate scores (default: True)
        
        Returns:
            Dictionary with ROUGE scores
        
        Example:
            >>> checker = DataQualityChecker()
            >>> scores = checker.compute_rouge(
            ...     predictions="The cat sat on the mat",
            ...     references="A cat was sitting on the mat"
            ... )
        """
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeS']
        
        # Normalize inputs to lists
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        
        # Compute ROUGE
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
            use_aggregator=use_aggregator
        )
        
        return results
    
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
            references: Reference texts (list of lists or list of strings)
                       For single reference per prediction, use List[str]
                       For multiple references, use List[List[str]]
            max_order: Maximum n-gram order (default: 4)
            smooth: Whether to apply smoothing (default: False)
        
        Returns:
            Dictionary with BLEU score and precisions
        
        Example:
            >>> checker = DataQualityChecker()
            >>> scores = checker.compute_bleu(
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
    
    def compute_all_metrics(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        compute_rouge: bool = True,
        compute_bleu: bool = True,
        rouge_types: Optional[List[str]] = None,
        stemmer: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all supported metrics at once.
        
        Args:
            predictions: Model predictions
            references: Reference texts
            compute_rouge: Whether to compute ROUGE metrics (default: True)
            compute_bleu: Whether to compute BLEU score (default: True)
            rouge_types: ROUGE types to compute
            stemmer: Use stemmer for ROUGE (default: False)
        
        Returns:
            Dictionary with all computed metrics
        
        Example:
            >>> checker = DataQualityChecker()
            >>> all_scores = checker.compute_all_metrics(
            ...     predictions=["the cat is sitting"],
            ...     references=["a cat is sitting"]
            ... )
        """
        results = {}
        
        if compute_rouge:
            results['rouge'] = self.compute_rouge(
                predictions, references, rouge_types, stemmer
            )
        
        if compute_bleu:
            results['bleu'] = self.compute_bleu(predictions, references)
        
        return results
    
    def batch_compute_metrics(
        self,
        predictions_list: List[List[str]],
        references_list: List[List[str]],
        metric_types: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for multiple batches.
        
        Args:
            predictions_list: List of prediction batches
            references_list: List of reference batches
            metric_types: Types of metrics to compute
        
        Returns:
            List of dictionaries with metrics for each batch
        """
        if metric_types is None:
            metric_types = ['rouge', 'bleu']
        
        results = []
        for preds, refs in zip(predictions_list, references_list):
            batch_result = {}
            
            if 'rouge' in metric_types:
                batch_result['rouge'] = self.compute_rouge(preds, refs)
            if 'bleu' in metric_types:
                batch_result['bleu'] = self.compute_bleu(preds, refs)
            
            results.append(batch_result)
        
        return results
    
    def get_rouge_score_explanation(self, rouge_type: str) -> str:
        """Get explanation of ROUGE metric types."""
        explanations = {
            'rouge1': 'ROUGE-1: Overlap of unigrams (single words)',
            'rouge2': 'ROUGE-2: Overlap of bigrams (consecutive word pairs)',
            'rougeL': 'ROUGE-L: Longest common subsequence',
            'rougeS': 'ROUGE-S: Skip-bigram (non-consecutive word pairs)'
        }
        return explanations.get(rouge_type, f'Unknown ROUGE type: {rouge_type}')
    
    def format_results(
        self,
        results: Dict,
        precision: int = 4
    ) -> str:
        """
        Format results for display.
        
        Args:
            results: Dictionary of metric results
            precision: Decimal precision for display
        
        Returns:
            Formatted string representation
        """
        lines = []
        
        for metric_group, scores in results.items():
            lines.append(f"\n{metric_group.upper()}:")
            lines.append("-" * 40)
            
            if isinstance(scores, dict):
                for key, value in scores.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.{precision}f}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class QualityMetricsAggregator:
    """Aggregate quality metrics across multiple samples."""
    
    @staticmethod
    def aggregate_rouge_scores(
        scores_list: List[Dict[str, float]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate ROUGE scores across samples.
        
        Args:
            scores_list: List of ROUGE score dictionaries
            aggregation_type: 'mean', 'median', 'min', 'max'
        
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
            values = [s[key] for s in scores_list if key in s and isinstance(s[key], (int, float))]
            
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
    
    @staticmethod
    def aggregate_bleu_scores(
        scores_list: List[Dict[str, float]],
        aggregation_type: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregate BLEU scores across samples.
        
        Args:
            scores_list: List of BLEU score dictionaries
            aggregation_type: 'mean', 'median', 'min', 'max'
        
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
