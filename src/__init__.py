"""Data quality checks for ML/NLP models."""

from .data_quality_checker import DataQualityChecker, QualityMetricsAggregator
from .unified_quality_checker import UnifiedDataQualityChecker
from .binary_classification import BinaryClassifierChecker, BinaryClassificationMetrics
from .regression import RegressionChecker, RegressionMetrics
from .generative_ai_text_model import (
    BLEUCalculator,
    BLEUAggregator,
    ROUGECalculator,
    ROUGEAggregator,
)
from .utils import (
    preprocess_text,
    tokenize_text,
    get_ngrams,
    calculate_precision_recall_f1,
    calculate_length_similarity,
    detect_data_quality_issues
)

__version__ = "0.1.0"
__all__ = [
    # Unified checker
    "UnifiedDataQualityChecker",
    # Legacy text metrics
    "DataQualityChecker",
    "QualityMetricsAggregator",
    # Binary classification
    "BinaryClassifierChecker",
    "BinaryClassificationMetrics",
    # Regression
    "RegressionChecker",
    "RegressionMetrics",
    # Generative AI text
    "BLEUCalculator",
    "BLEUAggregator",
    "ROUGECalculator",
    "ROUGEAggregator",
    # Utilities
    "preprocess_text",
    "tokenize_text",
    "get_ngrams",
    "calculate_precision_recall_f1",
    "calculate_length_similarity",
    "detect_data_quality_issues",
]
