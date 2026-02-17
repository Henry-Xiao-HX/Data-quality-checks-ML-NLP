"""
Binary Classification Metrics Calculator.

Computes various quality metrics for binary classification models using scikit-learn.
"""

from typing import Union, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


class BinaryClassificationMetrics:
    """
    Comprehensive binary classification metrics calculator.
    
    Computes metrics for binary classification models:
    - Area Under ROC (AUROC)
    - True Positive Rate (TPR)
    - Precision
    - F1-Measure
    - Logarithmic Loss
    - False Positive Rate (FPR)
    - Area Under PR (AUPR)
    - Recall
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics_cache = {}
    
    def compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_pred_proba: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Area Under the ROC Curve.
        
        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.
            sample_weight: Optional sample weights.
        
        Returns:
            float: AUROC score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        return float(roc_auc_score(y_true, y_pred_proba, sample_weight=sample_weight))
    
    def compute_true_positive_rate(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        pos_label: int = 1,
        average: Optional[str] = None,
    ) -> float:
        """
        Compute True Positive Rate (also known as Recall or Sensitivity).
        
        TPR = TP / (TP + FN)
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            pos_label: The positive label class.
            average: Type of averaging (None for binary classification).
        
        Returns:
            float: TPR score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(recall_score(y_true, y_pred, pos_label=pos_label, average=average))
    
    def compute_precision(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        pos_label: int = 1,
        average: Optional[str] = None,
    ) -> float:
        """
        Compute Precision.
        
        Precision = TP / (TP + FP)
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            pos_label: The positive label class.
            average: Type of averaging (None for binary classification).
        
        Returns:
            float: Precision score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(precision_score(y_true, y_pred, pos_label=pos_label, average=average))
    
    def compute_f1_measure(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        pos_label: int = 1,
        average: Optional[str] = None,
    ) -> float:
        """
        Compute F1-Measure (Harmonic Mean of Precision and Recall).
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            pos_label: The positive label class.
            average: Type of averaging (None for binary classification).
        
        Returns:
            float: F1-score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(f1_score(y_true, y_pred, pos_label=pos_label, average=average))
    
    def compute_log_loss(
        self,
        y_true: np.ndarray,
        y_pred_proba: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Logarithmic Loss (Cross-Entropy Loss).
        
        Log Loss = -1/n * Σ[y_true * log(y_pred_proba) + (1-y_true) * log(1-y_pred_proba)]
        
        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.
            sample_weight: Optional sample weights.
        
        Returns:
            float: Log loss (lower is better).
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        return float(log_loss(y_true, y_pred_proba, sample_weight=sample_weight))
    
    def compute_false_positive_rate(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
    ) -> float:
        """
        Compute False Positive Rate.
        
        FPR = FP / (FP + TN)
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
        
        Returns:
            float: FPR score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate confusion matrix values
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Handle edge case where there are no negatives
        if (fp + tn) == 0:
            return 0.0
        
        fpr = fp / (fp + tn)
        return float(fpr)
    
    def compute_auc_pr(
        self,
        y_true: np.ndarray,
        y_pred_proba: Union[np.ndarray, list],
        pos_label: int = 1,
    ) -> float:
        """
        Compute Area Under the Precision-Recall Curve.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            pos_label: The positive label class.
        
        Returns:
            float: AUPR score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        return float(average_precision_score(y_true, y_pred_proba, pos_label=pos_label))
    
    def compute_recall(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        pos_label: int = 1,
        average: Optional[str] = None,
    ) -> float:
        """
        Compute Recall (Sensitivity or True Positive Rate).
        
        Recall = TP / (TP + FN)
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            pos_label: The positive label class.
            average: Type of averaging (None for binary classification).
        
        Returns:
            float: Recall score between 0 and 1.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(recall_score(y_true, y_pred, pos_label=pos_label, average=average))
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: Union[np.ndarray, list],
        y_pred_proba: Union[np.ndarray, list],
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute all quality metrics at once.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            sample_weight: Optional sample weights.
        
        Returns:
            dict: Dictionary containing all computed metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        metrics = {
            'roc_auc': self.compute_roc_auc(y_true, y_pred_proba, sample_weight),
            'true_positive_rate': self.compute_true_positive_rate(y_true, y_pred),
            'precision': self.compute_precision(y_true, y_pred),
            'f1_measure': self.compute_f1_measure(y_true, y_pred),
            'log_loss': self.compute_log_loss(y_true, y_pred_proba, sample_weight),
            'false_positive_rate': self.compute_false_positive_rate(y_true, y_pred),
            'auc_pr': self.compute_auc_pr(y_true, y_pred_proba),
            'recall': self.compute_recall(y_true, y_pred),
        }
        
        return metrics
    
    def get_roc_curve_data(
        self,
        y_true: np.ndarray,
        y_pred_proba: Union[np.ndarray, list],
        pos_label: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data (FPR, TPR, thresholds) for plotting.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            pos_label: The positive label class.
        
        Returns:
            tuple: (fpr, tpr, thresholds) arrays.
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, pos_label=pos_label)
        return fpr, tpr, thresholds
    
    def get_pr_curve_data(
        self,
        y_true: np.ndarray,
        y_pred_proba: Union[np.ndarray, list],
        pos_label: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data for plotting.
        
        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            pos_label: The positive label class.
        
        Returns:
            tuple: (precision, recall, thresholds) arrays.
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_pred_proba, pos_label=pos_label
        )
        return precision, recall, thresholds
