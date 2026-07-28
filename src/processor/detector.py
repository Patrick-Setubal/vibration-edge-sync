"""
Anomaly Detector module.
Detects industrial vibration anomalies based on feature thresholding (Z-Score/RMS)
and tracks classification statistics (Precision, Recall, F1-Score).
"""

from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Evaluates window features against operational thresholds to flag potential equipment faults.
    Tracks confusion matrix components (TP, FP, TN, FN) to compute accuracy metrics.
    """

    def __init__(self, rms_threshold: float = 2.0, std_threshold: float = 1.8):
        """
        Args:
            rms_threshold: Threshold above baseline for Root Mean Square vibration.
            std_threshold: Threshold for standard deviation change.
        """
        self.rms_threshold = rms_threshold
        self.std_threshold = std_threshold

        # Confusion Matrix Counters
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def evaluate(self, features: Dict[str, float]) -> bool:
        """
        Determines if the provided window features indicate an anomaly.
        
        Args:
            features: Dictionary containing 'rms', 'mean', 'std', and 'dominant_freq_hz'.
            
        Returns:
            bool: True if an anomaly is flagged, False otherwise.
        """
        rms = features.get("rms", 0.0)
        std = features.get("std", 0.0)

        # Flag anomaly if RMS or StdDev breaches the operational envelope
        is_anomaly = (rms > self.rms_threshold) or (std > self.std_threshold)
        
        return is_anomaly

    def update_metrics(self, predicted_anomaly: bool, ground_truth_anomaly: bool) -> None:
        """
        Updates the confusion matrix based on model prediction vs actual fault window.
        
        Args:
            predicted_anomaly: Flag set by evaluate().
            ground_truth_anomaly: True ground truth flag from the simulator.
        """
        if predicted_anomaly and ground_truth_anomaly:
            self.true_positives += 1
        elif predicted_anomaly and not ground_truth_anomaly:
            self.false_positives += 1
        elif not predicted_anomaly and ground_truth_anomaly:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculates Precision, Recall, and F1-Score.
        
        Returns:
            Dict containing precision, recall, f1_score, and accuracy.
        """
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        tn = self.true_negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "accuracy": round(accuracy, 4),
            "total_windows": total
        }

    def print_metrics(self) -> None:
        """Logs the final precision/recall report to stdout/log file."""
        metrics = self.get_metrics()
        logger.info(
            f"Anomaly Detector Performance -> "
            f"Precision: {metrics['precision']:.2%}, "
            f"Recall: {metrics['recall']:.2%}, "
            f"F1-Score: {metrics['f1_score']:.2%}, "
            f"Total Windows: {metrics['total_windows']} "
            f"(TP={self.true_positives}, FP={self.false_positives}, "
            f"FN={self.false_negatives}, TN={self.true_negatives})"
        )