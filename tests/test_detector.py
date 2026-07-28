"""
Unit tests for the AnomalyDetector module.
Verifies threshold evaluation, metrics updating (Precision/Recall), and edge cases.
"""

import pytest
from src.processor.detector import AnomalyDetector


@pytest.fixture
def detector():
    """Fixture to provide a clean AnomalyDetector instance before each test."""
    return AnomalyDetector(rms_threshold=2.5, std_threshold=2.0)


def test_detector_initialization(detector):
    """Verifies default values and initial counters state."""
    assert detector.rms_threshold == 2.5
    assert detector.std_threshold == 2.0
    assert detector.true_positives == 0
    assert detector.false_positives == 0
    assert detector.true_negatives == 0
    assert detector.false_negatives == 0


def test_evaluate_normal_signal(detector):
    """Ensures normal operating features do not trigger an anomaly flag."""
    normal_features = {
        "rms": 1.2,
        "std": 0.8,
        "mean": 0.05,
        "dominant_freq_hz": 60.0
    }
    is_anomaly = detector.evaluate(normal_features)
    assert is_anomaly is False


def test_evaluate_rms_threshold_exceeded(detector):
    """Triggers anomaly when RMS exceeds threshold."""
    high_rms_features = {
        "rms": 3.0,  # Exceeds threshold of 2.5
        "std": 1.0,
        "mean": 0.0,
        "dominant_freq_hz": 60.0
    }
    assert detector.evaluate(high_rms_features) is True


def test_evaluate_std_threshold_exceeded(detector):
    """Triggers anomaly when STD exceeds threshold."""
    high_std_features = {
        "rms": 1.5,
        "std": 2.8,  # Exceeds threshold of 2.0
        "mean": 0.0,
        "dominant_freq_hz": 60.0
    }
    assert detector.evaluate(high_std_features) is True


def test_evaluate_both_thresholds_exceeded(detector):
    """Triggers anomaly when both RMS and STD exceed thresholds."""
    fault_features = {
        "rms": 4.1,
        "std": 3.5,
        "mean": 0.2,
        "dominant_freq_hz": 40.0
    }
    assert detector.evaluate(fault_features) is True


def test_evaluate_missing_or_empty_features(detector):
    """Ensures safe behavior with missing feature keys or empty dicts."""
    empty_features = {}
    # Missing keys default to 0.0, should not trigger anomaly
    assert detector.evaluate(empty_features) is False

    partial_features = {"rms": 1.0}  # 'std' is missing
    assert detector.evaluate(partial_features) is False


def test_confusion_matrix_and_metrics_calculation(detector):
    """Validates confusion matrix updates and calculation of Precision, Recall, and F1."""
    # 1. True Positive (Predicted True, Actual True)
    detector.update_metrics(predicted_anomaly=True, ground_truth_anomaly=True)

    # 2. False Positive (Predicted True, Actual False)
    detector.update_metrics(predicted_anomaly=True, ground_truth_anomaly=False)

    # 3. False Negative (Predicted False, Actual True)
    detector.update_metrics(predicted_anomaly=False, ground_truth_anomaly=True)

    # 4. True Negative (Predicted False, Actual False)
    detector.update_metrics(predicted_anomaly=False, ground_truth_anomaly=False)
    detector.update_metrics(predicted_anomaly=False, ground_truth_anomaly=False)

    metrics = detector.get_metrics()

    assert detector.true_positives == 1
    assert detector.false_positives == 1
    assert detector.false_negatives == 1
    assert detector.true_negatives == 2

    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    assert metrics["precision"] == 0.5

    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    assert metrics["recall"] == 0.5

    # F1-Score = 2 * (P * R) / (P + R) = 2 * (0.25) / (1.0) = 0.5
    assert metrics["f1_score"] == 0.5

    # Accuracy = (TP + TN) / Total = (1 + 2) / 5 = 0.6
    assert metrics["accuracy"] == 0.6
    assert metrics["total_windows"] == 5


def test_metrics_division_by_zero_safety(detector):
    """Ensures metrics calculation returns 0.0 without raising ZeroDivisionError on empty state."""
    metrics = detector.get_metrics()

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["total_windows"] == 0