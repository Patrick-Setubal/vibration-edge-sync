"""
Feature Extraction module.
Computes statistical and spectral characteristics over a bounded rolling window.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def extract_features(window_data: List[Tuple[float, float, bool]]) -> Dict[str, float]:
    """
    Computes summary features from a window of sensor readings.
    
    Args:
        window_data: List of tuples (timestamp, acceleration, ground_truth_anomaly).
        
    Returns:
        Dict containing RMS, mean, std_dev, and dominant_frequency_hz.
    """
    if not window_data:
        return {
            "rms": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "dominant_freq_hz": 0.0,
        }

    # Extract acceleration time series into a NumPy vector for fast C-level vectorized math
    accelerations = np.array([sample[1] for sample in window_data], dtype=np.float64)
    n_samples = len(accelerations)

    # Time-domain statistical features
    mean_val = float(np.mean(accelerations))
    std_val = float(np.std(accelerations))
    rms_val = float(np.sqrt(np.mean(np.square(accelerations))))

    # Frequency-domain feature (FFT Peak)
    # Estimate sampling rate dynamically based on timestamps in the window
    if n_samples > 1:
        time_span = window_data[-1][0] - window_data[0][0]
        sample_rate = n_samples / time_span if time_span > 0 else 1000.0
    else:
        sample_rate = 1000.0

    # Real FFT for real-valued signal
    fft_vals = np.abs(np.fft.rfft(accelerations))
    fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    # Ignore DC component (index 0) to find the primary vibration frequency
    if len(fft_vals) > 1:
        dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq_hz = float(fft_freqs[dominant_freq_idx])
    else:
        dominant_freq_hz = 0.0

    return {
        "rms": rms_val,
        "mean": mean_val,
        "std": std_val,
        "dominant_freq_hz": dominant_freq_hz,
    }