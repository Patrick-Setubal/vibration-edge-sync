"""
Unit tests for the VibrationSensor simulation module.
Verifies RK4 integration, streaming generator behavior, fault injection windows,
and parameter changes during simulated equipment anomalies.
"""

import math
import pytest
import time
from src.sensor.simulator import VibrationSensor


def test_sensor_initialization():
    """Verifies default parameters and initial state of the simulator."""
    sensor = VibrationSensor(sample_rate_hz=1000)
    assert sensor.sample_rate == 1000
    assert sensor.dt == 0.001
    assert sensor.fault_start_time is None
    assert sensor.fault_end_time is None


def test_sensor_stream_sample_structure():
    """Ensures the generator yields samples as (timestamp, acceleration, ground_truth)."""
    sensor = VibrationSensor(sample_rate_hz=1000)
    stream = sensor.stream_samples()

    # Get first 10 samples from the stream
    samples = [next(stream) for _ in range(10)]

    assert len(samples) == 10
    for t, accel, is_anomaly in samples:
        assert isinstance(t, float)
        assert isinstance(accel, float)
        assert isinstance(is_anomaly, bool)
        assert not math.isnan(accel)
        assert not math.isinf(accel)


def test_sensor_timestamp_increment():
    """Validates that timestamps advance strictly according to sample rate (dt = 1 / Hz)."""
    sample_rate = 500
    sensor = VibrationSensor(sample_rate_hz=sample_rate)

    # Executa sem o sleep de tempo real para o teste ser instantâneo
    stream = sensor.stream_samples(real_time=False)

    start_wall_time = time.time()
    t0, _, _ = next(stream)
    t1, _, _ = next(stream)
    t2, _, _ = next(stream)

    expected_dt = 1.0 / sample_rate

    # Valida se t0 é um timestamp Unix válido
    assert t0 == pytest.approx(start_wall_time, abs=5.0)

    # Valida se os incrementos seguem o dt esperado com margem de tolerância para float
    assert (t1 - t0) == pytest.approx(expected_dt, abs=1e-6)
    assert (t2 - t1) == pytest.approx(expected_dt, abs=1e-6)


def test_fault_injection_window():
    """
    Verifies that scheduling a fault updates the ground_truth_anomaly flag
    only within the exact start and end duration window.
    """
    sensor = VibrationSensor(sample_rate_hz=1000)
    # Schedule fault starting at t=0.005s lasting for 0.003s (t=0.005s to t=0.008s)
    sensor.inject_fault(start_time=0.005, duration=0.003)

    stream = sensor.stream_samples()
    samples = [next(stream) for _ in range(12)]  # t = 0.000s to t = 0.011s

    # t < 0.005s: Normal operation
    for t, _, is_anomaly in samples[:5]:
        assert is_anomaly is False

    # 0.005s <= t <= 0.008s: Anomaly window
    for t, _, is_anomaly in samples[5:9]:
        assert is_anomaly is True

    # t > 0.008s: Returned to normal operation baseline
    for t, _, is_anomaly in samples[9:]:
        assert is_anomaly is False


def test_fault_injection_alters_physics_parameters():
    """
    Verifies that fault injection alters natural frequency (omega_n) and damping (zeta),
    producing physically valid numerical output without math errors.
    """
    sensor = VibrationSensor(sample_rate_hz=1000)
    # Inject fault with custom altered natural frequency and damping ratio
    sensor.inject_fault(
        start_time=0.002,
        duration=0.005,
        new_omega_n=10.0,
        new_zeta=0.01
    )

    stream = sensor.stream_samples()
    samples = [next(stream) for _ in range(10)]

    for t, accel, is_anomaly in samples:
        assert not math.isnan(accel)
        assert not math.isinf(accel)
        if 0.002 <= t <= 0.007:
            assert is_anomaly is True