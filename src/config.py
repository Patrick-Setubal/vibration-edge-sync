"""
Global configuration settings for the Edge Sensor Processing pipeline.
Centralizes constants for logging, sensor physics, edge processing, and cloud sync.
Allows overrides via environment variables for easy Docker/Edge deployment.
"""

import os
from pathlib import Path
import math

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
DEFAULT_LOG_FILE: str = os.getenv("LOG_FILE", ".data/logs/edge.log")
DEFAULT_LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB per file
DEFAULT_LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 3))          # Keep up to 3 old logs

# ==========================================
# SENSOR & PHYSICS CONFIGURATION
# ==========================================
SAMPLE_RATE_HZ: int = int(os.getenv("SAMPLE_RATE_HZ", 1000))  # 1 kHz default sampling rate

# Base parameters for the damped harmonic oscillator
BASE_OMEGA_N: float = float(os.getenv("BASE_OMEGA_N", 2 * math.pi * 10))  # 10 Hz natural frequency in rad/s
BASE_ZETA: float = float(os.getenv("BASE_ZETA", 0.05))                    # Baseline damping ratio

# ==========================================
# EDGE PROCESSING CONFIGURATION
# ==========================================
# How many samples to keep in the rolling window (e.g., 1000 samples = 1 second)
WINDOW_SIZE: int = int(os.getenv("WINDOW_SIZE", SAMPLE_RATE_HZ))

# Rate Limiting / Pacing Configuration
# Controls stream consumption rate in samples per second (Hz).
# Example: 1000 = 1000 samples/sec (Simulated real-time)
# Value -1 = Unlimited throughput (Runs at maximum CPU speed for benchmarks)
TARGET_SAMPLES_PER_SEC: float = 1000.0

# ==========================================
# CLOUD SYNC & STORAGE CONFIGURATION
# ==========================================
# Local directory for the durable file-based queue
QUEUE_DIR: Path = Path(os.getenv("QUEUE_DIR", ".data/queue"))

# AWS / LocalStack S3 configuration
S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "industrial-sensor-data")
DEVICE_ID: str = os.getenv("DEVICE_ID", "edge-01")