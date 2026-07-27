"""
Edge Worker module.
Orchestrates the high-priority loop: reading sensor data, managing the rolling window,
extracting features, detecting anomalies, and dispatching payloads to the durable queue.
"""

import threading
from typing import Any

from src.config import WINDOW_SIZE
from src.utils.logger import get_logger
from src.processor.features import extract_features

from src.sensor.simulator import VibrationSensor
from src.processor.ring_buffer import RingBuffer
from src.sync.durable_queue import DurableQueue
from src.processor.detector import AnomalyDetector

logger = get_logger(__name__)


class EdgeWorker:
    """
    High-priority thread worker that consumes sensor data in real-time,
    computes rolling metrics, and safely stores results to disk.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        sensor: VibrationSensor,
        buffer: RingBuffer,
        queue: DurableQueue,
        detector: AnomalyDetector
    ):
        self.stop_event = stop_event
        self.sensor = sensor
        self.buffer = buffer
        self.queue = queue
        self.detector = detector

    def run(self) -> None:
        """
        Main execution loop for the edge processor.
        Consumes the generator, updates the sliding window, and batches processing.
        """
        logger.info("Starting Edge Worker thread...")
        samples_count = 0

        # Schedule a synthetic fault to test our detector (e.g., at 5s mark, lasting 2s)
        self.sensor.inject_fault(start_time=5.0, duration=2.0)

        try:
            # stream_samples() yields: (timestamp, acceleration, ground_truth_anomaly)
            for sample in self.sensor.stream_samples():
                if self.stop_event.is_set():
                    break
                
                self.buffer.append(sample)
                samples_count += 1
                
                # To maintain >1000Hz throughput efficiently, we process the window
                if samples_count % WINDOW_SIZE == 0:
                    self._process_window()
                    
        except Exception as e:
            logger.error(f"Edge Worker encountered a fatal error: {e}", exc_info=True)
        finally:
            logger.info("Edge Worker thread stopped.")

    def _process_window(self) -> None:
        """
        Extracts features from the current window, runs anomaly detection, 
        and queues the result for cloud sync.
        """
        window_data = self.buffer.get_window()
        
        # Extract statistical and frequency-domain features (RMS, FFT, etc.)
        features = extract_features(window_data)
        
        # Run anomaly detection logic
        is_anomaly = self.detector.evaluate(features)
        
        # Extract the ground truth from the last sample in the window for scoring
        # window_data format: list of tuples (timestamp, value, ground_truth)
        last_timestamp, _, ground_truth = window_data[-1]
        
        # Build the payload batch
        payload = {
            "timestamp": last_timestamp,
            "features": features,
            "anomaly_detected": is_anomaly,
            "ground_truth": ground_truth,
            # (Optional) In a real scenario, we might include a downsampled array here
        }
        
        # Push atomically to disk (JSON.gz) via DurableQueue
        self.queue.push(payload)