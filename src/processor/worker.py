"""
Edge Worker module.
Orchestrates the high-priority loop: reading sensor data, managing the rolling window,
extracting features, detecting anomalies, and dispatching payloads to the durable queue.
Includes rate limiting (pacing) to match simulated real-time speed or run uncapped.
"""

import time
import threading
from typing import Any

from src.config import WINDOW_SIZE, TARGET_SAMPLES_PER_SEC
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
        detector: AnomalyDetector,
        target_rate: float = TARGET_SAMPLES_PER_SEC
    ):
        self.stop_event = stop_event
        self.sensor = sensor
        self.buffer = buffer
        self.queue = queue
        self.detector = detector
        self.target_rate = target_rate

    def run(self) -> None:
        """
        Main execution loop for the edge processor.
        Consumes the generator, updates the sliding window, and batches processing.
        Applies pace control to throttle throughput if target_rate > 0.
        """
        logger.info(f"Starting Edge Worker thread (Target rate: {self.target_rate} Hz)...")
        samples_count = 0

        # Schedule a synthetic fault to test our detector (e.g., at 5s mark, lasting 2s)
        self.sensor.inject_fault(start_time=5.0, duration=2.0)

        # High-precision timer for rate control
        start_time = time.perf_counter()

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

                # Rate Limiting / Pace Control
                if self.target_rate > 0:
                    # Expected total elapsed time in seconds for the samples processed so far
                    expected_time = samples_count / self.target_rate
                    elapsed_time = time.perf_counter() - start_time
                    
                    # If processing is running faster than target rate, sleep for the difference
                    if elapsed_time < expected_time:
                        time.sleep(expected_time - elapsed_time)
                    
        except Exception as e:
            logger.error(f"Edge Worker encountered a fatal error: {e}", exc_info=True)
        finally:
            logger.info("Edge Worker thread stopped.")

    def _process_window(self) -> None:
        """
        Extracts features from the current window, runs anomaly detection, 
        updates detection confusion matrix metrics, and queues the result for cloud sync.
        """
        window_data = self.buffer.get_window()
        
        # Extract statistical and frequency-domain features
        features = extract_features(window_data)
        
        # Run anomaly detection logic
        is_anomaly = self.detector.evaluate(features)
        
        # Extract ground truth from the last sample in the window
        last_timestamp, _, ground_truth = window_data[-1]
        
        # Update detector metrics so total_windows and precision/recall are tracked
        self.detector.update_metrics(predicted_anomaly=is_anomaly, ground_truth_anomaly=ground_truth)
        
        # Build payload batch
        payload = {
            "timestamp": last_timestamp,
            "features": features,
            "anomaly_detected": is_anomaly,
            "ground_truth": ground_truth,
        }
        
        # Push atomically to disk
        self.queue.push(payload)