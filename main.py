"""
Main entry point for the Edge Sensor Processing pipeline.
Responsible only for dependency injection, component initialization, 
and thread orchestration (start & graceful shutdown).
"""

import threading
import time
import sys

from src.config import WINDOW_SIZE
from src.utils.logger import get_logger

# Core components
from src.sensor.simulator import VibrationSensor
from src.processor.ring_buffer import RingBuffer
from src.processor.detector import AnomalyDetector
from src.sync.durable_queue import DurableQueue
from src.sync.s3_uploader import S3Uploader

# Thread Workers (Loops)
from src.processor.worker import EdgeWorker
from src.sync.worker import SyncWorker

logger = get_logger(__name__)


def main():
    logger.info("Initializing Edge Sensor Pipeline...")
    
    # 1. Initialize core components (Dependency Injection)
    sensor = VibrationSensor()
    buffer = RingBuffer(max_len=WINDOW_SIZE)
    queue = DurableQueue()
    uploader = S3Uploader()
    detector = AnomalyDetector()
    
    stop_event = threading.Event()
    
    # 2. Initialize worker classes
    edge_worker = EdgeWorker(stop_event, sensor, buffer, queue, detector)
    sync_worker = SyncWorker(stop_event, queue, uploader)
    
    # 3. Create threads pointing to the workers' primary methods
    edge_thread = threading.Thread(
        target=edge_worker.run,
        name="EdgeProcessorThread",
        daemon=True
    )
    
    sync_thread = threading.Thread(
        target=sync_worker.run,
        name="CloudSyncThread",
        daemon=True
    )
    
    # 4. Start execution
    edge_thread.start()
    sync_thread.start()
    
    try:
        # Keep the main thread alive waiting for an interrupt (Ctrl+C)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating graceful shutdown...")
    finally:
        # 5. Graceful shutdown sequence
        stop_event.set()
        
        logger.info("Waiting for threads to finish...")
        edge_thread.join(timeout=3.0)
        sync_thread.join(timeout=3.0)
        
        # Print detector stats as required by the challenge
        logger.info("=== Final Execution Stats ===")
        detector.print_metrics()
        logger.info("Shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()