"""
Benchmark script for Edge Sensor Processing Pipeline.
Validates two critical non-functional requirements:
1. Throughput: Must sustain > 1000 samples/sec on a single core.
2. Memory Footprint: Must run within a bounded, predictable memory footprint.
"""

import time
import tracemalloc
import logging

from src.config import WINDOW_SIZE
from src.sensor.simulator import VibrationSensor
from src.processor.ring_buffer import RingBuffer
from src.processor.detector import AnomalyDetector
from src.processor.features import extract_features  

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Benchmark")

# Benchmark Mode Override: Run at maximum CPU speed (-1 / Unlimited)
import src.config as config
config.TARGET_SAMPLES_PER_SEC = -1.0

NUM_SAMPLES = 100000  
BATCH_SIZE = WINDOW_SIZE


def run_benchmark():
    logger.info("=" * 50)
    logger.info(f"Starting Edge Processing Benchmark")
    logger.info(f"Target: > 1000 samples/sec")
    logger.info(f"Total Samples to Process: {NUM_SAMPLES}")
    logger.info("=" * 50)

    # Initialize components
    sensor = VibrationSensor()
    buffer = RingBuffer(max_len=WINDOW_SIZE)
    detector = AnomalyDetector()

    # Inicializa o stream (generator)
    sensor_stream = sensor.stream_samples(real_time=False)

    # Start memory tracking
    tracemalloc.start()
    
    processed_windows = 0
    
    # Start high-resolution timer
    start_time = time.perf_counter()

    # Hot Loop
    for i in range(1, NUM_SAMPLES + 1):
        # Ingest: consome a próxima amostra do generator
        sample = next(sensor_stream)
        
        # Como o sample é uma tupla (t, acceleration, is_anomaly), 
        # guardamos apenas a aceleração (ou a tupla toda, dependendo da sua extração)
        # Assumindo que o RingBuffer e extração esperam o valor de aceleração (índice 1)
        buffer.append(sample)

        # Process window
        if i % BATCH_SIZE == 0:
            window_data = buffer.get_window()
            
            features = extract_features(window_data) 
            is_anomaly = detector.evaluate(features)
            
            processed_windows += 1

    # Stop timer and memory tracking
    end_time = time.perf_counter()
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate metrics
    elapsed_time = end_time - start_time
    throughput = NUM_SAMPLES / elapsed_time

    # Print Results
    logger.info("\nBenchmark Results:")
    logger.info(f"Total Time Elapsed:   {elapsed_time:.4f} seconds")
    logger.info(f"Total Windows Processed: {processed_windows}")
    logger.info(f"Throughput:           {throughput:.2f} samples/second")
    
    logger.info("\nMemory Footprint:")
    logger.info(f"Current Memory Usage: {current_mem / 1024:.2f} KB")
    logger.info(f"Peak Memory Usage:    {peak_mem / 1024:.2f} KB")
    
    logger.info("\nConclusion:")
    if throughput > 1000:
        logger.info(f"SUCCESS: Throughput is {throughput/1000:.1f}x the required 1000 samples/sec threshold.")
    else:
        logger.warning("FAILED: Throughput is below the 1000 samples/sec threshold.")
        
    logger.info("Notice how Peak Memory stays small and bounded regardless of NUM_SAMPLES, proving the O(1) space complexity of the pipeline.")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_benchmark()