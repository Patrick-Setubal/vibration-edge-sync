"""
Sync Worker module.
Handles background cloud synchronization by reading batches from the durable queue
and uploading them to AWS S3 (or LocalStack). Implements exponential backoff for network failures.
"""

import threading
import time
import random
from typing import Optional

from src.utils.logger import get_logger
from src.sync.durable_queue import DurableQueue
from src.sync.s3_uploader import S3Uploader

logger = get_logger(__name__)


class SyncWorker:
    """
    Background worker thread responsible for durable cloud synchronization.
    Features:
    - Non-blocking operation relative to the edge processing engine.
    - Two-phase commit logic (peek -> upload -> pop).
    - Exponential backoff with jitter for transient connection loss.
    - Fast interruption response during system shutdown.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        queue: DurableQueue,
        uploader: S3Uploader,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0
    ):
        """
        Args:
            stop_event: Threading event to signal graceful shutdown.
            queue: Instance of DurableQueue for disk persistence.
            uploader: Instance of S3Uploader configured for S3/LocalStack.
            initial_backoff: Starting retry delay in seconds.
            max_backoff: Maximum capped retry delay in seconds.
        """
        self.stop_event = stop_event
        self.queue = queue
        self.uploader = uploader
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def run(self) -> None:
        """
        Main execution loop for cloud sync.
        Continuously checks the durable queue and uploads pending batches.
        """
        logger.info("Starting Cloud Sync Worker thread...")
        current_backoff = self.initial_backoff

        while not self.stop_event.is_set():
            try:
                # Peek at the oldest batch without removing it from disk
                batch_id, batch_data = self.queue.peek()

                if batch_id and batch_data:
                    # Attempt upload to S3 / LocalStack
                    success = self.uploader.upload(batch_id, batch_data)

                    if success:
                        # On success, permanently remove the file from local disk
                        self.queue.pop(batch_id)
                        current_backoff = self.initial_backoff  # Reset backoff on success
                    else:
                        logger.warning(
                            f"Upload failed for {batch_id}."
                            f"Retrying in {current_backoff:.1f}s (Queue size: {len(self.queue)})"
                        )
                        
                        # Wait out the backoff period in small increments 
                        # so we can stop quickly if SIGINT/shutdown is triggered
                        self._interruptible_sleep(current_backoff)
                        
                        # Calculate exponential backoff with full jitter to avoid thundering herd
                        current_backoff = self._calculate_next_backoff(current_backoff)
                else:
                    # Queue is empty, sleep briefly before polling again
                    self._interruptible_sleep(0.5)

            except Exception as e:
                logger.error(f"Unexpected error in Cloud Sync Worker: {e}", exc_info=True)
                self._interruptible_sleep(self.initial_backoff)

        logger.info("Cloud Sync Worker thread stopped.")

    def _calculate_next_backoff(self, previous_backoff: float) -> float:
        """
        Computes exponential backoff with full random jitter.
        Formula: min(max_backoff, previous_backoff * 2) + jitter
        """
        next_base = min(self.max_backoff, previous_backoff * 2.0)
        # Add random jitter up to 20% of the backoff value
        jitter = random.uniform(0, 0.2 * next_base)
        return next_base + jitter

    def _interruptible_sleep(self, duration: float) -> None:
        """
        Sleeps in 100ms chunks to allow immediate response to stop_event.
        Prevents thread blocking during process shutdown.
        """
        chunks = int(duration * 10)
        for _ in range(chunks):
            if self.stop_event.is_set():
                break
            time.sleep(0.1)