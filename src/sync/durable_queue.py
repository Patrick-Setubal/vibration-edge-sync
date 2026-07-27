"""
Durable Queue module.
Implements a disk-backed FIFO queue resilient to network outages and process restarts.
Uses compressed JSON files (.json.gz) with atomic disk writes.
"""

import json
import gzip
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

from src.config import QUEUE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DurableQueue:
    """
    File-based FIFO Queue for Edge persistence.
    Stores processed payload batches on disk before uploading to the cloud.
    Guarantees no data loss across application restarts.
    """

    def __init__(self, queue_dir: Path = QUEUE_DIR):
        """
        Args:
            queue_dir: Directory where compressed batch files are persisted.
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def push(self, data: Dict[str, Any]) -> str:
        """
        Serializes and writes a batch atomically to disk using gzip compression.
        
        Args:
            data: Payload dictionary to persist.
            
        Returns:
            str: Unique batch ID (filename).
        """
        # High-resolution timestamp guarantees FIFO ordering when sorting filenames
        timestamp_ns = time.time_ns()
        batch_id = f"batch_{timestamp_ns}.json.gz"
        final_path = self.queue_dir / batch_id
        temp_path = self.queue_dir / f"{batch_id}.tmp"

        try:
            # Write to temporary file first
            json_bytes = json.dumps(data).encode("utf-8")
            with gzip.open(temp_path, "wb") as f:
                f.write(json_bytes)

            # Atomic rename to make it visible to the sync worker safely
            os.replace(temp_path, final_path)
            return batch_id

        except Exception as e:
            logger.error(f"Failed to write batch to durable queue: {e}", exc_info=True)
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def peek(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Retrieves the oldest pending batch from the queue without removing it.
        
        Returns:
            Tuple[Optional[str], Optional[Dict]]: (batch_id, payload) or (None, None) if queue is empty.
        """
        files = self._get_sorted_queue_files()
        if not files:
            return None, None

        oldest_file = files[0]
        batch_id = oldest_file.name

        try:
            with gzip.open(oldest_file, "rb") as f:
                content = f.read().decode("utf-8")
                payload = json.loads(content)
            return batch_id, payload
        except Exception as e:
            logger.error(f"Corrupted batch file detected: {batch_id}. Error: {e}")
            # Quarantine or move aside broken files to prevent infinite worker loops
            self._quarantine_file(oldest_file)
            return None, None

    def pop(self, batch_id: str) -> bool:
        """
        Permanently deletes a batch file from disk after successful sync.
        
        Args:
            batch_id: Name of the batch file to remove.
            
        Returns:
            bool: True if successfully removed, False otherwise.
        """
        file_path = self.queue_dir / batch_id
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete processed batch {batch_id}: {e}")
            return False

    def _get_sorted_queue_files(self) -> list[Path]:
        """Returns a list of .json.gz files sorted in chronological FIFO order."""
        return sorted([
            f for f in self.queue_dir.glob("batch_*.json.gz") 
            if not f.name.endswith(".tmp")
        ])

    def _quarantine_file(self, file_path: Path) -> None:
        """Moves corrupted files to a subfolder to unblock queue operations."""
        quarantine_dir = self.queue_dir / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        try:
            os.replace(file_path, quarantine_dir / file_path.name)
            logger.warning(f"Moved corrupted file {file_path.name} to quarantine.")
        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path.name}: {e}")

    def __len__(self) -> int:
        """Returns the number of pending batches in the queue."""
        return len(self._get_sorted_queue_files())