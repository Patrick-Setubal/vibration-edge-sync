"""
Ring Buffer module.
Maintains a bounded rolling window of sensor data.
Ensures O(1) time complexity for appending and strict bounded space complexity.
"""

from collections import deque
from typing import Tuple, List

class RingBuffer:
    """
    A memory-bounded circular buffer based on Python's native collections.deque.
    When the maximum length is reached, the oldest sample is automatically 
    and efficiently evicted at the C-level, preventing memory leaks.
    """

    def __init__(self, max_len: int):
        """
        Args:
            max_len: The maximum number of samples to keep in memory 
                     (e.g., 1000 samples for a 1-second window at 1kHz).
        """
        self.max_len = max_len
        # The deque handles the bounded memory logic natively
        self.buffer: deque = deque(maxlen=max_len)

    def append(self, sample: Tuple[float, float, bool]) -> None:
        """
        Adds a new sample to the right side of the buffer.
        If the buffer is full, the leftmost (oldest) element is discarded.
        
        Args:
            sample: A tuple containing (timestamp, acceleration, ground_truth_anomaly).
        """
        self.buffer.append(sample)

    def get_window(self) -> List[Tuple[float, float, bool]]:
        """
        Returns a snapshot of the current window.
        
        Returns:
            A list containing the buffered tuples in chronological order.
        """
        # Casting to list creates a snapshot copy so the processing thread 
        # can iterate safely even if the original buffer gets updated.
        return list(self.buffer)

    def is_full(self) -> bool:
        """
        Checks if the buffer has filled up to its maximum capacity.
        Useful to wait before processing the first window.
        """
        return len(self.buffer) == self.max_len

    def __len__(self) -> int:
        """Allows using len(ring_buffer)."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clears all items from the buffer."""
        self.buffer.clear()